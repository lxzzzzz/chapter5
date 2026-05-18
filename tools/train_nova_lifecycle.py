#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from generative_tracking.config import load_config
from generative_tracking.evaluator import evaluate_tracking_json
from generative_tracking.nova_data import NOVALifecycleDataset, nova_collate
from generative_tracking.nova_model import NOVALifecycleModel
from generative_tracking.nova_runtime import run_nova_lifecycle_tracking
from generative_tracking.runtime import select_device
from train_nova_association import is_better, load_checkpoint, save_checkpoint, to_plain

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NOVA V2 lifecycle tracker")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=50)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save_full_state", action="store_true")
    parser.add_argument("--eval_score_thresh", type=float, default=None)
    parser.add_argument("--association_threshold", type=float, default=None)
    return parser.parse_args()


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def append_log(path: Path, row: dict[str, float | int]) -> None:
    fields = ["epoch", "global_step", "L_action", "L_quality", "total", "action_acc", "positive_recall", "positive_count"]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, 0) for key in fields})


def run_validation(
    *,
    cfg: Any,
    model: torch.nn.Module,
    device: torch.device,
    epoch: int,
    global_step: int,
    output_dir: Path,
    progress_interval: int,
    use_tqdm: bool,
) -> dict[str, float]:
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    outputs, info_path = run_nova_lifecycle_tracking(
        cfg=cfg,
        model=model,
        device=device,
        split="val",
        max_frames=int(cfg.train.get("eval_max_frames", 0)),
        progress_interval=progress_interval,
        use_tqdm=use_tqdm,
        desc=f"nova lifecycle val epoch {epoch}",
    )
    result_path = validation_dir / f"epoch_{epoch:03d}_tracking_results.json"
    metrics_path = validation_dir / f"epoch_{epoch:03d}_metrics.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    metrics = evaluate_tracking_json(
        result_path,
        info_path,
        class_names=list(cfg.dataset.class_names),
        iou_threshold=float(cfg.evaluator.iou_threshold),
        output_path=metrics_path,
    )
    metrics["epoch"] = float(epoch)
    metrics["global_step"] = float(global_step)
    with (validation_dir / "latest_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.max_iters is not None:
        cfg.train.max_iters = args.max_iters
    if args.resume is not None:
        cfg.train.resume = args.resume
    if args.save_full_state:
        cfg.train.save_trainable_only = False
    if args.eval_score_thresh is not None:
        cfg.eval.score_thresh = float(args.eval_score_thresh)
    if args.association_threshold is not None:
        cfg.nova.association_threshold = float(args.association_threshold)

    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    device = select_device(str(cfg.device))

    dataset = NOVALifecycleDataset(cfg, split="train")
    if len(dataset) == 0:
        raise RuntimeError("NOVA lifecycle training dataset has zero samples. Check detection cache and GT info paths.")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=nova_collate,
        drop_last=False,
    )
    model = NOVALifecycleModel(cfg).to(device)
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=float(cfg.train.lr))

    output_dir = Path(cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(to_plain(cfg), f, indent=2)

    epochs = int(cfg.train.get("epochs", 1))
    steps_per_epoch = len(loader)
    max_iters = int(cfg.train.get("max_iters", 0))
    total_steps = epochs * steps_per_epoch if max_iters <= 0 else min(epochs * steps_per_epoch, max_iters)
    eval_interval_epochs = max(1, int(cfg.train.get("eval_interval_epochs", 3)))
    save_interval_epochs = max(1, int(cfg.train.get("save_interval_epochs", eval_interval_epochs)))
    best_metric_name = str(cfg.train.get("best_metric", "mota"))
    best_mode = str(cfg.train.get("best_mode", "max")).lower()
    best_metric_value: float | None = None
    global_step = 0
    start_epoch = 0
    if str(cfg.train.get("resume", "")):
        global_step, start_epoch = load_checkpoint(Path(cfg.train.resume), model, optimizer, device)
        print(f"resumed checkpoint={cfg.train.resume} epoch={start_epoch} global_step={global_step}", flush=True)

    print(
        f"device={device} lifecycle_samples={len(dataset)} batch_size={int(cfg.train.batch_size)} "
        f"steps_per_epoch={steps_per_epoch} epochs={epochs} total_steps={total_steps}",
        flush=True,
    )
    use_tqdm = tqdm is not None
    progress_interval = max(1, int(args.progress_interval))
    model.train()
    stop = False
    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_target_steps = min(steps_per_epoch, max(0, total_steps - (epoch - 1) * steps_per_epoch))
        epoch_progress = tqdm(total=epoch_target_steps, desc=f"nova lifecycle train epoch {epoch}/{epochs}", dynamic_ncols=True, leave=True) if use_tqdm and epoch_target_steps > 0 else None
        for step_in_epoch, batch in enumerate(loader, start=1):
            if global_step >= total_steps:
                stop = True
                break
            global_step += 1
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            out["loss"].backward()
            optimizer.step()
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "L_action": float(out["L_action"].detach().cpu()),
                "L_quality": float(out["L_quality"].detach().cpu()),
                "total": float(out["loss"].detach().cpu()),
                "action_acc": float(out["action_acc"].detach().cpu()),
                "positive_recall": float(out["positive_recall"].detach().cpu()),
                "positive_count": float(out["positive_count"].detach().cpu()),
            }
            if global_step % int(cfg.train.log_interval) == 0:
                append_log(output_dir / "training_log.csv", row)
            if epoch_progress is not None:
                epoch_progress.update(1)
                epoch_progress.set_postfix(step=f"{step_in_epoch}/{steps_per_epoch}", total=f"{row['total']:.4f}", acc=f"{row['action_acc']:.3f}")
            elif global_step == 1 or global_step % progress_interval == 0 or global_step == total_steps:
                print(f"nova lifecycle train iter={global_step}/{total_steps} epoch={epoch}/{epochs} loss={row['total']:.4f} acc={row['action_acc']:.4f}", flush=True)
        if epoch_progress is not None:
            epoch_progress.close()

        epoch_completed = not stop
        if epoch_completed and (epoch % save_interval_epochs == 0 or epoch == epochs):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                trainable_only=bool(cfg.train.get("save_trainable_only", True)),
            )
            save_checkpoint(
                checkpoint_dir / "latest.pth",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                trainable_only=bool(cfg.train.get("save_trainable_only", True)),
            )
        if epoch_completed and (epoch % eval_interval_epochs == 0 or (bool(cfg.train.get("eval_on_final", True)) and epoch == epochs)):
            was_training = model.training
            model.eval()
            metrics = run_validation(cfg=cfg, model=model, device=device, epoch=epoch, global_step=global_step, output_dir=output_dir, progress_interval=progress_interval, use_tqdm=use_tqdm)
            if was_training:
                model.train()
            metric_value = float(metrics.get(best_metric_name, float("nan")))
            if np.isfinite(metric_value) and is_better(metric_value, best_metric_value, best_mode):
                best_metric_value = metric_value
                save_checkpoint(
                    checkpoint_dir / "best.pth",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric_name=best_metric_name,
                    best_metric_value=best_metric_value,
                    trainable_only=bool(cfg.train.get("save_trainable_only", True)),
                )
            print(
                f"nova lifecycle validation epoch={epoch} ap_3d_iou_0_50={float(metrics.get('ap_3d_iou_0_50', 0.0)):.4f} "
                f"mota={float(metrics.get('mota', 0.0)):.4f} id_switches={float(metrics.get('id_switches', 0.0)):.0f}",
                flush=True,
            )
        if stop:
            break
    if global_step > 0:
        save_checkpoint(
            checkpoint_dir / "latest.pth",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=min(epoch, epochs) if "epoch" in locals() else start_epoch,
            global_step=global_step,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            trainable_only=bool(cfg.train.get("save_trainable_only", True)),
        )
    print(f"finished nova lifecycle training iter={global_step} best_{best_metric_name}={best_metric_value}", flush=True)


if __name__ == "__main__":
    main()
