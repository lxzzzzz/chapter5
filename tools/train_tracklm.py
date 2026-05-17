#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from generative_tracking.config import load_config
from generative_tracking.data import SequenceWindowDataset, tracklm_collate
from generative_tracking.evaluator import evaluate_tracking_json
from generative_tracking.model import TrackLMRS
from generative_tracking.runtime import select_device
from generative_tracking.track_manager import TrackEmbeddingManager

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train minimal TrackLM-RS prototype")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None, help="Optional global step limit for quick debug runs.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--progress_interval", type=int, default=10)
    parser.add_argument("--save_full_state", action="store_true")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    return value


def make_progress(iterable: Iterable, *, total: int, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)
    return iterable


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    cfg: dict,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    best_total: float,
    best_metric_name: str = "",
    best_metric_value: float | None = None,
    trainable_only: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if trainable_only:
        state_dict = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
    else:
        state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    checkpoint = {
        "iter": global_step,
        "global_step": global_step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "best_total": best_total,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "trainable_only": trainable_only,
        "model_state": state_dict,
        "cfg": to_plain(cfg),
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=not bool(checkpoint.get("trainable_only", False)))
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    global_step = int(checkpoint.get("global_step", checkpoint.get("iter", 0)))
    epoch = int(checkpoint.get("epoch", 0))
    return global_step, epoch


def run_validation(
    *,
    model: TrackLMRS,
    cfg: Any,
    device: torch.device,
    val_loader: DataLoader,
    val_info_path: Path,
    epoch: int,
    global_step: int,
    output_dir: Path,
    max_frames: int,
    use_tqdm: bool,
    progress_interval: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    manager = TrackEmbeddingManager(
        max_lost_frames=int(cfg.eval.max_lost_frames),
        match_threshold=float(cfg.eval.embedding_match_threshold),
    )
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    result_path = validation_dir / f"epoch_{epoch:03d}_tracking_results.json"
    metrics_path = validation_dir / f"epoch_{epoch:03d}_metrics.json"
    total_frames = len(val_loader.dataset) if max_frames <= 0 else min(len(val_loader.dataset), max_frames)
    outputs = []
    current_sequence = None
    progress_bar = tqdm(total=total_frames, desc=f"val epoch {epoch}", dynamic_ncols=True, leave=True) if use_tqdm and tqdm is not None else None
    with torch.inference_mode():
        for frame_count, batch_cpu in enumerate(val_loader, start=1):
            if max_frames > 0 and frame_count > max_frames:
                break
            sequence_id = batch_cpu["sequence_id"][0]
            if sequence_id != current_sequence:
                manager.reset()
                current_sequence = sequence_id
            batch = move_to_device(batch_cpu, device)
            out = model(batch)
            logits = out["pred_logits"][0].softmax(dim=-1)
            pred_scores = logits[:, 1]
            pred_classes = torch.zeros_like(pred_scores, dtype=torch.long)
            keep = pred_scores.ge(float(cfg.eval.score_thresh))
            outputs.append(
                manager.update(
                    sequence_id=sequence_id,
                    frame_id=batch_cpu["frame_id"][0],
                    boxes=out["pred_boxes"][0].detach().cpu(),
                    class_ids=pred_classes.detach().cpu(),
                    class_names=list(cfg.dataset.class_names),
                    scores=pred_scores.detach().cpu(),
                    embeddings=out["pred_track_embeds"][0].detach().cpu(),
                    valid_mask=keep.detach().cpu(),
                )
            )
            del keep, pred_classes, pred_scores, logits, out, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(frame=f"{frame_count}/{total_frames}")
            elif frame_count == 1 or frame_count % progress_interval == 0 or frame_count == total_frames:
                print(f"validation progress epoch={epoch} frames={frame_count}/{total_frames}", flush=True)
    if progress_bar is not None:
        progress_bar.close()

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    metrics = evaluate_tracking_json(
        result_path,
        val_info_path,
        class_names=list(cfg.dataset.class_names),
        iou_threshold=float(cfg.evaluator.iou_threshold),
        output_path=metrics_path,
    )
    metrics["epoch"] = float(epoch)
    metrics["global_step"] = float(global_step)
    latest_metrics_path = validation_dir / "latest_metrics.json"
    with latest_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    append_validation_log(validation_dir / "validation_metrics.csv", epoch, global_step, metrics)
    if was_training:
        model.train()
    return metrics


def append_validation_log(path: Path, epoch: int, global_step: int, metrics: dict[str, float]) -> None:
    fields = [
        "epoch",
        "global_step",
        "ap_3d_iou_0_50",
        "mota",
        "motp",
        "precision",
        "recall",
        "f1",
        "id_switches",
        "fragments",
        "mostly_tracked",
        "mostly_lost",
        "false_positive",
        "false_negative",
        "num_gt",
        "num_pred",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(fields)
        writer.writerow([epoch, global_step, *[f"{float(metrics.get(field, 0.0)):.6f}" for field in fields[2:]]])


def is_better_metric(value: float, best_value: float | None, mode: str) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return value < best_value
    return value > best_value


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.max_iters is not None:
        cfg.train.max_iters = args.max_iters
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.save_full_state:
        cfg.train.save_trainable_only = False
    if args.resume is not None:
        cfg.train.resume = args.resume

    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    device = select_device(str(cfg.device))

    dataset = SequenceWindowDataset(cfg, split="train")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=tracklm_collate,
        drop_last=False,
    )
    val_dataset = SequenceWindowDataset(cfg, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=tracklm_collate,
        drop_last=False,
    )
    model = TrackLMRS(cfg).to(device)
    steps_per_epoch = len(loader)
    max_iters = int(cfg.train.get("max_iters", 0))
    epochs = int(cfg.train.get("epochs", 1))
    eval_interval_epochs = max(1, int(cfg.train.get("eval_interval_epochs", 3)))
    save_interval_epochs = max(1, int(cfg.train.get("save_interval_epochs", eval_interval_epochs)))
    eval_on_final = bool(cfg.train.get("eval_on_final", True))
    eval_max_frames = int(cfg.train.get("eval_max_frames", 0))
    best_metric_name = str(cfg.train.get("best_metric", "mota"))
    best_mode = str(cfg.train.get("best_mode", "max")).lower()
    if best_mode not in {"max", "min"}:
        raise ValueError(f"Unsupported train.best_mode={best_mode!r}; expected 'max' or 'min'.")
    progress_interval = max(1, int(args.progress_interval))
    total_steps = epochs * steps_per_epoch
    if max_iters > 0:
        total_steps = min(total_steps, max_iters)
    use_tqdm = tqdm is not None
    print(
        f"device={device} samples={len(dataset)} batch_size={int(cfg.train.batch_size)} "
        f"steps_per_epoch={steps_per_epoch} val_samples={len(val_dataset)} epochs={epochs} "
        f"total_steps={total_steps} lr={float(cfg.train.lr)} eval_interval_epochs={eval_interval_epochs}",
        flush=True,
    )
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.train.lr))

    output_dir = Path(cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    start_global_step = 0
    start_epoch = 0
    resume_path = str(cfg.train.get("resume", ""))
    if resume_path:
        start_global_step, start_epoch = load_checkpoint(Path(resume_path), model=model, optimizer=optimizer, device=device)
        print(f"resumed checkpoint={resume_path} epoch={start_epoch} global_step={start_global_step}", flush=True)

    log_path = output_dir / "training_log.csv"
    if start_global_step == 0 or not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "step_in_epoch",
                    "global_step",
                    "L_obj",
                    "L_center",
                    "L_size",
                    "L_yaw",
                    "L_embed",
                    "total",
                    "match_count",
                    "objectness_acc",
                ]
            )
    model.train()
    global_step = start_global_step
    stop_training = False
    completed_epochs = start_global_step // steps_per_epoch
    best_total = float("inf")
    best_metric_value: float | None = None
    best_path = checkpoint_dir / "best.pth"
    overall_bar = make_progress(
        range(start_global_step, total_steps),
        total=total_steps,
        desc="train",
        enabled=use_tqdm,
    )
    for epoch in range(completed_epochs + 1, epochs + 1):
        epoch_target_steps = min(steps_per_epoch, max(0, total_steps - (epoch - 1) * steps_per_epoch))
        if not use_tqdm:
            print(f"train progress epoch={epoch}/{epochs} steps={epoch_target_steps}/{steps_per_epoch}", flush=True)
        for step_in_epoch, batch in enumerate(loader, start=1):
            absolute_step = (epoch - 1) * steps_per_epoch + step_in_epoch
            if absolute_step <= start_global_step:
                continue
            if global_step >= total_steps:
                stop_training = True
                break
            global_step += 1
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            out["loss"].backward()
            optimizer.step()
            metrics = {
                "L_obj": float(out["L_obj"].detach().cpu()),
                "L_center": float(out["L_center"].detach().cpu()),
                "L_size": float(out["L_size"].detach().cpu()),
                "L_yaw": float(out["L_yaw"].detach().cpu()),
                "L_embed": float(out["L_embed"].detach().cpu()),
                "total": float(out["loss"].detach().cpu()),
                "match_count": float(out["match_count"].detach().cpu()),
                "objectness_acc": float(out["objectness_acc"].detach().cpu()),
            }
            if metrics["match_count"] > 0 and metrics["total"] < best_total:
                best_total = metrics["total"]
            if use_tqdm:
                overall_bar.update(1)
                overall_bar.set_postfix(
                    epoch=f"{epoch}/{epochs}",
                    step=f"{step_in_epoch}/{steps_per_epoch}",
                    total=f"{metrics['total']:.4f}",
                    obj=f"{metrics['L_obj']:.4f}",
                )
            elif global_step == 1 or global_step % progress_interval == 0 or global_step == total_steps:
                print(
                    f"train progress iter={global_step}/{total_steps} epoch={epoch}/{epochs} "
                    f"step={step_in_epoch}/{steps_per_epoch}",
                    flush=True,
                )
            if global_step % int(cfg.train.log_interval) == 0:
                with log_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch,
                            step_in_epoch,
                            global_step,
                            f"{metrics['L_obj']:.6f}",
                            f"{metrics['L_center']:.6f}",
                            f"{metrics['L_size']:.6f}",
                            f"{metrics['L_yaw']:.6f}",
                            f"{metrics['L_embed']:.6f}",
                            f"{metrics['total']:.6f}",
                            f"{metrics['match_count']:.0f}",
                            f"{metrics['objectness_acc']:.6f}",
                        ]
                    )
                msg = (
                    f"epoch={epoch}/{epochs} step={step_in_epoch}/{steps_per_epoch} iter={global_step} "
                    f"L_obj={metrics['L_obj']:.4f} "
                    f"L_center={metrics['L_center']:.4f} "
                    f"L_size={metrics['L_size']:.4f} "
                    f"L_yaw={metrics['L_yaw']:.4f} "
                    f"L_embed={metrics['L_embed']:.4f} "
                    f"total={metrics['total']:.4f} "
                    f"match_count={metrics['match_count']:.0f} "
                    f"objectness_acc={metrics['objectness_acc']:.4f}"
                )
                if use_tqdm:
                    overall_bar.write(msg)
                else:
                    print(msg, flush=True)
        epoch_completed = not stop_training
        should_checkpoint = epoch_completed and (epoch % save_interval_epochs == 0)
        should_validate = epoch_completed and (epoch % eval_interval_epochs == 0 or (eval_on_final and epoch == epochs))
        if should_checkpoint:
            epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
            save_checkpoint(
                epoch_path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                step_in_epoch=min(step_in_epoch, steps_per_epoch) if "step_in_epoch" in locals() else 0,
                global_step=global_step,
                best_total=best_total,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                trainable_only=bool(cfg.train.get("save_trainable_only", True)),
            )
            message = f"saved epoch checkpoint epoch={epoch} iter={global_step} path={epoch_path}"
            if use_tqdm:
                overall_bar.write(message)
            else:
                print(message, flush=True)
        if should_validate:
            if use_tqdm:
                overall_bar.write(f"running validation epoch={epoch}")
            else:
                print(f"running validation epoch={epoch}", flush=True)
            val_metrics = run_validation(
                model=model,
                cfg=cfg,
                device=device,
                val_loader=val_loader,
                val_info_path=val_dataset.info_path,
                epoch=epoch,
                global_step=global_step,
                output_dir=output_dir,
                max_frames=eval_max_frames,
                use_tqdm=use_tqdm,
                progress_interval=progress_interval,
            )
            metric_value = float(val_metrics.get(best_metric_name, float("nan")))
            if np.isfinite(metric_value) and is_better_metric(metric_value, best_metric_value, best_mode):
                best_metric_value = metric_value
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    epoch=epoch,
                    step_in_epoch=min(step_in_epoch, steps_per_epoch) if "step_in_epoch" in locals() else 0,
                    global_step=global_step,
                    best_total=best_total,
                    best_metric_name=best_metric_name,
                    best_metric_value=best_metric_value,
                    trainable_only=bool(cfg.train.get("save_trainable_only", True)),
                )
                best_metrics_path = output_dir / "validation" / "best_metrics.json"
                with best_metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(val_metrics, f, indent=2)
                message = (
                    f"saved best checkpoint epoch={epoch} iter={global_step} "
                    f"{best_metric_name}={best_metric_value:.4f} path={best_path}"
                )
                if use_tqdm:
                    overall_bar.write(message)
                else:
                    print(message, flush=True)
            val_msg = (
                f"validation epoch={epoch} "
                f"ap_3d_iou_0_50={float(val_metrics.get('ap_3d_iou_0_50', 0.0)):.4f} "
                f"mota={float(val_metrics.get('mota', 0.0)):.4f} "
                f"motp={float(val_metrics.get('motp', 0.0)):.4f} "
                f"id_switches={float(val_metrics.get('id_switches', 0.0)):.0f}"
            )
            if use_tqdm:
                overall_bar.write(val_msg)
            else:
                print(val_msg, flush=True)
        if stop_training:
            break

    if use_tqdm:
        overall_bar.close()
    print(
        f"finished training iter={global_step} best_total={best_total:.4f} "
        f"best_{best_metric_name}={best_metric_value} checkpoint={best_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
