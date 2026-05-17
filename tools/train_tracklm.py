#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from generative_tracking.config import load_config
from generative_tracking.data import SequenceWindowDataset, tracklm_collate
from generative_tracking.model import TrackLMRS
from generative_tracking.runtime import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train minimal TrackLM-RS prototype")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None, help="Optional global step limit for quick debug runs.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
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


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    cfg: dict,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    best_total: float,
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
    torch.save(
        {
            "iter": global_step,
            "global_step": global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "best_total": best_total,
            "trainable_only": trainable_only,
            "model_state": state_dict,
            "cfg": to_plain(cfg),
        },
        path,
    )


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
    model = TrackLMRS(cfg).to(device)
    steps_per_epoch = len(loader)
    max_iters = int(cfg.train.get("max_iters", 0))
    epochs = int(cfg.train.get("epochs", 1))
    total_steps = epochs * steps_per_epoch
    if max_iters > 0:
        total_steps = min(total_steps, max_iters)
    print(
        f"device={device} samples={len(dataset)} batch_size={int(cfg.train.batch_size)} "
        f"steps_per_epoch={steps_per_epoch} epochs={epochs} total_steps={total_steps} lr={float(cfg.train.lr)}",
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
                    "L_det",
                    "L_cls",
                    "L_center",
                    "L_size",
                    "L_yaw",
                    "L_embed",
                    "total",
                    "match_count",
                    "track_cls_acc",
                ]
            )
    model.train()
    global_step = start_global_step
    stop_training = False
    completed_epochs = start_global_step // steps_per_epoch
    best_total = float("inf")
    best_path = checkpoint_dir / "best.pth"
    for epoch in range(completed_epochs + 1, epochs + 1):
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
                "L_det": float(out["L_det"].detach().cpu()),
                "L_cls": float(out["L_cls"].detach().cpu()),
                "L_center": float(out["L_center"].detach().cpu()),
                "L_size": float(out["L_size"].detach().cpu()),
                "L_yaw": float(out["L_yaw"].detach().cpu()),
                "L_embed": float(out["L_embed"].detach().cpu()),
                "total": float(out["loss"].detach().cpu()),
                "match_count": float(out["match_count"].detach().cpu()),
                "track_cls_acc": float(out["track_cls_acc"].detach().cpu()),
            }
            if metrics["match_count"] > 0 and metrics["total"] < best_total:
                best_total = metrics["total"]
                save_checkpoint(
                    best_path,
                    model=model,
                    cfg=cfg,
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    global_step=global_step,
                    best_total=best_total,
                    trainable_only=bool(cfg.train.get("save_trainable_only", True)),
                )
                print(f"saved best checkpoint iter={global_step} total={best_total:.4f} path={best_path}", flush=True)
            if global_step % int(cfg.train.log_interval) == 0:
                with log_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch,
                            step_in_epoch,
                            global_step,
                            f"{metrics['L_det']:.6f}",
                            f"{metrics['L_cls']:.6f}",
                            f"{metrics['L_center']:.6f}",
                            f"{metrics['L_size']:.6f}",
                            f"{metrics['L_yaw']:.6f}",
                            f"{metrics['L_embed']:.6f}",
                            f"{metrics['total']:.6f}",
                            f"{metrics['match_count']:.0f}",
                            f"{metrics['track_cls_acc']:.6f}",
                        ]
                    )
                print(
                    f"epoch={epoch}/{epochs} step={step_in_epoch}/{steps_per_epoch} iter={global_step} "
                    f"L_det={metrics['L_det']:.4f} "
                    f"L_cls={metrics['L_cls']:.4f} "
                    f"L_center={metrics['L_center']:.4f} "
                    f"L_size={metrics['L_size']:.4f} "
                    f"L_yaw={metrics['L_yaw']:.4f} "
                    f"L_embed={metrics['L_embed']:.4f} "
                    f"total={metrics['total']:.4f} "
                    f"match_count={metrics['match_count']:.0f} "
                    f"track_cls_acc={metrics['track_cls_acc']:.4f}",
                    flush=True,
                )
        if stop_training:
            break

    print(f"finished training iter={global_step} best_total={best_total:.4f} checkpoint={best_path}", flush=True)


if __name__ == "__main__":
    main()
