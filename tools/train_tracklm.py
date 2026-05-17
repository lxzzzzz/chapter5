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
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
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
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iter": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
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
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint.get("iter", 0))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    if args.max_iters is not None:
        cfg.train.max_iters = args.max_iters
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.save_interval is not None:
        cfg.train.save_interval = args.save_interval
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
    print(f"device={device} samples={len(dataset)}", flush=True)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.train.lr))

    output_dir = Path(cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    resume_path = str(cfg.train.get("resume", ""))
    if resume_path:
        start_step = load_checkpoint(Path(resume_path), model=model, optimizer=optimizer, device=device)
        print(f"resumed checkpoint={resume_path} iter={start_step}", flush=True)

    log_path = output_dir / "training_log.csv"
    if start_step == 0 or not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iter",
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
    iterator = iter(loader)
    model.train()
    for step in range(start_step + 1, int(cfg.train.max_iters) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        out["loss"].backward()
        optimizer.step()
        if step % int(cfg.train.log_interval) == 0:
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
            with log_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        step,
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
                f"iter={step} "
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
        if int(cfg.train.save_interval) > 0 and step % int(cfg.train.save_interval) == 0:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_iter_{step}.pth",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                step=step,
            )
            save_checkpoint(
                checkpoint_dir / "latest.pth",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                step=step,
            )
            print(f"saved checkpoint iter={step} dir={checkpoint_dir}", flush=True)

    final_step = int(cfg.train.max_iters)
    save_checkpoint(
        checkpoint_dir / f"checkpoint_iter_{final_step}.pth",
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        step=final_step,
    )
    save_checkpoint(
        checkpoint_dir / "latest.pth",
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        step=final_step,
    )
    save_checkpoint(
        checkpoint_dir / "final.pth",
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        step=final_step,
    )
    print(f"saved final checkpoint iter={final_step} dir={checkpoint_dir}", flush=True)


if __name__ == "__main__":
    main()
