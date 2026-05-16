#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

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
    return parser.parse_args()


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    if args.max_iters is not None:
        cfg.train.max_iters = args.max_iters
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

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

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    model.train()
    for step in range(1, int(cfg.train.max_iters) + 1):
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
            print(
                f"iter={step} "
                f"L_det={float(out['L_det'].detach().cpu()):.4f} "
                f"L_id={float(out['L_id'].detach().cpu()):.4f} "
                f"total={float(out['loss'].detach().cpu()):.4f} "
                f"pointer_acc={float(out['pointer_acc'].detach().cpu()):.4f} "
                f"new_acc={float(out['new_acc'].detach().cpu()):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
