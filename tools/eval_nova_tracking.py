#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from generative_tracking.config import load_config
from generative_tracking.evaluator import evaluate_tracking_json
from generative_tracking.nova_model import NOVAAssociationModel
from generative_tracking.nova_runtime import run_nova_tracking
from generative_tracking.runtime import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NOVA-style association tracker")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--eval_metrics", action="store_true")
    return parser.parse_args()


def load_model_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=not bool(isinstance(checkpoint, dict) and checkpoint.get("trainable_only", False)))
    return int(checkpoint.get("iter", 0)) if isinstance(checkpoint, dict) else 0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    device = select_device(str(cfg.device))
    model = NOVAAssociationModel(cfg).to(device)
    ckpt_arg = args.ckpt if args.ckpt is not None else str(cfg.eval.get("checkpoint", ""))
    ckpt_path = Path(ckpt_arg) if ckpt_arg else Path(cfg.output_dir) / "checkpoints" / "best.pth"
    if ckpt_path.exists():
        ckpt_iter = load_model_checkpoint(ckpt_path, model, device)
        print(f"loaded checkpoint={ckpt_path} iter={ckpt_iter}", flush=True)
    elif args.ckpt is not None or ckpt_arg:
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    else:
        print(f"warning: checkpoint not found at {ckpt_path}; evaluating randomly initialized model", flush=True)

    outputs, info_path = run_nova_tracking(
        cfg=cfg,
        model=model,
        device=device,
        split=str(args.split),
        max_frames=0 if args.max_frames is None else int(args.max_frames),
        progress_interval=max(1, int(args.progress_interval)),
    )
    output_path = Path(args.output or Path(cfg.output_dir) / "tracking_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    print(f"wrote {len(outputs)} frames to {output_path}", flush=True)

    if args.eval_metrics or bool(cfg.evaluator.get("enabled", False)):
        metrics_path = Path(cfg.evaluator.get("metrics_path", Path(cfg.output_dir) / "tracking_metrics.json"))
        metrics = evaluate_tracking_json(
            output_path,
            info_path,
            class_names=list(cfg.dataset.class_names),
            iou_threshold=float(cfg.evaluator.iou_threshold),
            output_path=metrics_path,
        )
        print(
            "metrics "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"ap_3d_iou_0_50={metrics['ap_3d_iou_0_50']:.4f} "
            f"mota={metrics['mota']:.4f} "
            f"id_switches={metrics['id_switches']:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
