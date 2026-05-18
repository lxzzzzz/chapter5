#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from generative_tracking.ab3dmot_evaluator import evaluate_ab3dmot_json
from generative_tracking.config import load_config
from generative_tracking.evaluator import evaluate_tracking_json
from generative_tracking.nova_model import NOVALifecycleModel
from generative_tracking.nova_runtime import run_nova_lifecycle_tracking
from generative_tracking.runtime import select_device
from eval_nova_tracking import load_model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NOVA V2 lifecycle tracker")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--eval_metrics", action="store_true")
    parser.add_argument("--eval_ab3dmot", action="store_true")
    parser.add_argument("--ab3dmot_output", default=None)
    parser.add_argument("--ab3dmot_recall_points", type=int, default=40)
    parser.add_argument("--eval_score_thresh", type=float, default=None)
    parser.add_argument("--association_threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    if args.eval_score_thresh is not None:
        cfg.eval.score_thresh = float(args.eval_score_thresh)
    if args.association_threshold is not None:
        cfg.nova.association_threshold = float(args.association_threshold)
    device = select_device(str(cfg.device))
    model = NOVALifecycleModel(cfg).to(device)
    ckpt_arg = args.ckpt if args.ckpt is not None else str(cfg.eval.get("checkpoint", ""))
    ckpt_path = Path(ckpt_arg) if ckpt_arg else Path(cfg.output_dir) / "checkpoints" / "best.pth"
    if ckpt_path.exists():
        ckpt_iter = load_model_checkpoint(ckpt_path, model, device)
        print(f"loaded checkpoint={ckpt_path} iter={ckpt_iter}", flush=True)
    elif args.ckpt is not None or ckpt_arg:
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    else:
        print(f"warning: checkpoint not found at {ckpt_path}; evaluating randomly initialized model", flush=True)

    outputs, info_path = run_nova_lifecycle_tracking(
        cfg=cfg,
        model=model,
        device=device,
        split=str(args.split),
        max_frames=0 if args.max_frames is None else int(args.max_frames),
        progress_interval=max(1, int(args.progress_interval)),
        use_tqdm=True,
        desc=f"nova lifecycle eval {args.split}",
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
    if args.eval_ab3dmot:
        output_ab3d = args.ab3dmot_output or str(Path(cfg.output_dir) / "tracking_metrics_ab3dmot.json")
        ab3d = evaluate_ab3dmot_json(
            output_path,
            info_path,
            class_names=list(cfg.dataset.class_names),
            iou_threshold=float(cfg.evaluator.iou_threshold),
            recall_points=int(args.ab3dmot_recall_points),
            output_path=output_ab3d,
        )
        print(
            "ab3dmot "
            f"sAMOTA={float(ab3d['sAMOTA']):.4f} "
            f"AMOTA={float(ab3d['AMOTA']):.4f} "
            f"AMOTP={float(ab3d['AMOTP']):.4f} "
            f"MOTA={float(ab3d['MOTA']):.4f} "
            f"MOTP={float(ab3d['MOTP']):.4f} "
            f"IDS={float(ab3d['IDS']):.0f} "
            f"FRAG={float(ab3d['FRAG']):.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
