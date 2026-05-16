#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from generative_tracking.config import load_config, resolve_info_path
from generative_tracking.evaluator import evaluate_tracking_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TrackLM-RS tracking JSON against tracking infos")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default=None)
    parser.add_argument("--iou_threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    metrics = evaluate_tracking_json(
        args.result_json,
        resolve_info_path(cfg, args.split),
        class_names=list(cfg.dataset.class_names),
        iou_threshold=float(args.iou_threshold if args.iou_threshold is not None else cfg.evaluator.iou_threshold),
        output_path=args.output or cfg.evaluator.metrics_path,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
