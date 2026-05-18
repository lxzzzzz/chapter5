#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from generative_tracking.ab3dmot_evaluator import evaluate_ab3dmot_json
from generative_tracking.config import load_config, resolve_info_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tracking JSON with AB3DMOT-style 3D MOT metrics")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default=None)
    parser.add_argument("--iou_threshold", type=float, default=None)
    parser.add_argument("--recall_points", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    metrics = evaluate_ab3dmot_json(
        args.result_json,
        resolve_info_path(cfg, args.split),
        class_names=list(cfg.dataset.class_names),
        iou_threshold=float(args.iou_threshold if args.iou_threshold is not None else cfg.evaluator.iou_threshold),
        recall_points=int(args.recall_points),
        output_path=args.output,
    )
    printable = {key: value for key, value in metrics.items() if key != "curve"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
