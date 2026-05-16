#!/usr/bin/env python
from __future__ import annotations

import argparse

from generative_tracking.chapter3_export import export_chapter3_detections
from generative_tracking.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export chapter3 detector predictions for TrackLM-RS")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector_ckpt", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    if args.detector_ckpt is not None:
        cfg.detector_ckpt = args.detector_ckpt
    output = export_chapter3_detections(
        cfg,
        split=args.split,
        output_path=args.output,
        batch_size=args.batch_size,
        workers=args.workers,
        max_frames=args.max_frames,
    )
    print(f"wrote detector predictions to {output}")


if __name__ == "__main__":
    main()
