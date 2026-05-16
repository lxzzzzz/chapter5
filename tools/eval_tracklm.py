#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_tracking.config import load_config
from generative_tracking.data import SequenceWindowDataset, tracklm_collate
from generative_tracking.evaluator import evaluate_tracking_json
from generative_tracking.model import TrackLMRS
from generative_tracking.runtime import select_device
from generative_tracking.track_manager import TrackManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate minimal TrackLM-RS prototype")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--eval_metrics", action="store_true")
    return parser.parse_args()


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    device = select_device(str(cfg.device))
    dataset = SequenceWindowDataset(cfg, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=tracklm_collate)
    model = TrackLMRS(cfg).to(device)
    model.eval()
    manager = TrackManager(max_lost_frames=int(cfg.eval.max_lost_frames))
    outputs = []
    current_sequence = None
    with torch.no_grad():
        for frame_count, batch_cpu in enumerate(loader, start=1):
            if args.max_frames is not None and frame_count > args.max_frames:
                break
            sequence_id = batch_cpu["sequence_id"][0]
            if sequence_id != current_sequence:
                manager.reset()
                current_sequence = sequence_id
            batch = move_to_device(batch_cpu, device)
            out = model(batch)
            logits = out["logits"][0]
            valid_current = batch_cpu["current_valid_mask"][0]
            n_current = int(valid_current.sum().item())
            preds = logits[:n_current].argmax(dim=-1).cpu()
            result = manager.update(
                sequence_id=sequence_id,
                frame_id=batch_cpu["frame_id"][0],
                boxes=batch_cpu["current_boxes"][0, :n_current],
                class_names=batch_cpu["current_class_names"][0][:n_current],
                scores=batch_cpu["current_scores"][0, :n_current],
                pointer_preds=preds,
                history_track_ids=batch_cpu["history_track_ids"][0],
                history_valid_mask=batch_cpu["history_valid_mask"][0],
            )
            outputs.append(result)

    output_path = Path(args.output or Path(cfg.output_dir) / "tracking_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    print(f"wrote {len(outputs)} frames to {output_path}")
    if args.eval_metrics or bool(cfg.evaluator.get("enabled", False)):
        metrics = evaluate_tracking_json(
            output_path,
            dataset.info_path,
            class_names=list(cfg.dataset.class_names),
            iou_threshold=float(cfg.evaluator.iou_threshold),
            output_path=cfg.evaluator.metrics_path,
        )
        print(
            "metrics "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"mota_like={metrics['mota_like']:.4f} "
            f"id_switches={metrics['id_switches']:.0f}"
        )


if __name__ == "__main__":
    main()
