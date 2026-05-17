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
from generative_tracking.track_manager import TrackEmbeddingManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate minimal TrackLM-RS prototype")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--eval_metrics", action="store_true")
    return parser.parse_args()


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def load_model_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict, strict=not bool(checkpoint.get("trainable_only", False)))
    return int(checkpoint.get("iter", 0)) if isinstance(checkpoint, dict) else 0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    device = select_device(str(cfg.device))
    dataset = SequenceWindowDataset(cfg, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=tracklm_collate)
    model = TrackLMRS(cfg).to(device)
    ckpt_arg = args.ckpt if args.ckpt is not None else str(cfg.eval.get("checkpoint", ""))
    ckpt_path = Path(ckpt_arg) if ckpt_arg else Path(cfg.output_dir) / "checkpoints" / "best.pth"
    if ckpt_path.exists():
        ckpt_iter = load_model_checkpoint(ckpt_path, model, device)
        print(f"loaded checkpoint={ckpt_path} iter={ckpt_iter}")
    elif args.ckpt is not None or ckpt_arg:
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    else:
        print(f"warning: checkpoint not found at {ckpt_path}; evaluating randomly initialized model")
    model.eval()
    manager = TrackEmbeddingManager(
        max_lost_frames=int(cfg.eval.max_lost_frames),
        match_threshold=float(cfg.eval.embedding_match_threshold),
    )
    outputs = []
    current_sequence = None
    with torch.inference_mode():
        for frame_count, batch_cpu in enumerate(loader, start=1):
            if args.max_frames is not None and frame_count > args.max_frames:
                break
            sequence_id = batch_cpu["sequence_id"][0]
            if sequence_id != current_sequence:
                manager.reset()
                current_sequence = sequence_id
            batch = move_to_device(batch_cpu, device)
            out = model(batch)
            logits = out["pred_logits"][0].softmax(dim=-1)
            no_object = logits.shape[-1] - 1
            pred_scores, pred_classes = logits[:, :no_object].max(dim=-1)
            keep = pred_scores.ge(float(cfg.eval.score_thresh))
            result = manager.update(
                sequence_id=sequence_id,
                frame_id=batch_cpu["frame_id"][0],
                boxes=out["pred_boxes"][0].detach().cpu(),
                class_ids=pred_classes.detach().cpu(),
                class_names=list(cfg.dataset.class_names),
                scores=pred_scores.detach().cpu(),
                embeddings=out["pred_track_embeds"][0].detach().cpu(),
                valid_mask=keep.detach().cpu(),
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
