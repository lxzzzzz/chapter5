#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from generative_tracking.config import load_config, resolve_info_path
from generative_tracking.nova_data import filter_detection_frame, validate_detection_frame

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache VoxelNeXt detections for NOVA-style association")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--score_thresh", type=float, default=None)
    parser.add_argument("--max_dets_per_frame", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--progress_interval", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    score_thresh = float(args.score_thresh if args.score_thresh is not None else cfg.detection_cache.score_thresh)
    max_dets = int(args.max_dets_per_frame if args.max_dets_per_frame is not None else cfg.detection_cache.max_dets_per_frame)
    output_root = Path(args.output_dir or cfg.detection_cache.root)
    split_dir = output_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    detector = _build_detector(cfg, args.split)
    dataset = detector["dataset"]
    model = detector["model"]
    load_data_to_gpu = detector["load_data_to_gpu"]
    det_class_names = [str(name) for name in detector["class_names"]]
    target_class = str(cfg.detection_cache.get("class_name", "Car"))
    if target_class not in det_class_names:
        raise ValueError(f"Detector class_names={det_class_names} does not contain target class {target_class!r}")
    raw_target_label = det_class_names.index(target_class) + 1

    infos = _dataset_infos(dataset)
    total = len(dataset)
    use_tqdm = tqdm is not None
    progress = tqdm(range(total), total=total, desc=f"cache {args.split}", dynamic_ncols=True) if use_tqdm else range(total)
    manifest_frames: list[dict[str, Any]] = []
    for index in progress:
        raw = dataset[index]
        batch_dict = dataset.collate_batch([raw])
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            model_out = model(batch_dict)
        pred_dict = _first_pred_dict(model_out, batch_dict)
        info = infos[index] if index < len(infos) else raw
        frame = _prediction_to_frame(
            pred_dict,
            info,
            raw_target_label=raw_target_label,
            output_class_id=0,
        )
        frame = filter_detection_frame(frame, class_id=0, score_thresh=score_thresh, max_dets=max_dets)
        validate_detection_frame(frame)
        seq = _safe_path_part(frame["sequence_id"])
        frame_id = _safe_path_part(frame["frame_id"])
        frame_path = split_dir / seq / f"{frame_id}.pkl"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        with frame_path.open("wb") as f:
            pickle.dump(frame, f)
        manifest_frames.append(frame)
        if not use_tqdm and (index == 0 or (index + 1) % max(1, int(args.progress_interval)) == 0 or index + 1 == total):
            print(f"cache progress split={args.split} frames={index + 1}/{total}", flush=True)

    manifest_path = output_root / f"detections_{args.split}.pkl"
    with manifest_path.open("wb") as f:
        pickle.dump({"split": args.split, "frames": manifest_frames}, f)
    print(f"cached {len(manifest_frames)} frames to {split_dir}")
    print(f"wrote manifest {manifest_path}")


def _build_detector(cfg: Any, split: str) -> dict[str, Any]:
    chapter3_root = Path(cfg.chapter3_root)
    tools_dir = chapter3_root / "tools"
    for path in (str(tools_dir), str(chapter3_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from pcdet.config import cfg as det_cfg
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    old_cwd = Path.cwd()
    os.chdir(chapter3_root)
    try:
        det_cfg.clear()
        det_cfg.ROOT_DIR = chapter3_root.resolve()
        det_cfg.LOCAL_RANK = 0
        cfg_from_yaml_file(str(cfg.detector_cfg_file), det_cfg)
        info_path = Path(resolve_info_path(cfg, split))
        det_cfg.DATA_CONFIG.INFO_PATH["test"] = [str(info_path)]
    finally:
        os.chdir(old_cwd)

    logger = common_utils.create_logger(log_file=None, rank=0)
    dataset, _loader, _sampler = build_dataloader(
        dataset_cfg=det_cfg.DATA_CONFIG,
        class_names=det_cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        root_path=chapter3_root / det_cfg.DATA_CONFIG.DATA_PATH,
        workers=0,
        logger=logger,
        training=False,
    )
    model = build_network(det_cfg.MODEL, num_class=len(det_cfg.CLASS_NAMES), dataset=dataset)
    if not str(cfg.detector_ckpt):
        raise ValueError("detector_ckpt must be set before caching detections.")
    model.load_params_from_file(filename=str(cfg.detector_ckpt), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return {
        "dataset": dataset,
        "model": model,
        "load_data_to_gpu": load_data_to_gpu,
        "class_names": list(det_cfg.CLASS_NAMES),
    }


def _dataset_infos(dataset: Any) -> list[dict[str, Any]]:
    infos = getattr(dataset, "det_infos", None) or getattr(dataset, "tracking_infos", None) or getattr(dataset, "infos", None)
    if infos is None:
        return []
    return list(infos)


def _first_pred_dict(model_out: Any, batch_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    pred_dicts = None
    if isinstance(model_out, tuple) and model_out:
        pred_dicts = model_out[0]
    elif isinstance(model_out, list):
        pred_dicts = model_out
    elif isinstance(model_out, dict):
        pred_dicts = model_out.get("pred_dicts", model_out)
    if pred_dicts is None:
        pred_dicts = batch_dict.get("pred_dicts")
    if isinstance(pred_dicts, list) and pred_dicts:
        return pred_dicts[0]
    if isinstance(pred_dicts, dict):
        return pred_dicts
    raise KeyError("Detector output does not contain pred_dicts/pred_boxes/pred_scores/pred_labels")


def _prediction_to_frame(
    pred_dict: dict[str, torch.Tensor],
    info: dict[str, Any],
    *,
    raw_target_label: int,
    output_class_id: int,
) -> dict[str, Any]:
    boxes = _to_numpy(pred_dict.get("pred_boxes", torch.zeros((0, 7), dtype=torch.float32))).reshape(-1, 7)
    scores = _to_numpy(pred_dict.get("pred_scores", torch.zeros((len(boxes),), dtype=torch.float32))).reshape(-1)
    raw_labels = _to_numpy(pred_dict.get("pred_labels", torch.full((len(boxes),), raw_target_label))).astype(np.int64).reshape(-1)
    keep = raw_labels == int(raw_target_label)
    labels = np.full((int(keep.sum()),), int(output_class_id), dtype=np.int64)
    return {
        "sequence_id": str(info.get("sequence_id", "")),
        "frame_id": str(info.get("frame_id", info.get("frame_idx", ""))),
        "frame_idx": int(info.get("frame_idx", 0)),
        "pred_boxes": boxes[keep].astype(np.float32),
        "pred_scores": scores[keep].astype(np.float32),
        "pred_labels": labels,
    }


def _to_numpy(value: torch.Tensor | np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_path_part(value: Any) -> str:
    text = str(value)
    return text.replace("/", "_").replace("\\", "_") or "empty"


if __name__ == "__main__":
    main()
