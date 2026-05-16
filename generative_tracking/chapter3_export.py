from __future__ import annotations

import copy
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config


def export_chapter3_detections(
    cfg: Config,
    split: str,
    output_path: str | Path,
    batch_size: int = 1,
    workers: int = 1,
    max_frames: int | None = None,
) -> Path:
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

    if not str(cfg.detector_ckpt):
        raise ValueError("Set detector_ckpt to a chapter3 checkpoint before exporting detector outputs.")

    old_cwd = Path.cwd()
    os.chdir(chapter3_root)
    try:
        det_cfg.clear()
        det_cfg.ROOT_DIR = chapter3_root.resolve()
        det_cfg.LOCAL_RANK = 0
        cfg_from_yaml_file(str(cfg.detector_cfg_file), det_cfg)
        det_cfg.DATA_CONFIG.DATA_SPLIT["test"] = split
        if split == "train" and "train" in det_cfg.DATA_CONFIG.INFO_PATH:
            det_cfg.DATA_CONFIG.INFO_PATH["test"] = det_cfg.DATA_CONFIG.INFO_PATH["train"]
    finally:
        os.chdir(old_cwd)
    logger = common_utils.create_logger(log_file=None, rank=0)
    dataset, loader, _sampler = build_dataloader(
        dataset_cfg=det_cfg.DATA_CONFIG,
        class_names=det_cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        root_path=chapter3_root / det_cfg.DATA_CONFIG.DATA_PATH,
        workers=workers,
        logger=logger,
        training=False,
    )
    model = build_network(det_cfg.MODEL, num_class=len(det_cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(cfg.detector_ckpt), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for frame_count, batch_dict in enumerate(loader, start=1):
            if max_frames is not None and frame_count > max_frames:
                break
            batch_cpu = copy.deepcopy(batch_dict)
            load_data_to_gpu(batch_dict)
            pred_dicts, _recall = model(batch_dict)
            frame_tokens = extract_detector_frame_tokens(batch_dict, int(cfg.model.max_detector_tokens))
            for batch_idx, pred in enumerate(pred_dicts):
                sequence_id = _get_batch_value(batch_cpu, "sequence_id", batch_idx, "")
                raw_frame_id = _get_batch_value(batch_cpu, "frame_id", batch_idx, frame_count - 1)
                frame_id = _normalize_frame_id(sequence_id, raw_frame_id)
                record = {
                    "sequence_id": str(sequence_id),
                    "frame_id": str(frame_id),
                    "frame_idx": int(_get_batch_value(batch_cpu, "frame_idx", batch_idx, frame_count - 1)),
                    "boxes_lidar": pred["pred_boxes"].detach().cpu().numpy().astype(np.float32),
                    "score": pred["pred_scores"].detach().cpu().numpy().astype(np.float32),
                    "pred_labels": pred["pred_labels"].detach().cpu().numpy().astype(np.int64),
                    "name": np.asarray(det_cfg.CLASS_NAMES, dtype=object)[pred["pred_labels"].detach().cpu().numpy().astype(np.int64) - 1],
                }
                if batch_idx < len(frame_tokens):
                    record["detector_tokens"] = frame_tokens[batch_idx]
                records.append(record)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump({"frames": records, "class_names": list(det_cfg.CLASS_NAMES)}, f)
    return output


def extract_detector_frame_tokens(batch_dict: dict[str, Any], max_tokens: int) -> list[np.ndarray]:
    if "spatial_features" in batch_dict and isinstance(batch_dict["spatial_features"], torch.Tensor):
        bev = batch_dict["spatial_features"].detach()
        bsz, channels, height, width = bev.shape
        flat = bev.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
        score = flat.norm(dim=-1)
        k = min(max_tokens, flat.shape[1])
        indices = score.topk(k=k, dim=1).indices
        return [flat[b, indices[b]].cpu().numpy().astype(np.float32) for b in range(bsz)]

    sparse = batch_dict.get("encoded_spconv_tensor", None)
    if sparse is not None and hasattr(sparse, "features") and hasattr(sparse, "indices"):
        features = sparse.features.detach()
        indices = sparse.indices.detach()
        batch_size = int(batch_dict.get("batch_size", int(indices[:, 0].max().item()) + 1))
        tokens = []
        for batch_idx in range(batch_size):
            mask = indices[:, 0].eq(batch_idx)
            cur = features[mask]
            if len(cur) == 0:
                tokens.append(np.zeros((0, features.shape[-1]), dtype=np.float32))
                continue
            score = cur.norm(dim=-1)
            k = min(max_tokens, len(cur))
            top = score.topk(k=k).indices
            tokens.append(cur[top].cpu().numpy().astype(np.float32))
        return tokens

    return []


def _get_batch_value(batch: dict[str, Any], key: str, idx: int, default: Any) -> Any:
    if key not in batch:
        return default
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)[idx].item()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[idx].item()
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value


def _normalize_frame_id(sequence_id: Any, frame_id: Any) -> str:
    text = str(frame_id)
    prefix = f"{sequence_id}_"
    if str(sequence_id) and text.startswith(prefix):
        return text[len(prefix):]
    return text
