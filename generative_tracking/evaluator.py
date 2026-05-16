from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .data import frame_to_objects
from .geometry import boxes_iou_bev_axis_aligned, greedy_match_by_iou


def evaluate_tracking_json(
    result_json: str | Path,
    info_path: str | Path,
    class_names: list[str],
    iou_threshold: float = 0.5,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    with Path(result_json).open("r", encoding="utf-8") as f:
        results = json.load(f)
    with Path(info_path).open("rb") as f:
        infos = pickle.load(f)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    gt_by_key = {(str(info.get("sequence_id", "")), str(info.get("frame_id", info.get("frame_idx", "")))): info for info in infos}

    total_gt = 0
    total_pred = 0
    total_matches = 0
    false_pos = 0
    false_neg = 0
    id_switches = 0
    gt_to_pred_last: dict[tuple[str, int], int] = {}

    for frame in results:
        seq = str(frame.get("sequence_id", ""))
        frame_id = str(frame.get("frame_id", ""))
        gt_info = gt_by_key.get((seq, frame_id))
        if gt_info is None:
            continue
        gt = frame_to_objects(gt_info, class_to_id)
        pred_tracks = frame.get("tracks", [])
        pred_boxes = np.asarray([track.get("box3d", [0.0] * 7) for track in pred_tracks], dtype=np.float32).reshape(-1, 7)
        pred_ids = np.asarray([track.get("id", -1) for track in pred_tracks], dtype=np.int64)
        total_gt += len(gt.boxes)
        total_pred += len(pred_boxes)
        iou = boxes_iou_bev_axis_aligned(gt.boxes, pred_boxes)
        matches = greedy_match_by_iou(iou, iou_threshold)
        total_matches += len(matches)
        false_neg += len(gt.boxes) - len(matches)
        false_pos += len(pred_boxes) - len(matches)
        for gt_idx, pred_idx, _score in matches:
            gt_key = (seq, int(gt.track_ids[gt_idx]))
            pred_id = int(pred_ids[pred_idx])
            last_pred = gt_to_pred_last.get(gt_key)
            if last_pred is not None and last_pred != pred_id:
                id_switches += 1
            gt_to_pred_last[gt_key] = pred_id

    precision = total_matches / max(total_pred, 1)
    recall = total_matches / max(total_gt, 1)
    mota = 1.0 - (false_neg + false_pos + id_switches) / max(total_gt, 1)
    metrics = {
        "num_gt": float(total_gt),
        "num_pred": float(total_pred),
        "matches": float(total_matches),
        "precision": float(precision),
        "recall": float(recall),
        "mota_like": float(mota),
        "id_switches": float(id_switches),
        "false_positive": float(false_pos),
        "false_negative": float(false_neg),
        "iou_threshold": float(iou_threshold),
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return metrics
