from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import boxes_iou_bev_axis_aligned, greedy_match_by_iou


def frame_key(sequence_id: str, frame_id: str | int) -> tuple[str, str]:
    return str(sequence_id), str(frame_id)


def load_detection_annotations(path: str | Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not path:
        return {}
    det_path = Path(path)
    if not det_path.exists():
        raise FileNotFoundError(f"Detection file not found: {det_path}")
    if det_path.suffix.lower() == ".json":
        with det_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        with det_path.open("rb") as f:
            records = pickle.load(f)
    if isinstance(records, dict) and "frames" in records:
        records = records["frames"]
    if isinstance(records, dict):
        out = {}
        for key, value in records.items():
            if isinstance(key, tuple) and len(key) == 2:
                out[frame_key(key[0], key[1])] = value
            else:
                seq, frame = str(key).split("/", 1) if "/" in str(key) else ("", str(key))
                out[frame_key(seq, frame)] = value
        return out
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if "tracks" in record:
            boxes = [track.get("box3d", [0.0] * 7) for track in record["tracks"]]
            names = [track.get("class", "Car") for track in record["tracks"]]
            scores = [track.get("score", 1.0) for track in record["tracks"]]
            record = {
                **record,
                "boxes_lidar": boxes,
                "name": names,
                "score": scores,
            }
        seq = record.get("sequence_id", "")
        frame = record.get("frame_id", record.get("frame_idx", ""))
        out[frame_key(seq, frame)] = record
    return out


def detection_record_to_arrays(
    record: dict[str, Any] | None,
    class_names: list[str],
    score_thresh: float,
) -> dict[str, np.ndarray]:
    if not record:
        return _empty_detection()
    boxes = np.asarray(
        record.get("boxes_lidar", record.get("gt_boxes_lidar", record.get("box3d", np.zeros((0, 7))))),
        dtype=np.float32,
    ).reshape(-1, 7)
    scores = np.asarray(record.get("score", record.get("scores", np.ones((len(boxes),)))), dtype=np.float32).reshape(-1)
    if len(scores) != len(boxes):
        scores = np.ones((len(boxes),), dtype=np.float32)
    names = record.get("name", record.get("names", None))
    labels = record.get("pred_labels", record.get("labels", None))
    if names is None and labels is not None:
        labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
        names = [class_names[max(0, min(int(label) - 1, len(class_names) - 1))] for label in labels_np]
    if names is None:
        names = ["Car"] * len(boxes)
    names_np = np.asarray(names).astype(str)
    keep = scores >= float(score_thresh)
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "names": names_np[keep],
    }


def assign_detection_track_ids(
    detection_boxes: np.ndarray,
    detection_names: np.ndarray,
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    gt_track_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    assigned = np.full((len(detection_boxes),), -1, dtype=np.int64)
    for class_name in np.unique(detection_names).tolist():
        det_idx = np.where(detection_names == class_name)[0]
        gt_idx = np.where(gt_names.astype(str) == str(class_name))[0]
        if len(det_idx) == 0 or len(gt_idx) == 0:
            continue
        iou = boxes_iou_bev_axis_aligned(detection_boxes[det_idx], gt_boxes[gt_idx])
        for local_det, local_gt, _score in greedy_match_by_iou(iou, iou_threshold):
            assigned[det_idx[local_det]] = int(gt_track_ids[gt_idx[local_gt]])
    return assigned


def _empty_detection() -> dict[str, np.ndarray]:
    return {
        "boxes": np.zeros((0, 7), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "names": np.asarray([], dtype=str),
    }
