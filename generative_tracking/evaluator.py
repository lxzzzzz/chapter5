from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import frame_to_objects
from .geometry import boxes_iou_3d_axis_aligned


def evaluate_tracking_json(
    result_json: str | Path,
    info_path: str | Path,
    class_names: list[str],
    iou_threshold: float = 0.5,
    bev_range: list[float] | tuple[float, float, float, float] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    with Path(result_json).open("r", encoding="utf-8") as f:
        results = json.load(f)
    with Path(info_path).open("rb") as f:
        infos = pickle.load(f)

    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    gt_by_key = {
        (str(info.get("sequence_id", "")), str(info.get("frame_id", info.get("frame_idx", "")))): info
        for info in infos
    }
    result_by_key = {
        (str(frame.get("sequence_id", "")), str(frame.get("frame_id", ""))): frame
        for frame in results
    }
    ordered_keys = sorted(
        gt_by_key,
        key=lambda key: (key[0], int(gt_by_key[key].get("frame_idx", 0)), key[1]),
    )

    frame_records: list[dict[str, Any]] = []
    gt_track_total: dict[tuple[str, int], int] = defaultdict(int)
    gt_track_matched: dict[tuple[str, int], int] = defaultdict(int)
    gt_track_was_matched: dict[tuple[str, int], bool] = defaultdict(bool)
    gt_track_had_gap: dict[tuple[str, int], bool] = defaultdict(bool)
    gt_to_pred_last: dict[tuple[str, int], int] = {}
    sequence_last_seen: dict[tuple[str, int], int] = {}

    total_gt = 0
    total_pred = 0
    total_matches = 0
    false_pos = 0
    false_neg = 0
    id_switches = 0
    fragments = 0
    motp_iou_sum = 0.0

    for global_frame_idx, key in enumerate(ordered_keys):
        seq, frame_id = key
        gt = frame_to_objects(gt_by_key[key], class_to_id)
        frame = result_by_key.get(key, {"tracks": []})
        pred_tracks = frame.get("tracks", [])
        pred_boxes = np.asarray([track.get("box3d", [0.0] * 7) for track in pred_tracks], dtype=np.float32).reshape(-1, 7)
        pred_ids = np.asarray([track.get("id", -1) for track in pred_tracks], dtype=np.int64)
        pred_scores = np.asarray([track.get("score", 1.0) for track in pred_tracks], dtype=np.float32).reshape(-1)
        gt_boxes, gt_track_ids = _filter_gt_by_bev_range(gt.boxes, gt.track_ids.astype(np.int64), bev_range)
        pred_boxes, pred_ids, pred_scores = _filter_pred_by_bev_range(pred_boxes, pred_ids, pred_scores, bev_range)

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        for track_id in gt_track_ids.tolist():
            gt_key = (seq, int(track_id))
            gt_track_total[gt_key] += 1
            sequence_last_seen[gt_key] = global_frame_idx

        iou = boxes_iou_3d_axis_aligned(gt_boxes, pred_boxes)
        matches = _match_by_iou(iou, iou_threshold)
        total_matches += len(matches)
        false_neg += len(gt_boxes) - len(matches)
        false_pos += len(pred_boxes) - len(matches)
        motp_iou_sum += sum(score for _gt_idx, _pred_idx, score in matches)

        matched_gt_indices = {gt_idx for gt_idx, _pred_idx, _score in matches}
        for gt_idx, pred_idx, _score in matches:
            gt_key = (seq, int(gt_track_ids[gt_idx]))
            pred_id = int(pred_ids[pred_idx]) if pred_idx < len(pred_ids) else -1
            last_pred = gt_to_pred_last.get(gt_key)
            if last_pred is not None and last_pred != pred_id:
                id_switches += 1
            if gt_track_had_gap[gt_key]:
                fragments += 1
                gt_track_had_gap[gt_key] = False
            gt_to_pred_last[gt_key] = pred_id
            gt_track_matched[gt_key] += 1
            gt_track_was_matched[gt_key] = True

        for gt_idx, track_id in enumerate(gt_track_ids.tolist()):
            gt_key = (seq, int(track_id))
            if gt_idx not in matched_gt_indices and gt_track_was_matched[gt_key]:
                gt_track_had_gap[gt_key] = True

        frame_records.append(
            {
                "key": key,
                "gt_boxes": gt_boxes,
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
            }
        )

    precision = total_matches / max(total_pred, 1)
    recall = total_matches / max(total_gt, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    mota = 1.0 - (false_neg + false_pos + id_switches) / max(total_gt, 1)
    motp = motp_iou_sum / max(total_matches, 1)
    mostly_tracked = 0
    mostly_lost = 0
    for gt_key, total in gt_track_total.items():
        ratio = gt_track_matched[gt_key] / max(total, 1)
        if ratio >= 0.8:
            mostly_tracked += 1
        if ratio <= 0.2:
            mostly_lost += 1

    ap = _average_precision_3d(frame_records, iou_threshold)
    metrics = {
        "num_gt": float(total_gt),
        "num_pred": float(total_pred),
        "matches": float(total_matches),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ap_3d_iou_0_50": float(ap),
        "mota": float(mota),
        "motp": float(motp),
        "mota_like": float(mota),
        "id_switches": float(id_switches),
        "fragments": float(fragments),
        "mostly_tracked": float(mostly_tracked),
        "mostly_lost": float(mostly_lost),
        "false_positive": float(false_pos),
        "false_negative": float(false_neg),
        "iou_threshold": float(iou_threshold),
        "iou_type": "axis_aligned_3d",
        "bev_range": [float(value) for value in bev_range] if bev_range is not None else None,
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def _bev_mask(boxes: np.ndarray, bev_range: list[float] | tuple[float, float, float, float] | None) -> np.ndarray:
    if bev_range is None:
        return np.ones((len(boxes),), dtype=bool)
    if len(bev_range) != 4:
        raise ValueError("bev_range must contain [x_min, y_min, x_max, y_max]")
    if len(boxes) == 0:
        return np.zeros((0,), dtype=bool)
    x_min, y_min, x_max, y_max = [float(value) for value in bev_range]
    centers = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)[:, :2]
    return (
        (centers[:, 0] >= x_min)
        & (centers[:, 0] <= x_max)
        & (centers[:, 1] >= y_min)
        & (centers[:, 1] <= y_max)
    )


def _filter_gt_by_bev_range(
    boxes: np.ndarray,
    track_ids: np.ndarray,
    bev_range: list[float] | tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    keep = _bev_mask(boxes, bev_range)
    return boxes[keep], track_ids[keep]


def _filter_pred_by_bev_range(
    boxes: np.ndarray,
    track_ids: np.ndarray,
    scores: np.ndarray,
    bev_range: list[float] | tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = _bev_mask(boxes, bev_range)
    return boxes[keep], track_ids[keep], scores[keep]


def _match_by_iou(iou: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    if iou.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        gt_idx, pred_idx = linear_sum_assignment(-iou)
        matches = [
            (int(g), int(p), float(iou[g, p]))
            for g, p in zip(gt_idx.tolist(), pred_idx.tolist())
            if float(iou[g, p]) >= threshold
        ]
        matches.sort(key=lambda item: item[2], reverse=True)
        return matches
    except ImportError:
        return _greedy_match_by_iou(iou, threshold)


def _greedy_match_by_iou(iou: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    flat_order = np.argsort(iou.reshape(-1))[::-1]
    num_pred = iou.shape[1]
    for flat_idx in flat_order.tolist():
        score = float(iou.reshape(-1)[flat_idx])
        if score < threshold:
            break
        gt_idx = flat_idx // num_pred
        pred_idx = flat_idx % num_pred
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, score))
    return matches


def _average_precision_3d(frame_records: list[dict[str, Any]], threshold: float) -> float:
    predictions: list[tuple[float, int, int]] = []
    total_gt = 0
    for frame_idx, record in enumerate(frame_records):
        total_gt += len(record["gt_boxes"])
        for pred_idx, score in enumerate(record["pred_scores"].tolist()):
            predictions.append((float(score), frame_idx, pred_idx))
    if total_gt == 0:
        return 0.0
    if not predictions:
        return 0.0

    predictions.sort(key=lambda item: item[0], reverse=True)
    matched_gt_by_frame: dict[int, set[int]] = defaultdict(set)
    tp = np.zeros((len(predictions),), dtype=np.float32)
    fp = np.zeros((len(predictions),), dtype=np.float32)
    for idx, (_score, frame_idx, pred_idx) in enumerate(predictions):
        record = frame_records[frame_idx]
        gt_boxes = record["gt_boxes"]
        pred_boxes = record["pred_boxes"]
        if len(gt_boxes) == 0:
            fp[idx] = 1.0
            continue
        iou = boxes_iou_3d_axis_aligned(gt_boxes, pred_boxes[pred_idx:pred_idx + 1]).reshape(-1)
        best_gt = int(np.argmax(iou))
        best_iou = float(iou[best_gt])
        if best_iou >= threshold and best_gt not in matched_gt_by_frame[frame_idx]:
            tp[idx] = 1.0
            matched_gt_by_frame[frame_idx].add(best_gt)
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(total_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return float(_integrate_pr_curve(recalls, precisions))


def _integrate_pr_curve(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))
