from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import frame_to_objects
from .geometry import boxes_iou_3d_axis_aligned


@dataclass(frozen=True)
class ClearMotMetrics:
    num_gt: int
    num_pred: int
    matches: int
    false_positive: int
    false_negative: int
    id_switches: int
    fragments: int
    mostly_tracked: int
    mostly_lost: int
    motp_sum: float
    recall: float
    precision: float
    mota: float
    motp: float
    matched_scores: tuple[float, ...] = ()


def evaluate_ab3dmot_json(
    result_json: str | Path,
    info_path: str | Path,
    class_names: list[str],
    iou_threshold: float = 0.5,
    recall_points: int = 40,
    bev_range: list[float] | tuple[float, float, float, float] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, float | list[dict[str, float]]]:
    records = _load_records(result_json, info_path, class_names, bev_range=bev_range)
    full_metrics = _clear_mot_at_threshold(
        records,
        iou_threshold=iou_threshold,
        score_threshold=-float("inf"),
        collect_matched_scores=True,
    )
    thresholds = _thresholds_from_matched_scores(
        full_metrics.matched_scores,
        total_gt=full_metrics.num_gt,
        recall_points=recall_points,
    )
    curves: list[tuple[float, ClearMotMetrics]] = []
    seen_recalls: set[float] = set()
    for threshold in thresholds:
        metrics = _clear_mot_at_threshold(records, iou_threshold=iou_threshold, score_threshold=threshold)
        recall_key = round(metrics.recall, 8)
        if recall_key in seen_recalls and threshold not in {float("inf"), -float("inf")}:
            continue
        seen_recalls.add(recall_key)
        curves.append((threshold, metrics))
    curves.sort(key=lambda item: item[1].recall)

    target_recalls = np.linspace(1.0 / int(recall_points), 1.0, int(recall_points), dtype=np.float32)
    sampled = [_sample_curve_at_recall(curves, float(target)) for target in target_recalls]
    num_gt = max(full_metrics.num_gt, 1)
    amota_values = []
    samota_values = []
    amotp_values = []
    for target, _threshold, metrics in sampled:
        amota_values.append(metrics.mota)
        amotp_values.append(metrics.motp)
        denom = max(target * num_gt, 1e-12)
        smota = 1.0 - (
            metrics.id_switches
            + metrics.false_positive
            + metrics.false_negative
            - (1.0 - target) * num_gt
        ) / denom
        samota_values.append(max(0.0, smota))

    out: dict[str, float | list[dict[str, float]]] = {
        "sAMOTA": float(np.mean(samota_values)),
        "AMOTA": float(np.mean(amota_values)),
        "AMOTP": float(np.mean(amotp_values)),
        "MOTA": float(full_metrics.mota),
        "MOTP": float(full_metrics.motp),
        "RECALL": float(full_metrics.recall),
        "PRECISION": float(full_metrics.precision),
        "IDS": float(full_metrics.id_switches),
        "FRAG": float(full_metrics.fragments),
        "FP": float(full_metrics.false_positive),
        "FN": float(full_metrics.false_negative),
        "MT": float(full_metrics.mostly_tracked),
        "ML": float(full_metrics.mostly_lost),
        "num_gt": float(full_metrics.num_gt),
        "num_pred": float(full_metrics.num_pred),
        "matches": float(full_metrics.matches),
        "iou_threshold": float(iou_threshold),
        "recall_points": float(recall_points),
        "iou_type": "axis_aligned_3d",
        "bev_range": [float(value) for value in bev_range] if bev_range is not None else None,
        "curve": [
            {
                "target_recall": float(target),
                "score_threshold": float(threshold),
                "recall": float(metrics.recall),
                "mota": float(metrics.mota),
                "motp": float(metrics.motp),
                "precision": float(metrics.precision),
                "ids": float(metrics.id_switches),
                "fp": float(metrics.false_positive),
                "fn": float(metrics.false_negative),
            }
            for target, threshold, metrics in sampled
        ],
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out


def _load_records(
    result_json: str | Path,
    info_path: str | Path,
    class_names: list[str],
    *,
    bev_range: list[float] | tuple[float, float, float, float] | None = None,
) -> list[dict[str, Any]]:
    with Path(result_json).open("r", encoding="utf-8") as f:
        results = json.load(f)
    with Path(info_path).open("rb") as f:
        infos = pickle.load(f)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    result_by_key = {
        (str(frame.get("sequence_id", "")), str(frame.get("frame_id", ""))): frame
        for frame in results
    }
    ordered_infos = sorted(infos, key=lambda x: (str(x.get("sequence_id", "")), int(x.get("frame_idx", 0))))
    records = []
    for info in ordered_infos:
        seq = str(info.get("sequence_id", ""))
        frame_id = str(info.get("frame_id", info.get("frame_idx", "")))
        gt = frame_to_objects(info, class_to_id)
        frame = result_by_key.get((seq, frame_id), {"tracks": []})
        tracks = frame.get("tracks", [])
        pred_boxes = np.asarray([track.get("box3d", [0.0] * 7) for track in tracks], dtype=np.float32).reshape(-1, 7)
        pred_ids = np.asarray([track.get("id", -1) for track in tracks], dtype=np.int64).reshape(-1)
        pred_scores = np.asarray([track.get("score", 1.0) for track in tracks], dtype=np.float32).reshape(-1)
        gt_boxes, gt_ids = _filter_gt_by_bev_range(gt.boxes, gt.track_ids.astype(np.int64), bev_range)
        pred_boxes, pred_ids, pred_scores = _filter_pred_by_bev_range(pred_boxes, pred_ids, pred_scores, bev_range)
        records.append(
            {
                "sequence_id": seq,
                "frame_id": frame_id,
                "frame_idx": int(info.get("frame_idx", 0)),
                "gt_boxes": gt_boxes,
                "gt_ids": gt_ids,
                "pred_boxes": pred_boxes,
                "pred_ids": pred_ids,
                "pred_scores": pred_scores,
            }
        )
    return records


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


def _clear_mot_at_threshold(
    records: list[dict[str, Any]],
    *,
    iou_threshold: float,
    score_threshold: float,
    collect_matched_scores: bool = False,
) -> ClearMotMetrics:
    total_gt = 0
    total_pred = 0
    total_matches = 0
    false_positive = 0
    false_negative = 0
    id_switches = 0
    fragments = 0
    motp_sum = 0.0
    matched_scores: list[float] = []
    gt_track_total: dict[tuple[str, int], int] = defaultdict(int)
    gt_track_matched: dict[tuple[str, int], int] = defaultdict(int)
    gt_track_was_matched: dict[tuple[str, int], bool] = defaultdict(bool)
    gt_track_had_gap: dict[tuple[str, int], bool] = defaultdict(bool)
    gt_to_pred_last: dict[tuple[str, int], int] = {}

    for record in records:
        seq = str(record["sequence_id"])
        gt_boxes = record["gt_boxes"]
        gt_ids = record["gt_ids"]
        keep = record["pred_scores"] >= float(score_threshold)
        pred_boxes = record["pred_boxes"][keep]
        pred_ids = record["pred_ids"][keep]
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        for gt_id in gt_ids.tolist():
            gt_track_total[(seq, int(gt_id))] += 1
        iou = boxes_iou_3d_axis_aligned(gt_boxes, pred_boxes)
        matches = _match_by_iou(iou, iou_threshold)
        total_matches += len(matches)
        false_negative += len(gt_boxes) - len(matches)
        false_positive += len(pred_boxes) - len(matches)
        motp_sum += sum(score for _gt_idx, _pred_idx, score in matches)
        matched_gt_indices = {gt_idx for gt_idx, _pred_idx, _score in matches}
        for gt_idx, pred_idx, _score in matches:
            gt_key = (seq, int(gt_ids[gt_idx]))
            pred_id = int(pred_ids[pred_idx])
            if collect_matched_scores:
                matched_scores.append(float(record["pred_scores"][keep][pred_idx]))
            last_pred = gt_to_pred_last.get(gt_key)
            if last_pred is not None and last_pred != pred_id:
                id_switches += 1
            if gt_track_had_gap[gt_key]:
                fragments += 1
                gt_track_had_gap[gt_key] = False
            gt_to_pred_last[gt_key] = pred_id
            gt_track_matched[gt_key] += 1
            gt_track_was_matched[gt_key] = True
        for gt_idx, gt_id in enumerate(gt_ids.tolist()):
            gt_key = (seq, int(gt_id))
            if gt_idx not in matched_gt_indices and gt_track_was_matched[gt_key]:
                gt_track_had_gap[gt_key] = True

    mostly_tracked = 0
    mostly_lost = 0
    for gt_key, total in gt_track_total.items():
        ratio = gt_track_matched[gt_key] / max(total, 1)
        if ratio >= 0.8:
            mostly_tracked += 1
        if ratio <= 0.2:
            mostly_lost += 1
    recall = total_matches / max(total_gt, 1)
    precision = total_matches / max(total_pred, 1)
    mota = 1.0 - (false_negative + false_positive + id_switches) / max(total_gt, 1)
    motp = motp_sum / max(total_matches, 1)
    return ClearMotMetrics(
        num_gt=total_gt,
        num_pred=total_pred,
        matches=total_matches,
        false_positive=false_positive,
        false_negative=false_negative,
        id_switches=id_switches,
        fragments=fragments,
        mostly_tracked=mostly_tracked,
        mostly_lost=mostly_lost,
        motp_sum=motp_sum,
        recall=recall,
        precision=precision,
        mota=mota,
        motp=motp,
        matched_scores=tuple(matched_scores),
    )


def _thresholds_from_matched_scores(
    matched_scores: tuple[float, ...],
    *,
    total_gt: int,
    recall_points: int,
) -> list[float]:
    if not matched_scores:
        return [float("inf"), -float("inf")]
    scores = sorted((float(score) for score in matched_scores), reverse=True)
    thresholds = [float("inf")]
    total_gt = max(int(total_gt), 1)
    for target in np.linspace(1.0 / int(recall_points), 1.0, int(recall_points), dtype=np.float32):
        target_tp = int(np.ceil(float(target) * total_gt))
        idx = min(max(target_tp - 1, 0), len(scores) - 1)
        thresholds.append(scores[idx])
    thresholds.append(-float("inf"))
    return sorted(set(thresholds), reverse=True)


def _sample_curve_at_recall(
    curves: list[tuple[float, ClearMotMetrics]],
    target_recall: float,
) -> tuple[float, float, ClearMotMetrics]:
    if not curves:
        raise ValueError("Cannot sample an empty AB3DMOT curve")
    eligible = [(threshold, metrics) for threshold, metrics in curves if metrics.recall >= target_recall]
    if eligible:
        threshold, metrics = min(eligible, key=lambda item: (item[1].recall - target_recall, -item[0]))
        return target_recall, threshold, metrics
    threshold, metrics = curves[-1]
    return target_recall, threshold, metrics


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
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        matches = []
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
