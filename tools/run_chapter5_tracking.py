#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from generative_tracking.ab3dmot_evaluator import evaluate_ab3dmot_json
from generative_tracking.config import load_config, resolve_info_path
from generative_tracking.evaluator import evaluate_tracking_json
from generative_tracking.geometry import boxes_iou_3d_axis_aligned
from generative_tracking.nova_data import DetectionFrame, filter_detection_frame, load_detection_cache
from tools.run_ab3dmot_tracking import _initial_velocity_from_box, _normalize_dt_hypotheses, match_by_iou

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


VARIANTS = {
    "centerpoint_track",
    "simpletrack",
    "eagermot",
    "full_5_4",
    "baseline_motion",
    "baseline_nekfm",
    "baseline_nekfm_tlom",
    "fmca_single_stage",
    "fmca_two_stage",
    "fmca_full",
    "full_5_5",
}


@dataclass
class TrackState:
    track_id: int
    box: np.ndarray
    velocity: np.ndarray
    score: float
    class_id: int
    lost_frames: int = 0
    age: int = 1
    hits: int = 1
    history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=6))
    residual_history: deque[float] = field(default_factory=lambda: deque(maxlen=6))
    reliability: float = 1.0

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(self.box.astype(np.float32).copy())


class Chapter5Tracker:
    """Unified cache-input tracker variants for Chapter 5 table rows.

    The variants are engineering baselines that make each table row executable
    under the same detector cache. Learning-heavy rows such as NEKFM/TLOM are
    represented by their online inference behavior: smoothed motion update,
    residual-aware lifecycle probability, and state-cascade association.
    """

    def __init__(
        self,
        *,
        class_names: list[str],
        variant: str,
        max_lost_frames: int,
        min_hits: int,
        high_score_thresh: float,
        low_score_thresh: float,
        iou_threshold: float,
        center_distance: float,
        tlom_threshold: float,
        dt_hypotheses: list[float] | None = None,
        init_velocity_mode: str = "zero",
        init_speed_prior: float = 0.0,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Chapter 5 tracking variant: {variant}")
        self.class_names = class_names
        self.variant = variant
        self.max_lost_frames = int(max_lost_frames)
        self.min_hits = int(min_hits)
        self.high_score_thresh = float(high_score_thresh)
        self.low_score_thresh = float(low_score_thresh)
        self.iou_threshold = float(iou_threshold)
        self.center_distance = float(center_distance)
        self.tlom_threshold = float(tlom_threshold)
        self.dt_hypotheses = _normalize_dt_hypotheses(dt_hypotheses)
        self.init_velocity_mode = str(init_velocity_mode)
        self.init_speed_prior = float(init_speed_prior)
        self.next_id = 0
        self.tracks: dict[int, TrackState] = {}

    def reset(self) -> None:
        self.next_id = 0
        self.tracks.clear()

    def update(self, det: DetectionFrame) -> dict[str, Any]:
        if self.variant == "centerpoint_track":
            self._single_stage_update(det, cost="center", use_smooth_motion=False, use_tlom=False)
        elif self.variant == "simpletrack":
            self._two_stage_update(det, cost="center", use_smooth_motion=True, use_tlom=False, start_low=False)
        elif self.variant == "eagermot":
            self._two_stage_update(det, cost="fused", use_smooth_motion=True, use_tlom=False, start_low=False)
        elif self.variant == "full_5_4":
            self._two_stage_update(det, cost="fused", use_smooth_motion=True, use_tlom=False, start_low=False)
        elif self.variant == "baseline_motion":
            self._single_stage_update(det, cost="center", use_smooth_motion=False, use_tlom=False)
        elif self.variant == "baseline_nekfm":
            self._single_stage_update(det, cost="center", use_smooth_motion=True, use_tlom=False)
        elif self.variant == "baseline_nekfm_tlom":
            self._single_stage_update(det, cost="center", use_smooth_motion=True, use_tlom=True)
        elif self.variant == "fmca_single_stage":
            self._single_stage_update(det, cost="fused", use_smooth_motion=True, use_tlom=True)
        elif self.variant == "fmca_two_stage":
            self._two_stage_update(det, cost="fused", use_smooth_motion=True, use_tlom=True, start_low=False)
        elif self.variant in {"fmca_full", "full_5_5"}:
            self._fmca_update(det)
        else:
            raise AssertionError(f"Unhandled variant: {self.variant}")
        outputs = [self._track_output(track) for track in self.tracks.values() if track.lost_frames == 0 and track.hits >= self.min_hits]
        outputs.sort(key=lambda item: int(item["id"]))
        return {"sequence_id": det.sequence_id, "frame_id": det.frame_id, "tracks": outputs}

    def _single_stage_update(self, det: DetectionFrame, *, cost: str, use_smooth_motion: bool, use_tlom: bool) -> None:
        active_ids = sorted(self.tracks)
        matches, unmatched_tracks, unmatched_dets = self._match(active_ids, det, cost=cost, use_smooth_motion=use_smooth_motion)
        self._apply_matches(active_ids, det, matches, use_smooth_motion=use_smooth_motion)
        for det_idx in unmatched_dets:
            if det.pred_scores[det_idx] >= self.high_score_thresh:
                self._start_track(det, det_idx)
        self._age_unmatched(active_ids, unmatched_tracks, use_smooth_motion=use_smooth_motion, use_tlom=use_tlom)

    def _two_stage_update(
        self,
        det: DetectionFrame,
        *,
        cost: str,
        use_smooth_motion: bool,
        use_tlom: bool,
        start_low: bool,
    ) -> None:
        high = _subset_detection(det, det.pred_scores >= self.high_score_thresh)
        low = _subset_detection(det, (det.pred_scores >= self.low_score_thresh) & (det.pred_scores < self.high_score_thresh))
        active_ids = sorted(self.tracks)
        matches, unmatched_track_pos, unmatched_high = self._match(active_ids, high, cost=cost, use_smooth_motion=use_smooth_motion)
        self._apply_matches(active_ids, high, matches, use_smooth_motion=use_smooth_motion)
        matched_track_pos = {track_pos for track_pos, _det_idx, _score, _dt_scale in matches}

        if unmatched_track_pos and len(low.pred_boxes):
            low_track_ids = [active_ids[pos] for pos in unmatched_track_pos]
            low_matches, low_unmatched_local, _low_unmatched = self._match(low_track_ids, low, cost="center", threshold=1e-6, use_smooth_motion=use_smooth_motion)
            accepted_local: set[int] = set()
            for local_track_pos, det_idx, score, dt_scale in low_matches:
                track_pos = unmatched_track_pos[local_track_pos]
                track_id = active_ids[track_pos]
                distance = self._best_center_distance(track_id, low.pred_boxes[det_idx], use_smooth_motion=use_smooth_motion)
                if distance <= self.center_distance:
                    self._update_track(track_id, low, det_idx, use_smooth_motion=use_smooth_motion, dt_scale=dt_scale)
                    matched_track_pos.add(track_pos)
                    accepted_local.add(local_track_pos)
            unmatched_track_pos = sorted(
                set(unmatched_track_pos[idx] for idx in low_unmatched_local)
                | set(pos for idx, pos in enumerate(unmatched_track_pos) if idx not in accepted_local and idx not in low_unmatched_local)
            )

        for det_idx in unmatched_high:
            self._start_track(high, det_idx)
        if start_low:
            high_count = len(high.pred_boxes)
            high_indices = set(np.where(det.pred_scores >= self.high_score_thresh)[0].tolist())
            for det_idx, score in enumerate(det.pred_scores.tolist()):
                if det_idx not in high_indices and score >= self.low_score_thresh:
                    self._start_track(det, det_idx)
            _ = high_count
        self._age_unmatched(active_ids, [idx for idx in unmatched_track_pos if idx not in matched_track_pos], use_smooth_motion=use_smooth_motion, use_tlom=use_tlom)

    def _fmca_update(self, det: DetectionFrame) -> None:
        confirmed = [tid for tid, trk in self.tracks.items() if trk.lost_frames == 0 and trk.hits >= self.min_hits]
        short_lost = [tid for tid, trk in self.tracks.items() if 0 < trk.lost_frames <= max(1, self.max_lost_frames // 2)]
        long_lost = [tid for tid, trk in self.tracks.items() if trk.lost_frames > max(1, self.max_lost_frames // 2)]
        high = _subset_detection(det, det.pred_scores >= self.high_score_thresh)
        low = _subset_detection(det, (det.pred_scores >= self.low_score_thresh) & (det.pred_scores < self.high_score_thresh))

        matched_ids: set[int] = set()
        used_high: set[int] = set()
        used_low: set[int] = set()
        for group_ids, group_det, cost, threshold, used in (
            (sorted(confirmed), high, "fused", None, used_high),
            (sorted(short_lost), high, "center", 1e-6, used_high),
            (sorted(long_lost), low, "center", 1e-6, used_low),
        ):
            available_det, index_map = _remove_indices(group_det, used)
            if not group_ids or len(available_det.pred_boxes) == 0:
                continue
            matches, _unmatched_tracks, _unmatched_dets = self._match(group_ids, available_det, cost=cost, threshold=threshold, use_smooth_motion=True)
            for track_pos, det_idx, _score, dt_scale in matches:
                track_id = group_ids[track_pos]
                if track_id in matched_ids:
                    continue
                distance = self._best_center_distance(track_id, available_det.pred_boxes[det_idx], use_smooth_motion=True)
                if cost == "center" and distance > self.center_distance:
                    continue
                original_idx = int(index_map[det_idx])
                self._update_track(track_id, available_det, det_idx, use_smooth_motion=True, dt_scale=dt_scale)
                matched_ids.add(track_id)
                used.add(original_idx)

        for det_idx, score in enumerate(high.pred_scores.tolist()):
            if det_idx not in used_high and score >= self.high_score_thresh:
                self._start_track(high, det_idx)
        active_ids = sorted(self.tracks)
        unmatched_positions = [idx for idx, tid in enumerate(active_ids) if tid not in matched_ids]
        self._age_unmatched(active_ids, unmatched_positions, use_smooth_motion=True, use_tlom=True)

    def _match(
        self,
        track_ids: list[int],
        det: DetectionFrame,
        *,
        cost: str,
        threshold: float | None = None,
        use_smooth_motion: bool = True,
    ) -> tuple[list[tuple[int, int, float, float]], list[int], list[int]]:
        if not track_ids:
            return [], [], list(range(len(det.pred_boxes)))
        if len(det.pred_boxes) == 0:
            return [], list(range(len(track_ids))), []
        accept = self._accept_threshold(cost, threshold)
        matches: list[tuple[int, int, float, float]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for dt_scale in self.dt_hypotheses:
            remaining_track_pos = [idx for idx in range(len(track_ids)) if idx not in matched_tracks]
            remaining_det_idx = [idx for idx in range(len(det.pred_boxes)) if idx not in matched_dets]
            if not remaining_track_pos or not remaining_det_idx:
                break
            remaining_track_ids = [track_ids[pos] for pos in remaining_track_pos]
            pred_boxes = np.stack(
                [self._predicted_box(track_id, use_smooth_motion=use_smooth_motion, dt_scale=dt_scale) for track_id in remaining_track_ids]
            ).astype(np.float32)
            det_boxes = det.pred_boxes[remaining_det_idx]
            score = self._association_score(pred_boxes, det_boxes, cost)
            for row, track_id in enumerate(remaining_track_ids):
                track_cls = int(self.tracks[track_id].class_id)
                score[row, det.pred_labels[remaining_det_idx] != track_cls] = 0.0
            local_matches, _local_unmatched_tracks, _local_unmatched_dets = match_by_iou(score, accept)
            for local_track_pos, local_det_idx, match_score in local_matches:
                track_pos = remaining_track_pos[local_track_pos]
                det_idx = remaining_det_idx[local_det_idx]
                matches.append((track_pos, det_idx, match_score, float(dt_scale)))
                matched_tracks.add(track_pos)
                matched_dets.add(det_idx)
        unmatched_tracks = [idx for idx in range(len(track_ids)) if idx not in matched_tracks]
        unmatched_dets = [idx for idx in range(len(det.pred_boxes)) if idx not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _accept_threshold(self, cost: str, threshold: float | None) -> float:
        if threshold is not None:
            return float(threshold)
        if cost == "iou":
            return self.iou_threshold
        if cost == "center":
            return 1e-6
        if cost == "fused":
            return 0.05
        raise ValueError(f"Unknown cost type: {cost}")

    def _association_score(self, pred_boxes: np.ndarray, det_boxes: np.ndarray, cost: str) -> np.ndarray:
        if cost == "iou":
            return boxes_iou_3d_axis_aligned(pred_boxes, det_boxes)
        if cost == "center":
            distance = _center_distance(pred_boxes, det_boxes)
            return np.maximum(0.0, 1.0 - distance / max(self.center_distance, 1e-6)).astype(np.float32)
        if cost == "fused":
            iou = boxes_iou_3d_axis_aligned(pred_boxes, det_boxes)
            distance = _center_distance(pred_boxes, det_boxes)
            center_score = np.maximum(0.0, 1.0 - distance / max(self.center_distance, 1e-6)).astype(np.float32)
            return (0.55 * iou + 0.45 * center_score).astype(np.float32)
        raise ValueError(f"Unknown cost type: {cost}")

    def _apply_matches(self, active_ids: list[int], det: DetectionFrame, matches: list[tuple[int, int, float, float]], *, use_smooth_motion: bool) -> None:
        for track_pos, det_idx, _score, dt_scale in matches:
            self._update_track(active_ids[track_pos], det, det_idx, use_smooth_motion=use_smooth_motion, dt_scale=dt_scale)

    def _age_unmatched(self, active_ids: list[int], unmatched_track_pos: list[int], *, use_smooth_motion: bool, use_tlom: bool) -> None:
        for track_pos in unmatched_track_pos:
            if track_pos >= len(active_ids):
                continue
            track_id = active_ids[track_pos]
            track = self.tracks.get(track_id)
            if track is None:
                continue
            track.box = self._predicted_box(track_id, use_smooth_motion=use_smooth_motion)
            track.history.append(track.box.astype(np.float32).copy())
            track.lost_frames += 1
            track.age += 1
            track.reliability *= 0.82
            keep = track.lost_frames <= self.max_lost_frames
            if use_tlom:
                keep = keep and self._survival_probability(track) >= self.tlom_threshold
            if not keep:
                del self.tracks[track_id]

    def _start_track(self, det: DetectionFrame, det_idx: int) -> None:
        track_id = self.next_id
        self.next_id += 1
        box = det.pred_boxes[det_idx].astype(np.float32)
        self.tracks[track_id] = TrackState(
            track_id=track_id,
            box=box,
            velocity=_initial_velocity_from_box(
                box,
                mode=self.init_velocity_mode,
                speed_prior=self.init_speed_prior,
            ),
            score=float(det.pred_scores[det_idx]),
            class_id=int(det.pred_labels[det_idx]),
        )

    def _update_track(self, track_id: int, det: DetectionFrame, det_idx: int, *, use_smooth_motion: bool, dt_scale: float) -> None:
        track = self.tracks[track_id]
        new_box = det.pred_boxes[det_idx].astype(np.float32)
        pred_box = self._predicted_box(track_id, use_smooth_motion=use_smooth_motion, dt_scale=dt_scale)
        residual = float(np.linalg.norm(new_box[:3] - pred_box[:3]))
        raw_velocity = (new_box - track.box) / max(float(dt_scale), 1e-6)
        if use_smooth_motion and track.history:
            track.velocity = 0.65 * track.velocity + 0.35 * raw_velocity
        else:
            track.velocity = raw_velocity
        track.box = new_box
        track.score = float(det.pred_scores[det_idx])
        track.class_id = int(det.pred_labels[det_idx])
        track.lost_frames = 0
        track.age += 1
        track.hits += 1
        track.history.append(new_box.copy())
        track.residual_history.append(residual)
        track.reliability = min(1.0, 0.7 * track.reliability + 0.3 * float(det.pred_scores[det_idx]))

    def _predicted_box(self, track_id: int, *, use_smooth_motion: bool, dt_scale: float = 1.0) -> np.ndarray:
        track = self.tracks[track_id]
        pred = track.box.astype(np.float32).copy()
        velocity = track.velocity.astype(np.float32)
        if use_smooth_motion and len(track.history) >= 2:
            boxes = list(track.history)
            deltas = [boxes[idx] - boxes[idx - 1] for idx in range(1, len(boxes))]
            weights = np.linspace(0.5, 1.0, num=len(deltas), dtype=np.float32)
            smooth = np.average(np.stack(deltas), axis=0, weights=weights)
            velocity = 0.5 * velocity + 0.5 * smooth.astype(np.float32)
        pred[:3] += velocity[:3] * float(dt_scale)
        pred[6] += velocity[6] * float(dt_scale)
        return pred

    def _best_center_distance(self, track_id: int, det_box: np.ndarray, *, use_smooth_motion: bool) -> float:
        distances = [
            float(np.linalg.norm(self._predicted_box(track_id, use_smooth_motion=use_smooth_motion, dt_scale=dt_scale)[:3] - det_box[:3]))
            for dt_scale in self.dt_hypotheses
        ]
        return min(distances) if distances else float("inf")

    def _survival_probability(self, track: TrackState) -> float:
        residual = float(np.mean(track.residual_history)) if track.residual_history else 0.0
        residual_term = np.exp(-residual / max(self.center_distance, 1e-6))
        lost_term = np.exp(-track.lost_frames / max(self.max_lost_frames, 1))
        hit_term = min(1.0, track.hits / 4.0)
        return float(0.45 * track.reliability + 0.30 * residual_term + 0.15 * lost_term + 0.10 * hit_term)

    def _track_output(self, track: TrackState) -> dict[str, Any]:
        class_name = self.class_names[track.class_id] if 0 <= track.class_id < len(self.class_names) else "Unknown"
        return {
            "id": int(track.track_id),
            "class": str(class_name),
            "box3d": [float(x) for x in track.box.tolist()],
            "score": float(track.score),
            "state": "active",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run executable Chapter 5 tracking table variants")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--detection_cache_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--score_thresh", type=float, default=0.6)
    parser.add_argument("--high_score_thresh", type=float, default=0.6)
    parser.add_argument("--low_score_thresh", type=float, default=0.3)
    parser.add_argument("--max_dets_per_frame", type=int, default=100)
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--center_distance", type=float, default=4.0)
    parser.add_argument("--tlom_threshold", type=float, default=0.5)
    parser.add_argument("--eval_iou_threshold", type=float, default=None)
    parser.add_argument("--max_lost_frames", type=int, default=2)
    parser.add_argument("--min_hits", type=int, default=1)
    parser.add_argument("--dt_hypotheses", type=float, nargs="+", default=None)
    parser.add_argument("--init_velocity_mode", choices=["zero", "heading_prior"], default="zero")
    parser.add_argument("--init_speed_prior", type=float, default=0.0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--progress_interval", type=int, default=50)
    parser.add_argument("--eval_metrics", action="store_true")
    parser.add_argument("--eval_ab3dmot", action="store_true")
    parser.add_argument("--ab3dmot_recall_points", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    if args.detection_cache_root:
        cfg.detection_cache.root = str(args.detection_cache_root)
    if args.output_dir:
        cfg.output_dir = str(args.output_dir)

    output_dir = Path(args.output_dir or cfg.output_dir)
    output_path = Path(args.output) if args.output else output_dir / f"{args.variant}_{args.split}_tracking_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    outputs, info_path = run_tracking(
        cfg=cfg,
        split=str(args.split),
        variant=str(args.variant),
        score_thresh=float(args.score_thresh),
        high_score_thresh=float(args.high_score_thresh),
        low_score_thresh=float(args.low_score_thresh),
        max_dets_per_frame=int(args.max_dets_per_frame),
        iou_threshold=float(args.iou_threshold),
        center_distance=float(args.center_distance),
        tlom_threshold=float(args.tlom_threshold),
        max_lost_frames=int(args.max_lost_frames),
        min_hits=int(args.min_hits),
        dt_hypotheses=args.dt_hypotheses,
        init_velocity_mode=str(args.init_velocity_mode),
        init_speed_prior=float(args.init_speed_prior),
        max_frames=int(args.max_frames),
        progress_interval=max(1, int(args.progress_interval)),
    )
    tracking_seconds = time.perf_counter() - start_time
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    print(f"wrote {len(outputs)} frames to {output_path}", flush=True)
    _write_runtime_summary(
        output_dir / "runtime_summary.json",
        method=str(args.variant),
        frames=len(outputs),
        tracking_seconds=tracking_seconds,
    )

    eval_iou = float(args.eval_iou_threshold if args.eval_iou_threshold is not None else cfg.evaluator.iou_threshold)
    if args.eval_metrics or args.eval_ab3dmot:
        metrics_path = output_dir / f"{args.variant}_{args.split}_tracking_metrics.json"
        metrics = evaluate_tracking_json(
            output_path,
            info_path,
            class_names=list(cfg.dataset.class_names),
            iou_threshold=eval_iou,
            output_path=metrics_path,
        )
        print(
            "metrics "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"ap_3d_iou_0_50={metrics['ap_3d_iou_0_50']:.4f} "
            f"mota={metrics['mota']:.4f} "
            f"id_switches={metrics['id_switches']:.0f}",
            flush=True,
        )
    if args.eval_ab3dmot:
        ab3d_path = output_dir / f"ab3dmot_{args.split}_metrics_ab3dmot.json"
        ab3d = evaluate_ab3dmot_json(
            output_path,
            info_path,
            class_names=list(cfg.dataset.class_names),
            iou_threshold=eval_iou,
            recall_points=int(args.ab3dmot_recall_points),
            output_path=ab3d_path,
        )
        print(
            "ab3dmot "
            f"sAMOTA={float(ab3d['sAMOTA']):.4f} "
            f"AMOTA={float(ab3d['AMOTA']):.4f} "
            f"AMOTP={float(ab3d['AMOTP']):.4f} "
            f"MOTA={float(ab3d['MOTA']):.4f} "
            f"MOTP={float(ab3d['MOTP']):.4f} "
            f"IDS={float(ab3d['IDS']):.0f} "
            f"FRAG={float(ab3d['FRAG']):.0f}",
            flush=True,
        )


def run_tracking(
    *,
    cfg: Any,
    split: str,
    variant: str,
    score_thresh: float,
    high_score_thresh: float,
    low_score_thresh: float,
    max_dets_per_frame: int,
    iou_threshold: float,
    center_distance: float,
    tlom_threshold: float,
    max_lost_frames: int,
    min_hits: int,
    dt_hypotheses: list[float] | None,
    init_velocity_mode: str,
    init_speed_prior: float,
    max_frames: int,
    progress_interval: int,
) -> tuple[list[dict[str, Any]], Path]:
    detections = load_detection_cache(cfg, split)
    info_path = Path(resolve_info_path(cfg, split))
    with info_path.open("rb") as f:
        infos = pickle.load(f)
    ordered_infos = sorted(infos, key=lambda x: (str(x.get("sequence_id", "")), int(x.get("frame_idx", 0))))
    if max_frames > 0:
        ordered_infos = ordered_infos[:max_frames]

    tracker = Chapter5Tracker(
        class_names=list(cfg.dataset.class_names),
        variant=variant,
        max_lost_frames=max_lost_frames,
        min_hits=min_hits,
        high_score_thresh=high_score_thresh,
        low_score_thresh=low_score_thresh,
        iou_threshold=iou_threshold,
        center_distance=center_distance,
        tlom_threshold=tlom_threshold,
        dt_hypotheses=dt_hypotheses,
        init_velocity_mode=init_velocity_mode,
        init_speed_prior=init_speed_prior,
    )
    outputs: list[dict[str, Any]] = []
    current_sequence = None
    progress = tqdm(total=len(ordered_infos), desc=f"chapter5 {variant} {split}", dynamic_ncols=True, leave=True) if tqdm is not None else None
    for frame_count, info in enumerate(ordered_infos, start=1):
        sequence_id = str(info.get("sequence_id", ""))
        frame_id = str(info.get("frame_id", info.get("frame_idx", "")))
        if sequence_id != current_sequence:
            tracker.reset()
            current_sequence = sequence_id
        det = detections.get((sequence_id, frame_id))
        if det is None:
            det = DetectionFrame(
                sequence_id=sequence_id,
                frame_id=frame_id,
                frame_idx=int(info.get("frame_idx", 0)),
                pred_boxes=np.zeros((0, 7), dtype=np.float32),
                pred_scores=np.zeros((0,), dtype=np.float32),
                pred_labels=np.zeros((0,), dtype=np.int64),
            )
        det = _filter_for_tracking(det, score_thresh=min(score_thresh, low_score_thresh), max_dets_per_frame=max_dets_per_frame)
        outputs.append(tracker.update(det))
        if progress is not None:
            progress.update(1)
            progress.set_postfix(frame=f"{frame_count}/{len(ordered_infos)}")
        elif frame_count == 1 or frame_count % progress_interval == 0 or frame_count == len(ordered_infos):
            print(f"chapter5 tracking progress frames={frame_count}/{len(ordered_infos)}", flush=True)
    if progress is not None:
        progress.close()
    return outputs, info_path


def _filter_for_tracking(det: DetectionFrame, *, score_thresh: float, max_dets_per_frame: int) -> DetectionFrame:
    frame = filter_detection_frame(
        {
            "sequence_id": det.sequence_id,
            "frame_id": det.frame_id,
            "frame_idx": det.frame_idx,
            "pred_boxes": det.pred_boxes,
            "pred_scores": det.pred_scores,
            "pred_labels": det.pred_labels,
        },
        class_id=0,
        score_thresh=score_thresh,
        max_dets=max_dets_per_frame,
    )
    return DetectionFrame(
        sequence_id=str(frame["sequence_id"]),
        frame_id=str(frame["frame_id"]),
        frame_idx=int(frame["frame_idx"]),
        pred_boxes=np.asarray(frame["pred_boxes"], dtype=np.float32).reshape(-1, 7),
        pred_scores=np.asarray(frame["pred_scores"], dtype=np.float32).reshape(-1),
        pred_labels=np.asarray(frame["pred_labels"], dtype=np.int64).reshape(-1),
    )


def _subset_detection(det: DetectionFrame, mask: np.ndarray) -> DetectionFrame:
    indices = np.where(np.asarray(mask, dtype=bool))[0]
    return DetectionFrame(
        sequence_id=det.sequence_id,
        frame_id=det.frame_id,
        frame_idx=det.frame_idx,
        pred_boxes=det.pred_boxes[indices].astype(np.float32),
        pred_scores=det.pred_scores[indices].astype(np.float32),
        pred_labels=det.pred_labels[indices].astype(np.int64),
    )


def _remove_indices(det: DetectionFrame, used: set[int]) -> tuple[DetectionFrame, np.ndarray]:
    indices = [idx for idx in range(len(det.pred_boxes)) if idx not in used]
    out = _subset_detection(det, np.asarray([idx in indices for idx in range(len(det.pred_boxes))], dtype=bool))
    return out, np.asarray(indices, dtype=np.int64)


def _center_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    diff = boxes_a[:, None, :3] - boxes_b[None, :, :3]
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def _write_runtime_summary(path: Path, *, method: str, frames: int, tracking_seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(frames) / max(float(tracking_seconds), 1e-12)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": method,
                "num_frames": int(frames),
                "tracking_time_s": float(tracking_seconds),
                "tracking_fps": fps,
                "tracking_ms_per_frame": 1000.0 / max(fps, 1e-12),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
