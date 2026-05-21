#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from generative_tracking.ab3dmot_evaluator import evaluate_ab3dmot_json
from generative_tracking.config import load_config, resolve_info_path
from generative_tracking.evaluator import evaluate_tracking_json
from generative_tracking.geometry import boxes_iou_3d_axis_aligned
from generative_tracking.nova_data import DetectionFrame, filter_detection_frame, load_detection_cache
from tools.run_ab3dmot_tracking import AB3DTrack, _initial_velocity_from_box, _normalize_dt_hypotheses, match_by_iou

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


@dataclass(frozen=True)
class DetectionSubset:
    frame: DetectionFrame


class HierarchicalTracker:
    """Tracking-by-detection baseline for Chapter 5 association ablations."""

    def __init__(
        self,
        *,
        class_names: list[str],
        mode: str,
        stage1_iou_threshold: float,
        stage2_center_distance: float,
        max_lost_frames: int,
        min_hits: int,
        dt_hypotheses: list[float] | None = None,
        init_velocity_mode: str = "zero",
        init_speed_prior: float = 0.0,
    ) -> None:
        if mode not in {"stage1", "stage1_stage2"}:
            raise ValueError(f"Unsupported hierarchical mode: {mode}")
        self.class_names = class_names
        self.mode = mode
        self.stage1_iou_threshold = float(stage1_iou_threshold)
        self.stage2_center_distance = float(stage2_center_distance)
        self.max_lost_frames = int(max_lost_frames)
        self.min_hits = int(min_hits)
        self.dt_hypotheses = _normalize_dt_hypotheses(dt_hypotheses)
        self.init_velocity_mode = str(init_velocity_mode)
        self.init_speed_prior = float(init_speed_prior)
        self.next_id = 0
        self.tracks: dict[int, AB3DTrack] = {}

    def reset(self) -> None:
        self.next_id = 0
        self.tracks.clear()

    def update(self, det: DetectionFrame, *, high_score_thresh: float, low_score_thresh: float) -> dict[str, Any]:
        high = _subset_detection(det, det.pred_scores >= float(high_score_thresh))
        low = _subset_detection((det), (det.pred_scores >= float(low_score_thresh)) & (det.pred_scores < float(high_score_thresh)))

        active_ids = sorted(self.tracks)
        stage1_matches, unmatched_track_pos, unmatched_high = self._match_high(active_ids, high.frame)
        matched_track_pos: set[int] = set()

        for track_pos, det_idx, _score, dt_scale in stage1_matches:
            track_id = active_ids[track_pos]
            self._update_track(track_id, high.frame, det_idx, dt_scale=dt_scale)
            matched_track_pos.add(track_pos)

        if self.mode == "stage1_stage2" and unmatched_track_pos and len(low.frame.pred_boxes):
            stage2_matches, still_unmatched_tracks = self._match_low(active_ids, unmatched_track_pos, low.frame)
            unmatched_track_pos = still_unmatched_tracks
            for track_pos, det_idx, _score, dt_scale in stage2_matches:
                track_id = active_ids[track_pos]
                self._update_track(track_id, low.frame, det_idx, dt_scale=dt_scale)
                matched_track_pos.add(track_pos)

        for det_idx in unmatched_high:
            self._start_track(high.frame, det_idx)

        for track_pos in unmatched_track_pos:
            if track_pos in matched_track_pos or track_pos >= len(active_ids):
                continue
            track_id = active_ids[track_pos]
            track = self.tracks.get(track_id)
            if track is None:
                continue
            track.box = track.predicted_box()
            track.lost_frames += 1
            track.age += 1
            if track.lost_frames > self.max_lost_frames:
                del self.tracks[track_id]

        outputs = [self._track_output(track) for track in self.tracks.values() if track.lost_frames == 0 and track.hits >= self.min_hits]
        outputs.sort(key=lambda item: int(item["id"]))
        return {"sequence_id": det.sequence_id, "frame_id": det.frame_id, "tracks": outputs}

    def _match_high(self, active_ids: list[int], det: DetectionFrame) -> tuple[list[tuple[int, int, float, float]], list[int], list[int]]:
        if not active_ids:
            return [], [], list(range(len(det.pred_boxes)))
        if len(det.pred_boxes) == 0:
            return [], list(range(len(active_ids))), []
        matches: list[tuple[int, int, float, float]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for dt_scale in self.dt_hypotheses:
            remaining_track_pos = [idx for idx in range(len(active_ids)) if idx not in matched_tracks]
            remaining_det_idx = [idx for idx in range(len(det.pred_boxes)) if idx not in matched_dets]
            if not remaining_track_pos or not remaining_det_idx:
                break
            track_ids = [active_ids[pos] for pos in remaining_track_pos]
            pred_boxes = _predicted_boxes(self.tracks, track_ids, dt_scale=dt_scale)
            iou = boxes_iou_3d_axis_aligned(pred_boxes, det.pred_boxes[remaining_det_idx])
            _mask_class_mismatch(iou, self.tracks, track_ids, det.pred_labels[remaining_det_idx])
            local_matches, _unmatched_tracks, _unmatched_dets = match_by_iou(iou, self.stage1_iou_threshold)
            for local_track_pos, local_det_idx, score in local_matches:
                track_pos = remaining_track_pos[local_track_pos]
                det_idx = remaining_det_idx[local_det_idx]
                matches.append((track_pos, det_idx, score, float(dt_scale)))
                matched_tracks.add(track_pos)
                matched_dets.add(det_idx)
        unmatched_tracks = [idx for idx in range(len(active_ids)) if idx not in matched_tracks]
        unmatched_dets = [idx for idx in range(len(det.pred_boxes)) if idx not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _match_low(
        self,
        active_ids: list[int],
        unmatched_track_pos: list[int],
        det: DetectionFrame,
    ) -> tuple[list[tuple[int, int, float, float]], list[int]]:
        if not unmatched_track_pos or len(det.pred_boxes) == 0:
            return [], unmatched_track_pos
        matches: list[tuple[int, int, float, float]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        distance_by_match: dict[tuple[int, int], float] = {}
        for dt_scale in self.dt_hypotheses:
            remaining_local_pos = [idx for idx in range(len(unmatched_track_pos)) if idx not in matched_tracks]
            remaining_det_idx = [idx for idx in range(len(det.pred_boxes)) if idx not in matched_dets]
            if not remaining_local_pos or not remaining_det_idx:
                break
            track_ids = [active_ids[unmatched_track_pos[pos]] for pos in remaining_local_pos]
            pred_boxes = _predicted_boxes(self.tracks, track_ids, dt_scale=dt_scale)
            distance = _center_distance(pred_boxes, det.pred_boxes[remaining_det_idx])
            score = np.maximum(0.0, 1.0 - distance / max(self.stage2_center_distance, 1e-6)).astype(np.float32)
            _mask_class_mismatch(score, self.tracks, track_ids, det.pred_labels[remaining_det_idx])
            local_matches, _unmatched_tracks, _unmatched_dets = match_by_iou(score, 1e-6)
            for row, col, match_score in local_matches:
                local_track_pos = remaining_local_pos[row]
                det_idx = remaining_det_idx[col]
                matches.append((local_track_pos, det_idx, match_score, float(dt_scale)))
                matched_tracks.add(local_track_pos)
                matched_dets.add(det_idx)
                distance_by_match[(local_track_pos, det_idx)] = float(distance[row, col])
        accepted = []
        for local_track_pos, det_idx, match_score, dt_scale in matches:
            if distance_by_match.get((local_track_pos, det_idx), float("inf")) <= self.stage2_center_distance:
                accepted.append((unmatched_track_pos[local_track_pos], det_idx, match_score, dt_scale))
        accepted_local = {local_track_pos for local_track_pos, det_idx, _score, _dt_scale in matches if distance_by_match.get((local_track_pos, det_idx), float("inf")) <= self.stage2_center_distance}
        still_unmatched = [unmatched_track_pos[idx] for idx in range(len(unmatched_track_pos)) if idx not in accepted_local]
        return accepted, still_unmatched

    def _start_track(self, det: DetectionFrame, det_idx: int) -> None:
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = AB3DTrack(
            track_id=track_id,
            box=det.pred_boxes[det_idx].astype(np.float32),
            velocity=_initial_velocity_from_box(
                det.pred_boxes[det_idx],
                mode=self.init_velocity_mode,
                speed_prior=self.init_speed_prior,
            ),
            score=float(det.pred_scores[det_idx]),
            class_id=int(det.pred_labels[det_idx]),
        )

    def _update_track(self, track_id: int, det: DetectionFrame, det_idx: int, *, dt_scale: float) -> None:
        track = self.tracks[track_id]
        new_box = det.pred_boxes[det_idx].astype(np.float32)
        track.velocity = (new_box - track.box) / max(float(dt_scale), 1e-6)
        track.box = new_box
        track.score = float(det.pred_scores[det_idx])
        track.class_id = int(det.pred_labels[det_idx])
        track.lost_frames = 0
        track.age += 1
        track.hits += 1

    def _track_output(self, track: AB3DTrack) -> dict[str, Any]:
        class_name = self.class_names[track.class_id] if 0 <= track.class_id < len(self.class_names) else "Unknown"
        return {
            "id": int(track.track_id),
            "class": str(class_name),
            "box3d": [float(x) for x in track.box.tolist()],
            "score": float(track.score),
            "state": "active",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 5 hierarchical association ablation tracker")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--detection_cache_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["stage1", "stage1_stage2"], default="stage1_stage2")
    parser.add_argument("--score_thresh", type=float, default=0.6, help="Minimum score loaded for any candidate.")
    parser.add_argument("--high_score_thresh", type=float, default=0.6)
    parser.add_argument("--low_score_thresh", type=float, default=0.3)
    parser.add_argument("--max_dets_per_frame", type=int, default=100)
    parser.add_argument("--stage1_iou_threshold", type=float, default=0.1)
    parser.add_argument("--stage2_center_distance", type=float, default=4.0)
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
    output_path = Path(args.output) if args.output else output_dir / f"hierarchical_{args.mode}_{args.split}_tracking_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    outputs, info_path = run_hierarchical(
        cfg=cfg,
        split=str(args.split),
        mode=str(args.mode),
        score_thresh=float(args.score_thresh),
        high_score_thresh=float(args.high_score_thresh),
        low_score_thresh=float(args.low_score_thresh),
        max_dets_per_frame=int(args.max_dets_per_frame),
        stage1_iou_threshold=float(args.stage1_iou_threshold),
        stage2_center_distance=float(args.stage2_center_distance),
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
        method=f"hierarchical_{args.mode}",
        frames=len(outputs),
        tracking_seconds=tracking_seconds,
    )

    eval_iou = float(args.eval_iou_threshold if args.eval_iou_threshold is not None else cfg.evaluator.iou_threshold)
    if args.eval_metrics or args.eval_ab3dmot:
        metrics_path = output_dir / f"hierarchical_{args.mode}_{args.split}_tracking_metrics.json"
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


def run_hierarchical(
    *,
    cfg: Any,
    split: str,
    mode: str,
    score_thresh: float,
    high_score_thresh: float,
    low_score_thresh: float,
    max_dets_per_frame: int,
    stage1_iou_threshold: float,
    stage2_center_distance: float,
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

    tracker = HierarchicalTracker(
        class_names=list(cfg.dataset.class_names),
        mode=mode,
        stage1_iou_threshold=stage1_iou_threshold,
        stage2_center_distance=stage2_center_distance,
        max_lost_frames=max_lost_frames,
        min_hits=min_hits,
        dt_hypotheses=dt_hypotheses,
        init_velocity_mode=init_velocity_mode,
        init_speed_prior=init_speed_prior,
    )
    outputs: list[dict[str, Any]] = []
    current_sequence = None
    progress = tqdm(total=len(ordered_infos), desc=f"hierarchical {mode} {split}", dynamic_ncols=True, leave=True) if tqdm is not None else None
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
        outputs.append(tracker.update(det, high_score_thresh=high_score_thresh, low_score_thresh=low_score_thresh))
        if progress is not None:
            progress.update(1)
            progress.set_postfix(frame=f"{frame_count}/{len(ordered_infos)}")
        elif frame_count == 1 or frame_count % progress_interval == 0 or frame_count == len(ordered_infos):
            print(f"hierarchical progress frames={frame_count}/{len(ordered_infos)}", flush=True)
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


def _subset_detection(det: DetectionFrame, mask: np.ndarray) -> DetectionSubset:
    indices = np.where(np.asarray(mask, dtype=bool))[0]
    return DetectionSubset(
        frame=DetectionFrame(
            sequence_id=det.sequence_id,
            frame_id=det.frame_id,
            frame_idx=det.frame_idx,
            pred_boxes=det.pred_boxes[indices].astype(np.float32),
            pred_scores=det.pred_scores[indices].astype(np.float32),
            pred_labels=det.pred_labels[indices].astype(np.int64),
        ),
    )


def _predicted_boxes(tracks: dict[int, AB3DTrack], track_ids: list[int], *, dt_scale: float = 1.0) -> np.ndarray:
    if not track_ids:
        return np.zeros((0, 7), dtype=np.float32)
    return np.stack([tracks[tid].predicted_box(dt_scale=dt_scale) for tid in track_ids]).astype(np.float32)


def _mask_class_mismatch(scores: np.ndarray, tracks: dict[int, AB3DTrack], track_ids: list[int], pred_labels: np.ndarray) -> None:
    if scores.size == 0:
        return
    for row, track_id in enumerate(track_ids):
        track_cls = int(tracks[track_id].class_id)
        scores[row, pred_labels != track_cls] = 0.0


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
