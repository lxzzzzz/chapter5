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

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


@dataclass
class AB3DTrack:
    track_id: int
    box: np.ndarray
    velocity: np.ndarray
    score: float
    class_id: int
    lost_frames: int = 0
    age: int = 1
    hits: int = 1

    def predicted_box(self) -> np.ndarray:
        pred = np.asarray(self.box, dtype=np.float32).copy()
        pred[:3] += self.velocity[:3]
        pred[6] += self.velocity[6]
        return pred


class SimpleAB3DMOT:
    """Dependency-light AB3DMOT-style tracker using 3D IoU assignment."""

    def __init__(
        self,
        *,
        class_names: list[str],
        iou_threshold: float,
        max_lost_frames: int,
        min_hits: int,
    ) -> None:
        self.class_names = class_names
        self.iou_threshold = float(iou_threshold)
        self.max_lost_frames = int(max_lost_frames)
        self.min_hits = int(min_hits)
        self.next_id = 0
        self.tracks: dict[int, AB3DTrack] = {}

    def reset(self) -> None:
        self.next_id = 0
        self.tracks.clear()

    def update(self, det: DetectionFrame) -> dict[str, Any]:
        active_ids = sorted(self.tracks)
        pred_boxes = np.stack([self.tracks[tid].predicted_box() for tid in active_ids]).astype(np.float32) if active_ids else np.zeros((0, 7), dtype=np.float32)
        iou = boxes_iou_3d_axis_aligned(pred_boxes, det.pred_boxes)
        if iou.size:
            for row, track_id in enumerate(active_ids):
                track_cls = int(self.tracks[track_id].class_id)
                iou[row, det.pred_labels != track_cls] = 0.0
        matches, unmatched_tracks, unmatched_dets = match_by_iou(iou, self.iou_threshold)

        for track_pos, det_idx, _score in matches:
            track_id = active_ids[track_pos]
            self._update_track(track_id, det, det_idx)

        for det_idx in unmatched_dets:
            self._start_track(det, det_idx)

        matched_track_pos = {track_pos for track_pos, _det_idx, _score in matches}
        for track_pos in unmatched_tracks:
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

    def _start_track(self, det: DetectionFrame, det_idx: int) -> None:
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = AB3DTrack(
            track_id=track_id,
            box=det.pred_boxes[det_idx].astype(np.float32),
            velocity=np.zeros((7,), dtype=np.float32),
            score=float(det.pred_scores[det_idx]),
            class_id=int(det.pred_labels[det_idx]),
        )

    def _update_track(self, track_id: int, det: DetectionFrame, det_idx: int) -> None:
        track = self.tracks[track_id]
        new_box = det.pred_boxes[det_idx].astype(np.float32)
        track.velocity = new_box - track.box
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
    parser = argparse.ArgumentParser(description="Run a pure AB3DMOT-style baseline on cached detections")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--dataset_name", default=None, help="Override cfg.dataset.name, e.g. xian, v2x_seq, v2x_real.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--detection_cache_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--score_thresh", type=float, default=None)
    parser.add_argument("--max_dets_per_frame", type=int, default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.1, help="3D IoU threshold for tracker association.")
    parser.add_argument("--eval_iou_threshold", type=float, default=None, help="3D IoU threshold for metric matching.")
    parser.add_argument("--max_lost_frames", type=int, default=2)
    parser.add_argument("--min_hits", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--progress_interval", type=int, default=50)
    parser.add_argument("--eval_metrics", action="store_true")
    parser.add_argument("--eval_ab3dmot", action="store_true")
    parser.add_argument("--ab3dmot_recall_points", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    if args.dataset_name:
        cfg.dataset.name = str(args.dataset_name)
    if args.detection_cache_root:
        cfg.detection_cache.root = str(args.detection_cache_root)
    if args.output_dir:
        cfg.output_dir = str(args.output_dir)
    if args.score_thresh is not None:
        cfg.eval.score_thresh = float(args.score_thresh)

    output_dir = Path(args.output_dir or cfg.output_dir)
    output_path = Path(args.output) if args.output else output_dir / f"ab3dmot_{args.split}_tracking_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    outputs, info_path = run_ab3dmot(
        cfg=cfg,
        split=str(args.split),
        score_thresh=float(args.score_thresh if args.score_thresh is not None else cfg.eval.get("score_thresh", 0.0)),
        max_dets_per_frame=int(args.max_dets_per_frame if args.max_dets_per_frame is not None else cfg.detection_cache.get("max_dets_per_frame", 100)),
        iou_threshold=float(args.iou_threshold),
        max_lost_frames=int(args.max_lost_frames),
        min_hits=int(args.min_hits),
        max_frames=int(args.max_frames),
        progress_interval=max(1, int(args.progress_interval)),
    )
    tracking_seconds = time.perf_counter() - start_time
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    print(f"wrote {len(outputs)} frames to {output_path}", flush=True)
    _write_runtime_summary(
        output_dir / "runtime_summary.json",
        method="ab3dmot",
        frames=len(outputs),
        tracking_seconds=tracking_seconds,
    )

    eval_iou = float(args.eval_iou_threshold if args.eval_iou_threshold is not None else cfg.evaluator.iou_threshold)
    if args.eval_metrics or args.eval_ab3dmot:
        metrics_path = output_dir / f"ab3dmot_{args.split}_tracking_metrics.json"
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


def run_ab3dmot(
    *,
    cfg: Any,
    split: str,
    score_thresh: float,
    max_dets_per_frame: int,
    iou_threshold: float,
    max_lost_frames: int,
    min_hits: int,
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

    tracker = SimpleAB3DMOT(
        class_names=list(cfg.dataset.class_names),
        iou_threshold=iou_threshold,
        max_lost_frames=max_lost_frames,
        min_hits=min_hits,
    )
    outputs: list[dict[str, Any]] = []
    current_sequence = None
    progress = tqdm(total=len(ordered_infos), desc=f"ab3dmot {split}", dynamic_ncols=True, leave=True) if tqdm is not None else None
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
        det = _filter_for_tracking(det, score_thresh=score_thresh, max_dets_per_frame=max_dets_per_frame)
        outputs.append(tracker.update(det))
        if progress is not None:
            progress.update(1)
            progress.set_postfix(frame=f"{frame_count}/{len(ordered_infos)}")
        elif frame_count == 1 or frame_count % progress_interval == 0 or frame_count == len(ordered_infos):
            print(f"ab3dmot progress frames={frame_count}/{len(ordered_infos)}", flush=True)
    if progress is not None:
        progress.close()
    return outputs, info_path


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


def match_by_iou(iou: np.ndarray, threshold: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    scores = np.asarray(iou, dtype=np.float32)
    num_tracks, num_dets = scores.shape if scores.ndim == 2 else (0, 0)
    if num_tracks == 0:
        return [], [], list(range(num_dets))
    if num_dets == 0:
        return [], list(range(num_tracks)), []
    try:
        from scipy.optimize import linear_sum_assignment

        track_idx, det_idx = linear_sum_assignment(-scores)
        raw_matches = [(int(t), int(d), float(scores[t, d])) for t, d in zip(track_idx.tolist(), det_idx.tolist())]
    except ImportError:
        raw_matches = _greedy_match(scores)
    matches = [(t, d, s) for t, d, s in raw_matches if s >= float(threshold)]
    matched_tracks = {t for t, _d, _s in matches}
    matched_dets = {d for _t, d, _s in matches}
    unmatched_tracks = [idx for idx in range(num_tracks) if idx not in matched_tracks]
    unmatched_dets = [idx for idx in range(num_dets) if idx not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def _greedy_match(scores: np.ndarray) -> list[tuple[int, int, float]]:
    matches: list[tuple[int, int, float]] = []
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    flat_order = np.argsort(scores.reshape(-1))[::-1]
    num_dets = scores.shape[1]
    for flat_idx in flat_order.tolist():
        track_idx = flat_idx // num_dets
        det_idx = flat_idx % num_dets
        if track_idx in used_tracks or det_idx in used_dets:
            continue
        used_tracks.add(track_idx)
        used_dets.add(det_idx)
        matches.append((track_idx, det_idx, float(scores[track_idx, det_idx])))
    return matches


if __name__ == "__main__":
    main()
