from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config, resolve_info_path
from .nova_data import DetectionFrame, NOVAFormulator, load_detection_cache

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


@dataclass
class NOVATrackState:
    track_id: int
    boxes: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    lost_frames: int = 0


class NOVAOnlineTracker:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.history_len = int(cfg.nova.get("history_len", 3))
        self.history_stride = max(1, int(cfg.nova.get("history_stride", 1)))
        self.association_threshold = float(cfg.nova.get("association_threshold", 0.5))
        self.max_lost_frames = int(cfg.nova.get("max_lost_frames", cfg.eval.get("max_lost_frames", 2)))
        self.class_names = list(cfg.dataset.class_names)
        self.formulator = NOVAFormulator(cfg)
        self.next_id = 0
        self.tracks: dict[int, NOVATrackState] = {}

    def reset(self) -> None:
        self.next_id = 0
        self.tracks.clear()

    def update(self, det: DetectionFrame, model: torch.nn.Module) -> dict[str, Any]:
        active_ids = sorted(self.tracks)
        score_matrix = self._score_pairs(det, active_ids, model)
        matches, unmatched_tracks, unmatched_dets = associate_by_scores(
            score_matrix,
            threshold=self.association_threshold,
        )
        outputs: list[dict[str, Any]] = []
        assigned_track_ids: set[int] = set()
        for track_pos, det_idx, match_score in matches:
            track_id = active_ids[track_pos]
            self._append_detection(track_id, det, det_idx)
            assigned_track_ids.add(track_id)
            outputs.append(self._output_track(track_id, det, det_idx, match_score))

        for det_idx in unmatched_dets:
            track_id = self._new_track_id()
            self._append_detection(track_id, det, det_idx)
            assigned_track_ids.add(track_id)
            outputs.append(self._output_track(track_id, det, det_idx, 1.0))

        unmatched_track_ids = {active_ids[idx] for idx in unmatched_tracks if idx < len(active_ids)}
        for track_id in sorted(unmatched_track_ids - assigned_track_ids):
            if track_id not in self.tracks:
                continue
            self.tracks[track_id].lost_frames += 1
            if self.tracks[track_id].lost_frames > self.max_lost_frames:
                del self.tracks[track_id]

        outputs.sort(key=lambda item: int(item["id"]))
        return {"sequence_id": det.sequence_id, "frame_id": det.frame_id, "tracks": outputs}

    def _score_pairs(self, det: DetectionFrame, active_ids: list[int], model: torch.nn.Module) -> np.ndarray:
        if len(active_ids) == 0 or len(det.pred_boxes) == 0:
            return np.zeros((len(active_ids), len(det.pred_boxes)), dtype=np.float32)
        rows: list[dict[str, Any]] = []
        for track_id in active_ids:
            hist = self._history_tensors(track_id)
            for det_idx in range(len(det.pred_boxes)):
                row = dict(hist)
                prompt_text = self.formulator.build_prompt(
                    track_id=int(track_id),
                    history_mask=hist["track_history_mask"],
                    history_class_ids=hist["track_history_class_ids"],
                    candidate_class_id=int(det.pred_labels[det_idx]),
                )
                row.update(
                    {
                        "candidate_box": det.pred_boxes[det_idx],
                        "candidate_score": np.float32(det.pred_scores[det_idx]),
                        "candidate_class_id": np.int64(det.pred_labels[det_idx]),
                        "prompt_text": prompt_text,
                        "box_token_mask": np.concatenate([hist["track_history_mask"], np.asarray([True], dtype=np.bool_)]),
                    }
                )
                rows.append(row)
        batch = _runtime_collate(rows, self.device)
        with torch.inference_mode():
            out = model(batch)
        scores = out["match_prob"].detach().float().cpu().numpy()
        return scores.reshape(len(active_ids), len(det.pred_boxes)).astype(np.float32)

    def _history_tensors(self, track_id: int) -> dict[str, np.ndarray]:
        state = self.tracks[track_id]
        boxes = np.zeros((self.history_len, 7), dtype=np.float32)
        scores = np.zeros((self.history_len,), dtype=np.float32)
        class_ids = np.zeros((self.history_len,), dtype=np.int64)
        mask = np.zeros((self.history_len,), dtype=np.bool_)
        selected = list(range(len(state.boxes) - 1, -1, -self.history_stride))[: self.history_len]
        selected.reverse()
        offset = self.history_len - len(selected)
        for idx, source_idx in enumerate(selected):
            slot = offset + idx
            boxes[slot] = np.asarray(state.boxes[source_idx], dtype=np.float32)
            scores[slot] = float(state.scores[source_idx])
            class_ids[slot] = int(state.class_ids[source_idx])
            mask[slot] = True
        return {
            "track_history_boxes": boxes,
            "track_history_scores": scores,
            "track_history_class_ids": class_ids,
            "track_history_mask": mask,
        }

    def _append_detection(self, track_id: int, det: DetectionFrame, det_idx: int) -> None:
        if track_id not in self.tracks:
            self.tracks[track_id] = NOVATrackState(track_id=track_id)
        state = self.tracks[track_id]
        state.boxes.append(det.pred_boxes[det_idx].astype(np.float32))
        state.scores.append(float(det.pred_scores[det_idx]))
        state.class_ids.append(int(det.pred_labels[det_idx]))
        state.lost_frames = 0

    def _output_track(self, track_id: int, det: DetectionFrame, det_idx: int, match_score: float) -> dict[str, Any]:
        class_id = int(det.pred_labels[det_idx])
        class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else "Unknown"
        score = float(det.pred_scores[det_idx]) * max(float(match_score), 0.0)
        return {
            "id": int(track_id),
            "class": str(class_name),
            "box3d": [float(x) for x in det.pred_boxes[det_idx].tolist()],
            "score": score,
            "state": "active",
        }

    def _new_track_id(self) -> int:
        while self.next_id in self.tracks:
            self.next_id += 1
        track_id = self.next_id
        self.next_id += 1
        return track_id


def associate_by_scores(score_matrix: np.ndarray, threshold: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    scores = np.asarray(score_matrix, dtype=np.float32)
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
        raw_matches = _greedy_score_match(scores)
    matches = [(t, d, s) for t, d, s in raw_matches if s >= float(threshold)]
    matched_tracks = {t for t, _d, _s in matches}
    matched_dets = {d for _t, d, _s in matches}
    unmatched_tracks = [idx for idx in range(num_tracks) if idx not in matched_tracks]
    unmatched_dets = [idx for idx in range(num_dets) if idx not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def _greedy_score_match(scores: np.ndarray) -> list[tuple[int, int, float]]:
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


def _runtime_collate(rows: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "track_history_boxes": torch.as_tensor(np.stack([row["track_history_boxes"] for row in rows]), dtype=torch.float32, device=device),
        "track_history_scores": torch.as_tensor(np.stack([row["track_history_scores"] for row in rows]), dtype=torch.float32, device=device),
        "track_history_class_ids": torch.as_tensor(np.stack([row["track_history_class_ids"] for row in rows]), dtype=torch.long, device=device),
        "track_history_mask": torch.as_tensor(np.stack([row["track_history_mask"] for row in rows]), dtype=torch.bool, device=device),
        "candidate_box": torch.as_tensor(np.stack([row["candidate_box"] for row in rows]), dtype=torch.float32, device=device),
        "candidate_score": torch.as_tensor([row["candidate_score"] for row in rows], dtype=torch.float32, device=device),
        "candidate_class_id": torch.as_tensor([row["candidate_class_id"] for row in rows], dtype=torch.long, device=device),
        "prompt_texts": [str(row.get("prompt_text", "")) for row in rows],
        "box_token_mask": torch.as_tensor(
            np.stack(
                [
                    row.get(
                        "box_token_mask",
                        np.concatenate([row["track_history_mask"], np.asarray([True], dtype=np.bool_)]),
                    )
                    for row in rows
                ]
            ),
            dtype=torch.bool,
            device=device,
        ),
    }


def run_nova_tracking(
    *,
    cfg: Config,
    model: torch.nn.Module,
    device: torch.device,
    split: str = "val",
    max_frames: int = 0,
    progress_interval: int = 50,
    use_tqdm: bool = False,
    desc: str = "nova eval",
) -> tuple[list[dict[str, Any]], Path]:
    detections = load_detection_cache(cfg, split)
    info_path = Path(resolve_info_path(cfg, split))
    import pickle

    with info_path.open("rb") as f:
        infos = pickle.load(f)
    ordered_infos = sorted(infos, key=lambda x: (str(x.get("sequence_id", "")), int(x.get("frame_idx", 0))))
    if max_frames > 0:
        ordered_infos = ordered_infos[:max_frames]

    tracker = NOVAOnlineTracker(cfg, device)
    outputs: list[dict[str, Any]] = []
    current_sequence = None
    model.eval()
    iterator = enumerate(ordered_infos, start=1)
    progress_bar = tqdm(total=len(ordered_infos), desc=desc, dynamic_ncols=True, leave=True) if use_tqdm and tqdm is not None else None
    for frame_count, info in iterator:
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
        else:
            det = filter_eval_detections(det, float(cfg.eval.get("score_thresh", 0.0)))
        outputs.append(tracker.update(det, model))
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(frame=f"{frame_count}/{len(ordered_infos)}")
        elif progress_interval > 0 and (frame_count == 1 or frame_count % progress_interval == 0 or frame_count == len(ordered_infos)):
            print(f"nova eval progress frames={frame_count}/{len(ordered_infos)}", flush=True)
    if progress_bar is not None:
        progress_bar.close()
    return outputs, info_path


def filter_eval_detections(det: DetectionFrame, score_thresh: float) -> DetectionFrame:
    if score_thresh <= 0.0 or len(det.pred_scores) == 0:
        return det
    keep = np.asarray(det.pred_scores >= float(score_thresh), dtype=bool)
    return DetectionFrame(
        sequence_id=det.sequence_id,
        frame_id=det.frame_id,
        frame_idx=det.frame_idx,
        pred_boxes=det.pred_boxes[keep],
        pred_scores=det.pred_scores[keep],
        pred_labels=det.pred_labels[keep],
    )
