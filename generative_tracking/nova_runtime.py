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
        scores = _runtime_predict_match_prob(rows, self.device, model, _runtime_batch_size(self.cfg))
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
        det_score = float(det.pred_scores[det_idx])
        score_mode = str(self.cfg.nova.get("output_score_mode", "det_assoc_product")).lower()
        if score_mode in {"detector", "det", "detection"}:
            score = det_score
        elif score_mode in {"match", "association", "assoc"}:
            score = max(float(match_score), 0.0)
        else:
            score = det_score * max(float(match_score), 0.0)
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


class NOVALifecycleOnlineTracker(NOVAOnlineTracker):
    """NOVA V2 online tracker with LLM Birth/Suppress and Keep/End decisions."""

    def update(self, det: DetectionFrame, model: torch.nn.Module) -> dict[str, Any]:
        active_ids = sorted(self.tracks)
        score_matrix = self._score_pairs(det, active_ids, model)
        matches, unmatched_tracks, unmatched_dets = associate_by_scores(score_matrix, threshold=self.association_threshold)
        outputs: list[dict[str, Any]] = []
        assigned_track_ids: set[int] = set()
        for track_pos, det_idx, match_score in matches:
            track_id = active_ids[track_pos]
            self._append_detection(track_id, det, det_idx)
            assigned_track_ids.add(track_id)
            outputs.append(self._output_track(track_id, det, det_idx, match_score))

        birth_decisions = self._birth_decisions(det, unmatched_dets, score_matrix, model)
        for det_idx, should_birth in birth_decisions.items():
            if not should_birth:
                continue
            track_id = self._new_track_id()
            self._append_detection(track_id, det, det_idx)
            assigned_track_ids.add(track_id)
            outputs.append(self._output_track(track_id, det, det_idx, 1.0))

        unmatched_track_ids = {active_ids[idx] for idx in unmatched_tracks if idx < len(active_ids)}
        lifecycle_decisions = self._lifecycle_decisions(unmatched_track_ids - assigned_track_ids, score_matrix, active_ids, model)
        for track_id in sorted(unmatched_track_ids - assigned_track_ids):
            if track_id not in self.tracks:
                continue
            if lifecycle_decisions.get(track_id, False):
                self.tracks[track_id].lost_frames += 1
                if self.tracks[track_id].lost_frames > self.max_lost_frames:
                    del self.tracks[track_id]
            else:
                del self.tracks[track_id]

        outputs.sort(key=lambda item: int(item["id"]))
        return {"sequence_id": det.sequence_id, "frame_id": det.frame_id, "tracks": outputs}

    def _birth_decisions(
        self,
        det: DetectionFrame,
        unmatched_dets: list[int],
        score_matrix: np.ndarray,
        model: torch.nn.Module,
    ) -> dict[int, bool]:
        if not unmatched_dets:
            return {}
        rows: list[dict[str, Any]] = []
        empty = self._empty_history_tensors()
        best_scores = score_matrix.max(axis=0) if score_matrix.size else np.zeros((len(det.pred_boxes),), dtype=np.float32)
        for det_idx in unmatched_dets:
            prompt_text = self.formulator.build_birth_prompt(
                candidate_class_id=int(det.pred_labels[det_idx]),
                detector_score=float(det.pred_scores[det_idx]),
                best_assoc_score=float(best_scores[det_idx]) if det_idx < len(best_scores) else 0.0,
            )
            row = dict(empty)
            row.update(
                {
                    "candidate_box": det.pred_boxes[det_idx],
                    "candidate_score": np.float32(det.pred_scores[det_idx]),
                    "candidate_class_id": np.int64(det.pred_labels[det_idx]),
                    "prompt_text": prompt_text,
                    "box_token_mask": np.concatenate([empty["track_history_mask"], np.asarray([True], dtype=np.bool_)]),
                    "candidate_mask": np.bool_(True),
                    "task_name": "birth",
                }
            )
            rows.append(row)
        logits = _runtime_predict_action_logits(rows, self.device, model, _runtime_batch_size(self.cfg))
        decisions = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
        return {int(det_idx): bool(decisions[pos] == 1) for pos, det_idx in enumerate(unmatched_dets)}

    def _lifecycle_decisions(
        self,
        track_ids: set[int],
        score_matrix: np.ndarray,
        active_ids: list[int],
        model: torch.nn.Module,
    ) -> dict[int, bool]:
        if not track_ids:
            return {}
        rows: list[dict[str, Any]] = []
        ordered_track_ids = sorted(track_ids)
        for track_id in ordered_track_ids:
            hist = self._history_tensors(track_id)
            track_pos = active_ids.index(track_id) if track_id in active_ids else -1
            best_score = float(score_matrix[track_pos].max()) if track_pos >= 0 and score_matrix.shape[1] > 0 else 0.0
            prompt_text = self.formulator.build_lifecycle_prompt(
                track_id=int(track_id),
                lost_frames=int(self.tracks[track_id].lost_frames + 1),
                best_assoc_score=best_score,
                history_mask=hist["track_history_mask"],
                history_class_ids=hist["track_history_class_ids"],
            )
            row = dict(hist)
            row.update(
                {
                    "candidate_box": np.zeros((7,), dtype=np.float32),
                    "candidate_score": np.float32(0.0),
                    "candidate_class_id": np.int64(0),
                    "prompt_text": prompt_text,
                    "box_token_mask": np.concatenate([hist["track_history_mask"], np.asarray([False], dtype=np.bool_)]),
                    "candidate_mask": np.bool_(False),
                    "task_name": "lifecycle",
                }
            )
            rows.append(row)
        logits = _runtime_predict_action_logits(rows, self.device, model, _runtime_batch_size(self.cfg))
        decisions = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
        return {int(track_id): bool(decisions[pos] == 1) for pos, track_id in enumerate(ordered_track_ids)}

    def _empty_history_tensors(self) -> dict[str, np.ndarray]:
        return {
            "track_history_boxes": np.zeros((self.history_len, 7), dtype=np.float32),
            "track_history_scores": np.zeros((self.history_len,), dtype=np.float32),
            "track_history_class_ids": np.zeros((self.history_len,), dtype=np.int64),
            "track_history_mask": np.zeros((self.history_len,), dtype=np.bool_),
        }


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
    out = {
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
    if any("task_name" in row for row in rows):
        out["task_name"] = [str(row.get("task_name", "association")) for row in rows]
        out["candidate_mask"] = torch.as_tensor([bool(row.get("candidate_mask", True)) for row in rows], dtype=torch.bool, device=device)
    return out


def _runtime_batch_size(cfg: Config) -> int:
    return max(1, int(cfg.nova.get("runtime_batch_size", 32)))


def _runtime_row_chunks(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def _runtime_predict_match_prob(
    rows: list[dict[str, Any]],
    device: torch.device,
    model: torch.nn.Module,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for chunk in _runtime_row_chunks(rows, batch_size):
            batch = _runtime_collate(chunk, device)
            out = model(batch)
            chunks.append(out["match_prob"].detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)


def _runtime_predict_action_logits(
    rows: list[dict[str, Any]],
    device: torch.device,
    model: torch.nn.Module,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in _runtime_row_chunks(rows, batch_size):
            batch = _runtime_collate(chunk, device)
            out = model(batch)
            logits = out.get("action_logits", out.get("match_logits"))
            chunks.append(logits.detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 2), dtype=torch.float32)


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
    bev_range: list[float] | tuple[float, float, float, float] | None = None,
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
            det = filter_eval_detections(det, float(cfg.eval.get("score_thresh", 0.0)), bev_range=bev_range)
        outputs.append(tracker.update(det, model))
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(frame=f"{frame_count}/{len(ordered_infos)}")
        elif progress_interval > 0 and (frame_count == 1 or frame_count % progress_interval == 0 or frame_count == len(ordered_infos)):
            print(f"nova eval progress frames={frame_count}/{len(ordered_infos)}", flush=True)
    if progress_bar is not None:
        progress_bar.close()
    return outputs, info_path


def run_nova_lifecycle_tracking(
    *,
    cfg: Config,
    model: torch.nn.Module,
    device: torch.device,
    split: str = "val",
    max_frames: int = 0,
    progress_interval: int = 50,
    use_tqdm: bool = False,
    desc: str = "nova lifecycle eval",
    bev_range: list[float] | tuple[float, float, float, float] | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    return _run_nova_tracking_with_tracker(
        cfg=cfg,
        model=model,
        device=device,
        split=split,
        max_frames=max_frames,
        progress_interval=progress_interval,
        use_tqdm=use_tqdm,
        desc=desc,
        bev_range=bev_range,
        tracker_cls=NOVALifecycleOnlineTracker,
    )


def _run_nova_tracking_with_tracker(
    *,
    cfg: Config,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    max_frames: int,
    progress_interval: int,
    use_tqdm: bool,
    desc: str,
    bev_range: list[float] | tuple[float, float, float, float] | None,
    tracker_cls: type[NOVAOnlineTracker],
) -> tuple[list[dict[str, Any]], Path]:
    detections = load_detection_cache(cfg, split)
    info_path = Path(resolve_info_path(cfg, split))
    import pickle

    with info_path.open("rb") as f:
        infos = pickle.load(f)
    ordered_infos = sorted(infos, key=lambda x: (str(x.get("sequence_id", "")), int(x.get("frame_idx", 0))))
    if max_frames > 0:
        ordered_infos = ordered_infos[:max_frames]

    tracker = tracker_cls(cfg, device)
    outputs: list[dict[str, Any]] = []
    current_sequence = None
    model.eval()
    progress_bar = tqdm(total=len(ordered_infos), desc=desc, dynamic_ncols=True, leave=True) if use_tqdm and tqdm is not None else None
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
        else:
            det = filter_eval_detections(det, float(cfg.eval.get("score_thresh", 0.0)), bev_range=bev_range)
        outputs.append(tracker.update(det, model))
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(frame=f"{frame_count}/{len(ordered_infos)}")
        elif progress_interval > 0 and (frame_count == 1 or frame_count % progress_interval == 0 or frame_count == len(ordered_infos)):
            print(f"nova lifecycle eval progress frames={frame_count}/{len(ordered_infos)}", flush=True)
    if progress_bar is not None:
        progress_bar.close()
    return outputs, info_path


def filter_eval_detections(
    det: DetectionFrame,
    score_thresh: float,
    bev_range: list[float] | tuple[float, float, float, float] | None = None,
) -> DetectionFrame:
    if len(det.pred_scores) == 0:
        return det
    keep = np.ones((len(det.pred_scores),), dtype=bool)
    if score_thresh > 0.0:
        keep &= np.asarray(det.pred_scores >= float(score_thresh), dtype=bool)
    if bev_range is not None:
        keep &= _bev_mask(det.pred_boxes, bev_range)
    return DetectionFrame(
        sequence_id=det.sequence_id,
        frame_id=det.frame_id,
        frame_idx=det.frame_idx,
        pred_boxes=det.pred_boxes[keep],
        pred_scores=det.pred_scores[keep],
        pred_labels=det.pred_labels[keep],
    )


def _bev_mask(boxes: np.ndarray, bev_range: list[float] | tuple[float, float, float, float]) -> np.ndarray:
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
