from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class TrackState:
    track_id: int
    box3d: list[float]
    class_name: str
    score: float
    lost_frames: int = 0
    state: str = "active"


class TrackManager:
    def __init__(self, max_lost_frames: int = 2, start_id: int = 0):
        self.max_lost_frames = int(max_lost_frames)
        self.next_id = int(start_id)
        self.tracks: dict[int, TrackState] = {}

    def reset(self) -> None:
        self.next_id = 0
        self.tracks.clear()

    def update(
        self,
        sequence_id: str,
        frame_id: str,
        boxes: torch.Tensor | np.ndarray,
        class_names: list[str],
        scores: torch.Tensor | np.ndarray,
        pointer_preds: torch.Tensor | np.ndarray,
        history_track_ids: torch.Tensor | np.ndarray,
        history_valid_mask: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, Any]:
        boxes_np = _to_numpy(boxes).reshape(-1, 7)
        scores_np = _to_numpy(scores).reshape(-1)
        preds_np = _to_numpy(pointer_preds).astype(np.int64).reshape(-1)
        hist_ids = _to_numpy(history_track_ids).astype(np.int64).reshape(-1)
        if history_valid_mask is None:
            hist_valid = hist_ids >= 0
        else:
            hist_valid = _to_numpy(history_valid_mask).astype(bool).reshape(-1)

        previous_tracks = set(self.tracks.keys())
        assigned: set[int] = set()
        outputs: list[dict[str, Any]] = []
        for idx, box in enumerate(boxes_np):
            pred = int(preds_np[idx]) if idx < len(preds_np) else len(hist_ids)
            if 0 <= pred < len(hist_ids) and hist_valid[pred] and int(hist_ids[pred]) >= 0:
                track_id = int(hist_ids[pred])
                self.next_id = max(self.next_id, track_id + 1)
            else:
                track_id = self._new_id()
            class_name = class_names[idx] if idx < len(class_names) else "Unknown"
            score = float(scores_np[idx]) if idx < len(scores_np) else 1.0
            state = TrackState(
                track_id=track_id,
                box3d=[float(x) for x in box.tolist()],
                class_name=str(class_name),
                score=score,
                lost_frames=0,
                state="active",
            )
            self.tracks[track_id] = state
            assigned.add(track_id)
            outputs.append(_state_to_json(state))

        for track_id in sorted(previous_tracks - assigned):
            if track_id not in self.tracks:
                continue
            state = self.tracks[track_id]
            state.lost_frames += 1
            state.state = "lost"
            if state.lost_frames > self.max_lost_frames:
                del self.tracks[track_id]

        return {
            "sequence_id": sequence_id,
            "frame_id": frame_id,
            "tracks": outputs,
        }

    def _new_id(self) -> int:
        while self.next_id in self.tracks:
            self.next_id += 1
        track_id = self.next_id
        self.next_id += 1
        return track_id


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _state_to_json(state: TrackState) -> dict[str, Any]:
    return {
        "id": state.track_id,
        "class": state.class_name,
        "box3d": state.box3d,
        "score": state.score,
        "state": state.state,
    }
