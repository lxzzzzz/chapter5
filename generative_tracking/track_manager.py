from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TrackState:
    track_id: int
    embedding: torch.Tensor
    box3d: list[float]
    class_name: str
    score: float
    lost_frames: int = 0
    state: str = "active"


class TrackEmbeddingManager:
    """Online ID assignment from generated track embeddings."""

    def __init__(self, max_lost_frames: int = 2, match_threshold: float = 0.5, start_id: int = 0):
        self.max_lost_frames = int(max_lost_frames)
        self.match_threshold = float(match_threshold)
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
        class_ids: torch.Tensor | np.ndarray,
        class_names: list[str],
        scores: torch.Tensor | np.ndarray,
        embeddings: torch.Tensor | np.ndarray,
        valid_mask: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, Any]:
        boxes_np = _to_numpy(boxes).reshape(-1, 7)
        class_ids_np = _to_numpy(class_ids).astype(np.int64).reshape(-1)
        scores_np = _to_numpy(scores).reshape(-1)
        embeds = _to_tensor(embeddings).float()
        if valid_mask is None:
            valid = np.ones((len(boxes_np),), dtype=bool)
        else:
            valid = _to_numpy(valid_mask).astype(bool).reshape(-1)

        previous_ids = list(self.tracks.keys())
        active_ids = [tid for tid, state in self.tracks.items() if state.state == "active"]
        assigned_tracks: set[int] = set()
        outputs: list[dict[str, Any]] = []
        for obj_idx in range(len(boxes_np)):
            if not valid[obj_idx]:
                continue
            embedding = F.normalize(embeds[obj_idx], dim=0).detach().cpu()
            track_id = self._match_track(embedding, active_ids, assigned_tracks)
            if track_id is None:
                track_id = self._new_id()
            assigned_tracks.add(track_id)
            class_id = int(class_ids_np[obj_idx]) if obj_idx < len(class_ids_np) else 0
            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else "Unknown"
            score = float(scores_np[obj_idx]) if obj_idx < len(scores_np) else 1.0
            state = TrackState(
                track_id=track_id,
                embedding=embedding,
                box3d=[float(x) for x in boxes_np[obj_idx].tolist()],
                class_name=str(class_name),
                score=score,
                lost_frames=0,
                state="active",
            )
            self.tracks[track_id] = state
            outputs.append(_state_to_json(state))

        for track_id in sorted(set(previous_ids) - assigned_tracks):
            if track_id not in self.tracks:
                continue
            state = self.tracks[track_id]
            state.lost_frames += 1
            state.state = "lost"
            if state.lost_frames > self.max_lost_frames:
                del self.tracks[track_id]

        return {"sequence_id": sequence_id, "frame_id": frame_id, "tracks": outputs}

    def _match_track(self, embedding: torch.Tensor, active_ids: list[int], assigned_tracks: set[int]) -> int | None:
        best_id = None
        best_score = self.match_threshold
        for track_id in active_ids:
            if track_id in assigned_tracks or track_id not in self.tracks:
                continue
            score = float(torch.dot(embedding, self.tracks[track_id].embedding))
            if score > best_score:
                best_score = score
                best_id = track_id
        return best_id

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


def _to_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _state_to_json(state: TrackState) -> dict[str, Any]:
    return {
        "id": state.track_id,
        "class": state.class_name,
        "box3d": state.box3d,
        "score": state.score,
        "state": state.state,
    }
