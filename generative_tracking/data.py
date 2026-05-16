from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config, resolve_info_path
from .detections import (
    assign_detection_track_ids,
    detection_record_to_arrays,
    frame_key,
    load_detection_annotations,
)
from .prompts import build_tracking_prompt


@dataclass(frozen=True)
class ObjectFrame:
    boxes: np.ndarray
    class_ids: np.ndarray
    class_names: np.ndarray
    track_ids: np.ndarray
    scores: np.ndarray


def frame_to_objects(
    info: dict[str, Any],
    class_to_id: dict[str, int],
    detection_record: dict[str, Any] | None = None,
    score_thresh: float = 0.0,
    match_iou: float = 0.1,
) -> ObjectFrame:
    annos = info.get("annos") or {}
    gt_boxes = np.asarray(annos.get("gt_boxes_lidar", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes.reshape(-1, 7)
    gt_names = np.asarray(annos.get("name", np.asarray(["Unknown"] * len(gt_boxes)))).astype(str)
    gt_track_ids = np.asarray(annos.get("track_id", np.full((len(gt_boxes),), -1)), dtype=np.int64)

    if detection_record is not None:
        class_names = list(class_to_id.keys())
        det = detection_record_to_arrays(detection_record, class_names, score_thresh)
        boxes = det["boxes"]
        names = det["names"]
        scores = det["scores"]
        track_ids = assign_detection_track_ids(boxes, names, gt_boxes, gt_names, gt_track_ids, match_iou)
        class_ids = np.asarray([class_to_id.get(str(name), 0) for name in names], dtype=np.int64)
        return ObjectFrame(boxes=boxes, class_ids=class_ids, class_names=names, track_ids=track_ids, scores=scores)

    boxes = gt_boxes
    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, 7)
    names = gt_names
    track_ids = gt_track_ids
    scores = np.asarray(annos.get("score", np.ones((len(boxes),), dtype=np.float32)), dtype=np.float32)
    if len(scores) != len(boxes):
        scores = np.ones((len(boxes),), dtype=np.float32)
    scores = np.where(scores >= 0.0, scores, 1.0).astype(np.float32)
    class_ids = np.asarray([class_to_id.get(str(name), 0) for name in names], dtype=np.int64)
    return ObjectFrame(boxes=boxes, class_ids=class_ids, class_names=names, track_ids=track_ids, scores=scores)


def build_history_candidates(
    history_frames: list[ObjectFrame],
    max_history_tracks: int,
) -> dict[str, np.ndarray]:
    """Build pointer candidates from newest history frame to oldest frame."""

    seen: set[int] = set()
    boxes: list[np.ndarray] = []
    class_ids: list[int] = []
    track_ids: list[int] = []
    for frame in reversed(history_frames):
        for obj_idx, raw_track_id in enumerate(frame.track_ids.tolist()):
            track_id = int(raw_track_id)
            if track_id < 0 or track_id in seen:
                continue
            seen.add(track_id)
            boxes.append(frame.boxes[obj_idx])
            class_ids.append(int(frame.class_ids[obj_idx]))
            track_ids.append(track_id)
            if len(track_ids) >= max_history_tracks:
                return _candidate_arrays(boxes, class_ids, track_ids)
    return _candidate_arrays(boxes, class_ids, track_ids)


def _candidate_arrays(boxes: list[np.ndarray], class_ids: list[int], track_ids: list[int]) -> dict[str, np.ndarray]:
    return {
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
        "class_ids": np.asarray(class_ids, dtype=np.int64),
        "track_ids": np.asarray(track_ids, dtype=np.int64),
    }


def build_pointer_labels(
    current_track_ids: np.ndarray,
    candidate_track_ids: np.ndarray,
    max_history_tracks: int,
    ignore_index: int = -1,
) -> np.ndarray:
    candidate_to_index = {int(track_id): idx for idx, track_id in enumerate(candidate_track_ids.tolist())}
    labels = np.full((len(current_track_ids),), ignore_index, dtype=np.int64)
    new_index = int(max_history_tracks)
    for obj_idx, raw_track_id in enumerate(current_track_ids.tolist()):
        track_id = int(raw_track_id)
        if track_id < 0:
            continue
        labels[obj_idx] = candidate_to_index.get(track_id, new_index)
    return labels


class SequenceWindowDataset(Dataset):
    def __init__(self, cfg: Config, split: str | None = None):
        self.cfg = cfg
        self.split = split or cfg.dataset.split
        self.info_path = Path(resolve_info_path(cfg, self.split))
        self.k = int(cfg.dataset.K)
        self.stride = int(cfg.dataset.stride)
        self.max_history_tracks = int(cfg.dataset.max_history_tracks)
        self.max_current_objects = int(cfg.dataset.max_current_objects)
        self.ignore_index = int(cfg.loss.ignore_index)
        self.class_to_id = {name: idx for idx, name in enumerate(cfg.dataset.class_names)}
        self.object_source = str(cfg.dataset.get("object_source", "gt"))
        self.detection_score_thresh = float(cfg.dataset.get("detection_score_thresh", 0.0))
        self.detection_match_iou = float(cfg.dataset.get("detection_match_iou", 0.1))
        self.detections = load_detection_annotations(cfg.dataset.detection_paths.get(self.split, "")) if self.object_source == "detections" else {}

        with self.info_path.open("rb") as f:
            infos = pickle.load(f)
        self.infos: list[dict[str, Any]] = sorted(
            infos,
            key=lambda x: (str(x.get("sequence_id", "")), int(x.get("frame_idx", 0))),
        )
        by_sequence: dict[str, list[int]] = defaultdict(list)
        for idx, info in enumerate(self.infos):
            by_sequence[str(info.get("sequence_id", ""))].append(idx)
        self.sequences = {seq: sorted(indices, key=lambda i: int(self.infos[i].get("frame_idx", 0))) for seq, indices in by_sequence.items()}
        self.index: list[tuple[str, int]] = []
        for seq, indices in self.sequences.items():
            for pos in range(len(indices)):
                self.index.append((seq, pos))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        seq, pos = self.index[item]
        seq_indices = self.sequences[seq]
        target_info = self.infos[seq_indices[pos]]
        frame_objects: list[ObjectFrame | None] = []
        frame_detector_tokens: list[np.ndarray] = []
        valid_mask: list[bool] = []
        for slot in range(self.k):
            offset = (self.k - 1 - slot) * self.stride
            hist_pos = pos - offset
            if hist_pos < 0:
                frame_objects.append(None)
                frame_detector_tokens.append(np.zeros((0, int(self.cfg.model.get("detector_token_dim", self.cfg.model.visual_dim))), dtype=np.float32))
                valid_mask.append(False)
            else:
                info = self.infos[seq_indices[hist_pos]]
                frame_objects.append(self._objects_for_info(info))
                frame_detector_tokens.append(self._detector_tokens_for_info(info))
                valid_mask.append(True)

        valid_history = [frame for frame in frame_objects[:-1] if frame is not None]
        candidates = build_history_candidates(valid_history, self.max_history_tracks)
        current = frame_objects[-1]
        if current is None:
            current = ObjectFrame(
                boxes=np.zeros((0, 7), dtype=np.float32),
                class_ids=np.zeros((0,), dtype=np.int64),
                class_names=np.asarray([], dtype=object),
                track_ids=np.zeros((0,), dtype=np.int64),
                scores=np.zeros((0,), dtype=np.float32),
            )
        current = _limit_current(current, self.max_current_objects)
        pointer_labels = build_pointer_labels(
            current.track_ids,
            candidates["track_ids"],
            self.max_history_tracks,
            self.ignore_index,
        )

        sample = {
            "sequence_id": str(target_info.get("sequence_id", "")),
            "frame_id": str(target_info.get("frame_id", target_info.get("frame_idx", ""))),
            "frame_idx": int(target_info.get("frame_idx", 0)),
            "frame_valid_mask": np.asarray(valid_mask, dtype=np.bool_),
            "window_objects": [_empty_frame() if frame is None else frame for frame in frame_objects],
            "window_detector_tokens": frame_detector_tokens,
            "current_boxes": current.boxes,
            "current_class_ids": current.class_ids,
            "current_class_names": current.class_names.astype(str),
            "current_track_ids": current.track_ids,
            "current_scores": current.scores,
            "history_boxes": candidates["boxes"],
            "history_class_ids": candidates["class_ids"],
            "history_track_ids": candidates["track_ids"],
            "pointer_labels": pointer_labels,
        }
        if bool(self.cfg.prompt.get("enabled", False)):
            sample["prompt_text"] = build_tracking_prompt(sample, str(self.cfg.prompt.template))
        return sample

    def _objects_for_info(self, info: dict[str, Any]) -> ObjectFrame:
        if self.object_source != "detections":
            return frame_to_objects(info, self.class_to_id)
        seq = str(info.get("sequence_id", ""))
        frame_id = str(info.get("frame_id", info.get("frame_idx", "")))
        record = self.detections.get(frame_key(seq, frame_id))
        return frame_to_objects(
            info,
            self.class_to_id,
            detection_record=record,
            score_thresh=self.detection_score_thresh,
            match_iou=self.detection_match_iou,
        )

    def _detector_tokens_for_info(self, info: dict[str, Any]) -> np.ndarray:
        token_dim = int(self.cfg.model.get("detector_token_dim", self.cfg.model.visual_dim))
        if self.object_source != "detections":
            return np.zeros((0, token_dim), dtype=np.float32)
        seq = str(info.get("sequence_id", ""))
        frame_id = str(info.get("frame_id", info.get("frame_idx", "")))
        record = self.detections.get(frame_key(seq, frame_id)) or {}
        tokens = np.asarray(record.get("detector_tokens", np.zeros((0, token_dim))), dtype=np.float32)
        if tokens.ndim == 1:
            tokens = tokens.reshape(1, -1)
        if tokens.shape[-1] != token_dim:
            raise ValueError(f"detector_tokens dim {tokens.shape[-1]} does not match model.detector_token_dim={token_dim}")
        return tokens[: int(self.cfg.model.get("max_detector_tokens", 256))]


def _empty_frame() -> ObjectFrame:
    return ObjectFrame(
        boxes=np.zeros((0, 7), dtype=np.float32),
        class_ids=np.zeros((0,), dtype=np.int64),
        class_names=np.asarray([], dtype=object),
        track_ids=np.zeros((0,), dtype=np.int64),
        scores=np.zeros((0,), dtype=np.float32),
    )


def _limit_current(frame: ObjectFrame, max_current_objects: int) -> ObjectFrame:
    keep = slice(0, max_current_objects)
    return ObjectFrame(
        boxes=frame.boxes[keep],
        class_ids=frame.class_ids[keep],
        class_names=frame.class_names[keep],
        track_ids=frame.track_ids[keep],
        scores=frame.scores[keep],
    )


def tracklm_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    bsz = len(batch)
    k = len(batch[0]["window_objects"])
    max_window_objects = max(1, max((len(frame.boxes) for sample in batch for frame in sample["window_objects"]), default=0))
    max_current = max((len(sample["current_boxes"]) for sample in batch), default=0)
    max_history = max((len(sample["history_boxes"]) for sample in batch), default=0)
    token_dim = 0
    max_detector_tokens = 0
    for sample in batch:
        for tokens in sample.get("window_detector_tokens", []):
            if len(tokens):
                token_dim = int(tokens.shape[-1])
                max_detector_tokens = max(max_detector_tokens, int(tokens.shape[0]))
    configured_history = max((int(sample["pointer_labels"].max()) for sample in batch if len(sample["pointer_labels"]) > 0), default=max_history)
    if configured_history >= max_history:
        max_history = configured_history

    window_boxes = torch.zeros((bsz, k, max_window_objects, 7), dtype=torch.float32)
    window_class_ids = torch.zeros((bsz, k, max_window_objects), dtype=torch.long)
    window_track_ids = torch.full((bsz, k, max_window_objects), -1, dtype=torch.long)
    window_valid = torch.zeros((bsz, k, max_window_objects), dtype=torch.bool)
    current_boxes = torch.zeros((bsz, max_current, 7), dtype=torch.float32)
    current_class_ids = torch.zeros((bsz, max_current), dtype=torch.long)
    current_track_ids = torch.full((bsz, max_current), -1, dtype=torch.long)
    current_scores = torch.ones((bsz, max_current), dtype=torch.float32)
    current_valid = torch.zeros((bsz, max_current), dtype=torch.bool)
    history_boxes = torch.zeros((bsz, max_history, 7), dtype=torch.float32)
    history_class_ids = torch.zeros((bsz, max_history), dtype=torch.long)
    history_track_ids = torch.full((bsz, max_history), -1, dtype=torch.long)
    history_valid = torch.zeros((bsz, max_history), dtype=torch.bool)
    pointer_labels = torch.full((bsz, max_current), -1, dtype=torch.long)
    detector_frame_tokens = None
    detector_frame_valid = None
    if token_dim > 0:
        detector_frame_tokens = torch.zeros((bsz, k * max_detector_tokens, token_dim), dtype=torch.float32)
        detector_frame_valid = torch.zeros((bsz, k * max_detector_tokens), dtype=torch.bool)

    class_names: list[list[str]] = []
    prompt_texts: list[str] = []
    for bidx, sample in enumerate(batch):
        for fidx, frame in enumerate(sample["window_objects"]):
            n = len(frame.boxes)
            if n:
                window_boxes[bidx, fidx, :n] = torch.as_tensor(frame.boxes, dtype=torch.float32)
                window_class_ids[bidx, fidx, :n] = torch.as_tensor(frame.class_ids, dtype=torch.long)
                window_track_ids[bidx, fidx, :n] = torch.as_tensor(frame.track_ids, dtype=torch.long)
                window_valid[bidx, fidx, :n] = True
        if detector_frame_tokens is not None:
            for fidx, tokens in enumerate(sample.get("window_detector_tokens", [])):
                n_tok = min(len(tokens), max_detector_tokens)
                if n_tok:
                    start = fidx * max_detector_tokens
                    detector_frame_tokens[bidx, start:start + n_tok] = torch.as_tensor(tokens[:n_tok], dtype=torch.float32)
                    detector_frame_valid[bidx, start:start + n_tok] = True
        n_current = len(sample["current_boxes"])
        if n_current:
            current_boxes[bidx, :n_current] = torch.as_tensor(sample["current_boxes"], dtype=torch.float32)
            current_class_ids[bidx, :n_current] = torch.as_tensor(sample["current_class_ids"], dtype=torch.long)
            current_track_ids[bidx, :n_current] = torch.as_tensor(sample["current_track_ids"], dtype=torch.long)
            current_scores[bidx, :n_current] = torch.as_tensor(sample["current_scores"], dtype=torch.float32).clamp_min(0.0)
            current_valid[bidx, :n_current] = True
            pointer_labels[bidx, :n_current] = torch.as_tensor(sample["pointer_labels"], dtype=torch.long)
        n_history = len(sample["history_boxes"])
        if n_history:
            history_boxes[bidx, :n_history] = torch.as_tensor(sample["history_boxes"], dtype=torch.float32)
            history_class_ids[bidx, :n_history] = torch.as_tensor(sample["history_class_ids"], dtype=torch.long)
            history_track_ids[bidx, :n_history] = torch.as_tensor(sample["history_track_ids"], dtype=torch.long)
            history_valid[bidx, :n_history] = True
        class_names.append([str(x) for x in sample["current_class_names"].tolist()])
        prompt_texts.append(str(sample.get("prompt_text", "")))

    collated = {
        "sequence_id": [sample["sequence_id"] for sample in batch],
        "frame_id": [sample["frame_id"] for sample in batch],
        "frame_idx": torch.tensor([sample["frame_idx"] for sample in batch], dtype=torch.long),
        "frame_valid_mask": torch.as_tensor(np.stack([sample["frame_valid_mask"] for sample in batch]), dtype=torch.bool),
        "window_boxes": window_boxes,
        "window_class_ids": window_class_ids,
        "window_track_ids": window_track_ids,
        "window_valid_mask": window_valid,
        "current_boxes": current_boxes,
        "current_class_ids": current_class_ids,
        "current_class_names": class_names,
        "current_track_ids": current_track_ids,
        "current_scores": current_scores,
        "current_valid_mask": current_valid,
        "history_boxes": history_boxes,
        "history_class_ids": history_class_ids,
        "history_track_ids": history_track_ids,
        "history_valid_mask": history_valid,
        "pointer_labels": pointer_labels,
    }
    if any(prompt_texts):
        collated["prompt_texts"] = prompt_texts
    if detector_frame_tokens is not None and detector_frame_valid is not None:
        collated["detector_frame_tokens"] = detector_frame_tokens
        collated["detector_frame_valid_mask"] = detector_frame_valid
    return collated
