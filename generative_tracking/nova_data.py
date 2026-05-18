from __future__ import annotations

import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config, resolve_info_path
from .data import ObjectFrame, frame_to_objects
from .geometry import boxes_iou_3d_axis_aligned

BOX_TOKEN_DEFAULT = "<box>"


@dataclass(frozen=True)
class DetectionFrame:
    sequence_id: str
    frame_id: str
    frame_idx: int
    pred_boxes: np.ndarray
    pred_scores: np.ndarray
    pred_labels: np.ndarray


def validate_detection_frame(frame: dict[str, Any]) -> DetectionFrame:
    required = {"sequence_id", "frame_id", "frame_idx", "pred_boxes", "pred_scores", "pred_labels"}
    missing = sorted(required - set(frame))
    if missing:
        raise KeyError(f"Detection frame is missing keys: {missing}")
    boxes = np.asarray(frame["pred_boxes"], dtype=np.float32).reshape(-1, 7)
    scores = np.asarray(frame["pred_scores"], dtype=np.float32).reshape(-1)
    labels = np.asarray(frame["pred_labels"], dtype=np.int64).reshape(-1)
    if len(scores) != len(boxes) or len(labels) != len(boxes):
        raise ValueError(
            "Detection frame has inconsistent lengths: "
            f"boxes={len(boxes)} scores={len(scores)} labels={len(labels)}"
        )
    return DetectionFrame(
        sequence_id=str(frame["sequence_id"]),
        frame_id=str(frame["frame_id"]),
        frame_idx=int(frame["frame_idx"]),
        pred_boxes=boxes,
        pred_scores=scores,
        pred_labels=labels,
    )


def detection_frame_to_dict(frame: DetectionFrame | dict[str, Any]) -> dict[str, Any]:
    if isinstance(frame, dict):
        frame = validate_detection_frame(frame)
    return {
        "sequence_id": frame.sequence_id,
        "frame_id": frame.frame_id,
        "frame_idx": int(frame.frame_idx),
        "pred_boxes": np.asarray(frame.pred_boxes, dtype=np.float32).reshape(-1, 7),
        "pred_scores": np.asarray(frame.pred_scores, dtype=np.float32).reshape(-1),
        "pred_labels": np.asarray(frame.pred_labels, dtype=np.int64).reshape(-1),
    }


def filter_detection_frame(
    frame: dict[str, Any],
    *,
    class_id: int = 0,
    score_thresh: float = 0.05,
    max_dets: int = 100,
) -> dict[str, Any]:
    det = validate_detection_frame(frame)
    keep = (det.pred_labels == int(class_id)) & (det.pred_scores >= float(score_thresh))
    indices = np.where(keep)[0]
    if len(indices):
        order = indices[np.argsort(det.pred_scores[indices])[::-1]]
        order = order[: int(max_dets)]
    else:
        order = np.zeros((0,), dtype=np.int64)
    return {
        "sequence_id": det.sequence_id,
        "frame_id": det.frame_id,
        "frame_idx": det.frame_idx,
        "pred_boxes": det.pred_boxes[order].astype(np.float32),
        "pred_scores": det.pred_scores[order].astype(np.float32),
        "pred_labels": det.pred_labels[order].astype(np.int64),
    }


def resolve_detection_cache_path(cfg: Config, split: str) -> Path:
    paths = cfg.detection_cache.get("paths", {})
    configured = paths.get(split, "") if isinstance(paths, dict) else ""
    if configured:
        return Path(configured)
    root = Path(str(cfg.detection_cache.root))
    manifest = root / f"detections_{split}.pkl"
    if manifest.exists():
        return manifest
    return root / split


def load_detection_cache(cfg: Config, split: str) -> dict[tuple[str, str], DetectionFrame]:
    path = resolve_detection_cache_path(cfg, split)
    if not path.exists():
        raise FileNotFoundError(f"Detection cache not found for split={split!r}: {path}")
    frames: list[dict[str, Any]] = []
    if path.is_dir():
        for item in sorted(path.rglob("*.pkl")):
            with item.open("rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict) and "frames" in loaded:
                frames.extend(loaded["frames"])
            elif isinstance(loaded, list):
                frames.extend(loaded)
            else:
                frames.append(loaded)
    else:
        with path.open("rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict) and "frames" in loaded:
            frames = list(loaded["frames"])
        elif isinstance(loaded, dict) and "detections" in loaded:
            frames = list(loaded["detections"])
        elif isinstance(loaded, list):
            frames = loaded
        elif isinstance(loaded, dict) and all(isinstance(key, tuple) for key in loaded):
            frames = list(loaded.values())
        else:
            frames = [loaded]

    out: dict[tuple[str, str], DetectionFrame] = {}
    for raw in frames:
        det = validate_detection_frame(raw)
        out[(det.sequence_id, det.frame_id)] = det
    return out


@dataclass(frozen=True)
class NOVAPairIndex:
    seq: str
    pos: int
    track_id: int
    det_idx: int
    label: int
    target_iou: float
    quality_valid: bool


class NOVAAssociationDataset(Dataset):
    def __init__(self, cfg: Config, split: str | None = None, detection_cache: dict[tuple[str, str], DetectionFrame] | None = None):
        self.cfg = cfg
        self.split = split or cfg.dataset.split
        self.info_path = Path(resolve_info_path(cfg, self.split))
        self.class_to_id = {name: idx for idx, name in enumerate(cfg.dataset.class_names)}
        self.history_len = int(cfg.nova.get("history_len", 5))
        self.history_stride = int(cfg.nova.get("history_stride", 1))
        self.det_gt_iou_threshold = float(cfg.nova.get("det_gt_iou_threshold", 0.5))
        self.negative_positive_ratio = float(cfg.nova.get("negative_positive_ratio", 0.0))
        self.formulator = NOVAFormulator(cfg)

        with self.info_path.open("rb") as f:
            infos = pickle.load(f)
        self.infos: list[dict[str, Any]] = sorted(
            infos,
            key=lambda x: (str(x.get("sequence_id", "")), int(x.get("frame_idx", 0))),
        )
        self.detections = detection_cache if detection_cache is not None else load_detection_cache(cfg, self.split)
        self.sequences: dict[str, list[int]] = defaultdict(list)
        for idx, info in enumerate(self.infos):
            self.sequences[str(info.get("sequence_id", ""))].append(idx)
        self.sequences = {
            seq: sorted(indices, key=lambda i: int(self.infos[i].get("frame_idx", 0)))
            for seq, indices in self.sequences.items()
        }
        self._objects_by_global_idx = {idx: frame_to_objects(info, self.class_to_id) for idx, info in enumerate(self.infos)}
        self.index: list[NOVAPairIndex] = self._build_pair_index()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        pair = self.index[item]
        seq_indices = self.sequences[pair.seq]
        global_idx = seq_indices[pair.pos]
        info = self.infos[global_idx]
        det = self._detections_for_info(info)
        hist = self._track_history(pair.seq, pair.pos, pair.track_id)
        candidate_box = det.pred_boxes[pair.det_idx].astype(np.float32)
        candidate_score = np.float32(det.pred_scores[pair.det_idx])
        candidate_class_id = np.int64(det.pred_labels[pair.det_idx])
        prompt_text = self.formulator.build_prompt(
            track_id=int(pair.track_id),
            history_mask=hist["mask"],
            history_class_ids=hist["class_ids"],
            candidate_class_id=int(candidate_class_id),
        )
        return {
            "sequence_id": str(info.get("sequence_id", "")),
            "frame_id": str(info.get("frame_id", info.get("frame_idx", ""))),
            "frame_idx": int(info.get("frame_idx", 0)),
            "track_id": int(pair.track_id),
            "det_idx": int(pair.det_idx),
            "track_history_boxes": hist["boxes"],
            "track_history_scores": hist["scores"],
            "track_history_class_ids": hist["class_ids"],
            "track_history_mask": hist["mask"],
            "candidate_box": candidate_box,
            "candidate_score": candidate_score,
            "candidate_class_id": candidate_class_id,
            "prompt_text": prompt_text,
            "box_token_mask": np.concatenate([hist["mask"], np.asarray([True], dtype=np.bool_)]),
            "match_label": np.int64(pair.label),
            "target_iou": np.float32(pair.target_iou),
            "quality_valid": np.bool_(pair.quality_valid),
        }

    def _build_pair_index(self) -> list[NOVAPairIndex]:
        pairs: list[NOVAPairIndex] = []
        rng = random.Random(int(self.cfg.seed) + (0 if self.split == "train" else 100000))
        for seq, seq_indices in self.sequences.items():
            for pos, global_idx in enumerate(seq_indices):
                info = self.infos[global_idx]
                det = self._detections_for_info(info)
                if len(det.pred_boxes) == 0:
                    continue
                active_track_ids = sorted(self._active_track_ids(seq, pos))
                if not active_track_ids:
                    continue
                gt = self._objects_by_global_idx[global_idx]
                det_track_ids, det_ious = assign_detections_to_gt_tracks(
                    gt,
                    det,
                    iou_threshold=self.det_gt_iou_threshold,
                )
                positives: list[NOVAPairIndex] = []
                negatives: list[NOVAPairIndex] = []
                for track_id in active_track_ids:
                    for det_idx in range(len(det.pred_boxes)):
                        det_track_id = int(det_track_ids[det_idx])
                        label = int(det_track_id >= 0 and det_track_id == int(track_id))
                        pair = NOVAPairIndex(
                            seq=seq,
                            pos=pos,
                            track_id=int(track_id),
                            det_idx=int(det_idx),
                            label=label,
                            target_iou=float(det_ious[det_idx]),
                            quality_valid=bool(det_ious[det_idx] >= self.det_gt_iou_threshold),
                        )
                        if label:
                            positives.append(pair)
                        else:
                            negatives.append(pair)
                if self.negative_positive_ratio > 0.0 and positives:
                    keep_neg = min(len(negatives), int(np.ceil(len(positives) * self.negative_positive_ratio)))
                    negatives = rng.sample(negatives, keep_neg) if keep_neg < len(negatives) else negatives
                pairs.extend(positives)
                pairs.extend(negatives)
        return pairs

    def _detections_for_info(self, info: dict[str, Any]) -> DetectionFrame:
        key = (str(info.get("sequence_id", "")), str(info.get("frame_id", info.get("frame_idx", ""))))
        if key not in self.detections:
            raise KeyError(f"Detection cache is missing frame {key}")
        return self.detections[key]

    def _active_track_ids(self, seq: str, pos: int) -> set[int]:
        active: set[int] = set()
        seq_indices = self.sequences[seq]
        for slot in range(self.history_len):
            hist_pos = pos - (slot + 1) * self.history_stride
            if hist_pos < 0:
                continue
            objects = self._objects_by_global_idx[seq_indices[hist_pos]]
            active.update(int(track_id) for track_id in objects.track_ids.tolist() if int(track_id) >= 0)
        return active

    def _track_history(self, seq: str, pos: int, track_id: int) -> dict[str, np.ndarray]:
        seq_indices = self.sequences[seq]
        boxes = np.zeros((self.history_len, 7), dtype=np.float32)
        scores = np.zeros((self.history_len,), dtype=np.float32)
        class_ids = np.zeros((self.history_len,), dtype=np.int64)
        mask = np.zeros((self.history_len,), dtype=np.bool_)
        for slot in range(self.history_len):
            hist_pos = pos - (self.history_len - slot) * self.history_stride
            if hist_pos < 0:
                continue
            objects = self._objects_by_global_idx[seq_indices[hist_pos]]
            matches = np.where(objects.track_ids == int(track_id))[0]
            if len(matches) == 0:
                continue
            obj_idx = int(matches[0])
            boxes[slot] = objects.boxes[obj_idx]
            scores[slot] = objects.scores[obj_idx]
            class_ids[slot] = objects.class_ids[obj_idx]
            mask[slot] = True
        return {"boxes": boxes, "scores": scores, "class_ids": class_ids, "mask": mask}


class NOVAFormulator:
    """Text prompt builder for NOVA-style pair association.

    The prompt keeps 3D geometry behind repeated ``<box>`` placeholders. The
    model replaces those token embeddings with Geometry Encoder outputs.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.box_token = str(cfg.nova.get("box_token", BOX_TOKEN_DEFAULT))
        self.class_names = [str(name) for name in cfg.dataset.class_names]

    def class_name(self, class_id: int) -> str:
        return self.class_names[class_id] if 0 <= int(class_id) < len(self.class_names) else "Unknown"

    def build_prompt(
        self,
        *,
        track_id: int,
        history_mask: np.ndarray,
        history_class_ids: np.ndarray,
        candidate_class_id: int,
    ) -> str:
        lines = ["Task: 3D Association", "History:"]
        history_mask = np.asarray(history_mask, dtype=np.bool_).reshape(-1)
        history_class_ids = np.asarray(history_class_ids, dtype=np.int64).reshape(-1)
        total = len(history_mask)
        for slot in range(total):
            rel = slot - total
            if bool(history_mask[slot]):
                cls = self.class_name(int(history_class_ids[slot]))
                lines.append(f"Frame {rel}: ID: {int(track_id)}, Class: {cls}, Box: {self.box_token}")
            else:
                lines.append(f"Frame {rel}: Observation: Missing")
        cand_cls = self.class_name(int(candidate_class_id))
        lines.extend(
            [
                "Candidate:",
                f"Class: {cand_cls}, Box: {self.box_token}",
                "Question: Is this the same object?",
                "Answer:",
            ]
        )
        return "\n".join(lines)


def assign_detections_to_gt_tracks(
    gt: ObjectFrame,
    det: DetectionFrame,
    *,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    det_track_ids = np.full((len(det.pred_boxes),), -1, dtype=np.int64)
    det_ious = np.zeros((len(det.pred_boxes),), dtype=np.float32)
    if len(gt.boxes) == 0 or len(det.pred_boxes) == 0:
        return det_track_ids, det_ious
    iou = boxes_iou_3d_axis_aligned(gt.boxes, det.pred_boxes)
    matches = _match_by_iou(iou, float(iou_threshold))
    for gt_idx, det_idx, score in matches:
        det_track_ids[det_idx] = int(gt.track_ids[gt_idx])
        det_ious[det_idx] = float(score)
    return det_track_ids, det_ious


def _match_by_iou(iou: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    if iou.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        gt_idx, det_idx = linear_sum_assignment(-iou)
        return [
            (int(g), int(d), float(iou[g, d]))
            for g, d in zip(gt_idx.tolist(), det_idx.tolist())
            if float(iou[g, d]) >= threshold
        ]
    except ImportError:
        flat_order = np.argsort(iou.reshape(-1))[::-1]
        out: list[tuple[int, int, float]] = []
        used_gt: set[int] = set()
        used_det: set[int] = set()
        num_det = iou.shape[1]
        for flat_idx in flat_order.tolist():
            score = float(iou.reshape(-1)[flat_idx])
            if score < threshold:
                break
            gt_idx = flat_idx // num_det
            det_idx = flat_idx % num_det
            if gt_idx in used_gt or det_idx in used_det:
                continue
            used_gt.add(gt_idx)
            used_det.add(det_idx)
            out.append((gt_idx, det_idx, score))
        return out


def nova_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sequence_id": [sample["sequence_id"] for sample in batch],
        "frame_id": [sample["frame_id"] for sample in batch],
        "frame_idx": torch.tensor([sample["frame_idx"] for sample in batch], dtype=torch.long),
        "track_id": torch.tensor([sample["track_id"] for sample in batch], dtype=torch.long),
        "det_idx": torch.tensor([sample["det_idx"] for sample in batch], dtype=torch.long),
        "track_history_boxes": torch.as_tensor(np.stack([sample["track_history_boxes"] for sample in batch]), dtype=torch.float32),
        "track_history_scores": torch.as_tensor(np.stack([sample["track_history_scores"] for sample in batch]), dtype=torch.float32),
        "track_history_class_ids": torch.as_tensor(np.stack([sample["track_history_class_ids"] for sample in batch]), dtype=torch.long),
        "track_history_mask": torch.as_tensor(np.stack([sample["track_history_mask"] for sample in batch]), dtype=torch.bool),
        "candidate_box": torch.as_tensor(np.stack([sample["candidate_box"] for sample in batch]), dtype=torch.float32),
        "candidate_score": torch.as_tensor([sample["candidate_score"] for sample in batch], dtype=torch.float32),
        "candidate_class_id": torch.as_tensor([sample["candidate_class_id"] for sample in batch], dtype=torch.long),
        "prompt_texts": [str(sample.get("prompt_text", "")) for sample in batch],
        "box_token_mask": torch.as_tensor(
            np.stack(
                [
                    sample.get(
                        "box_token_mask",
                        np.concatenate([sample["track_history_mask"], np.asarray([True], dtype=np.bool_)]),
                    )
                    for sample in batch
                ]
            ),
            dtype=torch.bool,
        ),
        "match_label": torch.as_tensor([sample["match_label"] for sample in batch], dtype=torch.long),
        "target_iou": torch.as_tensor([sample["target_iou"] for sample in batch], dtype=torch.float32),
        "quality_valid": torch.as_tensor([bool(sample["quality_valid"]) for sample in batch], dtype=torch.bool),
    }
