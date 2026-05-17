from __future__ import annotations

from typing import Any

import numpy as np
import torch


DETECTOR_BOX_CODE_DIM = 8
DETECTOR_POS_ENC_DIM = 8
DETECTOR_SCORE_DIM = 1
DETECTOR_TOKEN_EXTRA_DIM = DETECTOR_BOX_CODE_DIM + DETECTOR_SCORE_DIM + DETECTOR_POS_ENC_DIM
DEFAULT_QUERY_FEATURE_KEYS = (
    "query_voxel_features",
    "query_features",
    "voxel_query_features",
    "voxelnext_query_features",
)
DEFAULT_QUERY_BOX_KEYS = (
    "query_boxes",
    "pred_boxes",
    "voxel_query_boxes",
    "voxelnext_query_boxes",
)
DEFAULT_QUERY_SCORE_KEYS = (
    "query_scores",
    "pred_scores",
    "voxel_query_scores",
    "voxelnext_query_scores",
)


def expected_detector_token_dim(query_feature_dim: int) -> int:
    return int(query_feature_dim) + DETECTOR_TOKEN_EXTRA_DIM


def encode_detector_boxes(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[-1] < 7:
        raise ValueError(f"Expected detector boxes shaped [N, >=7], got {tuple(boxes.shape)}")
    center = boxes[:, 0:3]
    dims = torch.log(boxes[:, 3:6].clamp_min(1e-4))
    yaw = boxes[:, 6:7]
    return torch.cat([center, dims, torch.sin(yaw), torch.cos(yaw)], dim=-1)


def build_detector_position_encoding(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[-1] < 7:
        raise ValueError(f"Expected detector boxes shaped [N, >=7], got {tuple(boxes.shape)}")
    xyz = boxes[:, 0:3]
    yaw = boxes[:, 6:7]
    return torch.cat([torch.sin(xyz), torch.cos(xyz), torch.sin(yaw), torch.cos(yaw)], dim=-1)


def build_detector_tokens(query_features: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    if query_features.ndim != 2:
        raise ValueError(f"Expected query features shaped [N, C], got {tuple(query_features.shape)}")
    if boxes.shape[0] != query_features.shape[0]:
        raise ValueError("Detector boxes and query features must have the same candidate count")
    scores = scores.reshape(-1, 1)
    if scores.shape[0] != query_features.shape[0]:
        raise ValueError("Detector scores and query features must have the same candidate count")
    box_code = encode_detector_boxes(boxes)
    pos_code = build_detector_position_encoding(boxes)
    return torch.cat([query_features, box_code, scores, pos_code], dim=-1)


def extract_detector_frame_tokens(
    batch_dict: dict[str, Any],
    max_tokens: int,
    *,
    query_feature_keys: tuple[str, ...] = DEFAULT_QUERY_FEATURE_KEYS,
    query_box_keys: tuple[str, ...] = DEFAULT_QUERY_BOX_KEYS,
    query_score_keys: tuple[str, ...] = DEFAULT_QUERY_SCORE_KEYS,
) -> list[np.ndarray]:
    query_features = _find_batched_tensor(batch_dict, query_feature_keys)
    query_boxes = _find_batched_tensor(batch_dict, query_box_keys)
    query_scores = _find_batched_tensor(batch_dict, query_score_keys)
    if query_features is None or query_boxes is None or query_scores is None:
        raise KeyError(
            "Missing VoxelNeXt query tensors in batch_dict. Expected explicit query feature/box/score keys "
            f"such as {query_feature_keys}, {query_box_keys}, {query_score_keys}; "
            "spatial_features fallback has been removed."
        )

    tokens: list[np.ndarray] = []
    num_batches = _infer_batch_size(query_features, batch_dict)
    for batch_idx in range(num_batches):
        cur_features = _slice_batch_tensor(query_features, batch_idx)
        cur_boxes = _slice_batch_tensor(query_boxes, batch_idx)
        cur_scores = _slice_batch_tensor(query_scores, batch_idx)
        if cur_features.numel() == 0:
            tokens.append(np.zeros((0, expected_detector_token_dim(query_features.shape[-1])), dtype=np.float32))
            continue
        candidate_tokens = build_detector_tokens(cur_features, cur_boxes, cur_scores)
        score_order = cur_scores.reshape(-1).topk(k=min(max_tokens, cur_scores.numel()), largest=True).indices
        tokens.append(candidate_tokens[score_order].detach().cpu().numpy().astype(np.float32))
    return tokens


def _find_batched_tensor(batch_dict: dict[str, Any], keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        value = batch_dict.get(key)
        if isinstance(value, torch.Tensor):
            return value.detach()
    pred_dicts = batch_dict.get("pred_dicts")
    if isinstance(pred_dicts, dict):
        for key in keys:
            value = pred_dicts.get(key)
            if isinstance(value, torch.Tensor):
                return value.detach()
    if isinstance(pred_dicts, list) and pred_dicts and isinstance(pred_dicts[0], dict):
        collected: list[torch.Tensor] = []
        for item in pred_dicts:
            tensor = None
            for key in keys:
                value = item.get(key)
                if isinstance(value, torch.Tensor):
                    tensor = value.detach()
                    break
            if tensor is None:
                return None
            collected.append(tensor)
        return _stack_batched_tensors(collected)
    return None


def _stack_batched_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.zeros((0, 0), dtype=torch.float32)
    if tensors[0].ndim == 1:
        max_len = max(int(t.shape[0]) for t in tensors)
        out = tensors[0].new_zeros((len(tensors), max_len))
        for idx, tensor in enumerate(tensors):
            out[idx, : tensor.shape[0]] = tensor
        return out
    max_len = max(int(t.shape[0]) for t in tensors)
    feat_dim = int(tensors[0].shape[-1])
    out = tensors[0].new_zeros((len(tensors), max_len, feat_dim))
    for idx, tensor in enumerate(tensors):
        out[idx, : tensor.shape[0]] = tensor
    return out


def _infer_batch_size(tensor: torch.Tensor, batch_dict: dict[str, Any]) -> int:
    if tensor.ndim >= 3:
        return int(tensor.shape[0])
    return int(batch_dict.get("batch_size", 1))


def _slice_batch_tensor(tensor: torch.Tensor, batch_idx: int) -> torch.Tensor:
    if tensor.ndim >= 3:
        return tensor[batch_idx]
    if tensor.ndim == 2 and batch_idx == 0:
        return tensor
    if tensor.ndim == 1 and batch_idx == 0:
        return tensor
    raise ValueError(f"Cannot slice batched tensor with shape {tuple(tensor.shape)} at batch index {batch_idx}")
