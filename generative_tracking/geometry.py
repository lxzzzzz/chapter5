from __future__ import annotations

import numpy as np


def boxes_iou_bev_axis_aligned(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Fast BEV IoU using axis-aligned extents from lidar boxes [x,y,z,dx,dy,dz,yaw]."""

    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 7)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 7)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a_min = boxes_a[:, :2] - boxes_a[:, 3:5] * 0.5
    a_max = boxes_a[:, :2] + boxes_a[:, 3:5] * 0.5
    b_min = boxes_b[:, :2] - boxes_b[:, 3:5] * 0.5
    b_max = boxes_b[:, :2] + boxes_b[:, 3:5] * 0.5

    inter_min = np.maximum(a_min[:, None, :], b_min[None, :, :])
    inter_max = np.minimum(a_max[:, None, :], b_max[None, :, :])
    inter_wh = np.clip(inter_max - inter_min, a_min=0.0, a_max=None)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = np.clip(boxes_a[:, 3] * boxes_a[:, 4], a_min=0.0, a_max=None)
    area_b = np.clip(boxes_b[:, 3] * boxes_b[:, 4], a_min=0.0, a_max=None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, np.maximum(union, 1e-6), out=np.zeros_like(inter), where=union > 0)


def boxes_iou_3d_axis_aligned(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Axis-aligned 3D IoU for lidar boxes [x,y,z,dx,dy,dz,yaw].

    This is not a rotated-box official KITTI/AB3DMOT IoU implementation, but it
    keeps the local evaluator dependency-free and uses the full 3D extents.
    """

    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 7)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 7)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a_min = boxes_a[:, :3] - boxes_a[:, 3:6] * 0.5
    a_max = boxes_a[:, :3] + boxes_a[:, 3:6] * 0.5
    b_min = boxes_b[:, :3] - boxes_b[:, 3:6] * 0.5
    b_max = boxes_b[:, :3] + boxes_b[:, 3:6] * 0.5

    inter_min = np.maximum(a_min[:, None, :], b_min[None, :, :])
    inter_max = np.minimum(a_max[:, None, :], b_max[None, :, :])
    inter_size = np.clip(inter_max - inter_min, a_min=0.0, a_max=None)
    inter = inter_size[..., 0] * inter_size[..., 1] * inter_size[..., 2]
    vol_a = np.clip(boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5], a_min=0.0, a_max=None)
    vol_b = np.clip(boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5], a_min=0.0, a_max=None)
    union = vol_a[:, None] + vol_b[None, :] - inter
    return np.divide(inter, np.maximum(union, 1e-6), out=np.zeros_like(inter), where=union > 0)


def greedy_match_by_iou(iou: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    matches: list[tuple[int, int, float]] = []
    if iou.size == 0:
        return matches
    used_a: set[int] = set()
    used_b: set[int] = set()
    flat_order = np.argsort(iou.reshape(-1))[::-1]
    num_b = iou.shape[1]
    for flat_idx in flat_order.tolist():
        score = float(iou.reshape(-1)[flat_idx])
        if score < threshold:
            break
        idx_a = flat_idx // num_b
        idx_b = flat_idx % num_b
        if idx_a in used_a or idx_b in used_b:
            continue
        used_a.add(idx_a)
        used_b.add(idx_b)
        matches.append((idx_a, idx_b, score))
    return matches
