from __future__ import annotations

from typing import Any

import numpy as np
import torch


def extract_detector_frame_tokens(batch_dict: dict[str, Any], max_tokens: int) -> list[np.ndarray]:
    if "spatial_features" in batch_dict and isinstance(batch_dict["spatial_features"], torch.Tensor):
        bev = batch_dict["spatial_features"].detach()
        bsz, channels, height, width = bev.shape
        flat = bev.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
        score = flat.norm(dim=-1)
        k = min(max_tokens, flat.shape[1])
        indices = score.topk(k=k, dim=1).indices
        return [flat[b, indices[b]].cpu().numpy().astype(np.float32) for b in range(bsz)]

    sparse = batch_dict.get("encoded_spconv_tensor", None)
    if sparse is not None and hasattr(sparse, "features") and hasattr(sparse, "indices"):
        features = sparse.features.detach()
        indices = sparse.indices.detach()
        batch_size = int(batch_dict.get("batch_size", int(indices[:, 0].max().item()) + 1))
        tokens = []
        for batch_idx in range(batch_size):
            mask = indices[:, 0].eq(batch_idx)
            cur = features[mask]
            if len(cur) == 0:
                tokens.append(np.zeros((0, features.shape[-1]), dtype=np.float32))
                continue
            score = cur.norm(dim=-1)
            k = min(max_tokens, len(cur))
            top = score.topk(k=k).indices
            tokens.append(cur[top].cpu().numpy().astype(np.float32))
        return tokens

    return []
