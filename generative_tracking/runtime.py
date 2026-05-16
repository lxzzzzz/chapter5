from __future__ import annotations

import torch
import warnings


def select_device(device_cfg: str) -> torch.device:
    if device_cfg != "auto":
        return torch.device(device_cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_available = torch.cuda.is_available()
    if not cuda_available:
        return torch.device("cpu")
    try:
        torch.empty(1, device="cuda")
    except RuntimeError:
        return torch.device("cpu")
    return torch.device("cuda")
