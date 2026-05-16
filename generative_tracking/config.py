from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "chapter3_root": "/home/lx/chapter3",
    "detector_cfg_file": "/home/lx/chapter3/tools/cfgs/dair_v2x_models/fusion_voxelnext_v2x_xian.yaml",
    "detector_ckpt": "",
    "dataset": {
        "name": "xian",
        "split": "train",
        "object_source": "gt",
        "detection_paths": {"train": "", "val": ""},
        "detection_match_iou": 0.1,
        "detection_score_thresh": 0.0,
        "datasets": ["xian", "v2x_seq", "v2x_real"],
        "info_paths": {
            "xian": {
                "train": "/home/lx/chapter3/data/v2x_xian/tracking_infos_train.pkl",
                "val": "/home/lx/chapter3/data/v2x_xian/tracking_infos_val.pkl",
            },
            "v2x_real": {
                "train": "/home/lx/chapter3/data/v2x_real/tracking_infos_train.pkl",
                "val": "/home/lx/chapter3/data/v2x_real/tracking_infos_val.pkl",
            },
            "v2x_seq": {
                "train": "/home/lx/chapter3/data/v2x_xian/tracking_infos_train.pkl",
                "val": "/home/lx/chapter3/data/v2x_xian/tracking_infos_val.pkl",
            },
        },
        "K": 3,
        "stride": 1,
        "max_history_tracks": 32,
        "max_current_objects": 64,
        "class_names": ["Car", "Pedestrian", "Cyclist", "Truck", "Bus"],
    },
    "model": {
        "use_mock_visual": True,
        "use_mock_llm": True,
        "feature_source": "gt_boxes",
        "visual_dim": 128,
        "detector_token_dim": 128,
        "max_detector_tokens": 256,
        "qformer_hidden_size": 128,
        "num_queries": 16,
        "num_attention_heads": 4,
        "llm_hidden_size": 0,
        "mock_llm_hidden_size": 128,
        "mock_llm_layers": 2,
        "llm_model_name_or_path": "",
        "llm_torch_dtype": "float16",
        "llm_trust_remote_code": True,
        "llm_low_cpu_mem_usage": True,
        "llm_attn_implementation": "",
        "freeze_detector": True,
        "freeze_llm": True,
    },
    "prompt": {
        "enabled": False,
        "max_length": 128,
        "template": (
            "Sequence {sequence_id}, frame {frame_id}. "
            "Associate current 3D objects to recent history tracks; use NEW for unseen objects."
        ),
    },
    "evaluator": {
        "enabled": False,
        "iou_threshold": 0.5,
        "metrics_path": "outputs/tracklm_rs/tracking_metrics.json",
    },
    "loss": {
        "compute_det_loss": False,
        "lambda_id": 1.0,
        "id_loss_type": "pointer_ce",
        "ignore_index": -1,
    },
    "train": {
        "batch_size": 2,
        "num_workers": 0,
        "lr": 0.001,
        "max_iters": 100,
        "log_interval": 1,
    },
    "eval": {
        "batch_size": 1,
        "max_lost_frames": 2,
        "score_default": 1.0,
    },
    "output_dir": "outputs/tracklm_rs",
}


class Config(dict):
    """Dictionary with attribute access for compact prototype code."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Config({k: _to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config(v) for v in value]
    return value


def _merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(cfg_file: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    cfg = deepcopy(DEFAULT_CONFIG)
    if cfg_file:
        with Path(cfg_file).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = _merge(cfg, loaded)
    if overrides:
        cfg = _merge(cfg, overrides)
    return _to_config(cfg)


def resolve_info_path(cfg: Config, split: str | None = None) -> str:
    dataset_name = cfg.dataset.name
    split_name = split or cfg.dataset.split
    try:
        return cfg.dataset.info_paths[dataset_name][split_name]
    except KeyError as exc:
        known = ", ".join(cfg.dataset.info_paths.keys())
        raise KeyError(f"Unknown dataset/split {dataset_name!r}/{split_name!r}; known datasets: {known}") from exc
