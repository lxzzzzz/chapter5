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
        "max_objects": 64,
        "class_names": ["Car", "Pedestrian", "Cyclist", "Truck", "Bus"],
    },
    "model": {
        "use_mock_visual": True,
        "use_mock_llm": True,
        "feature_source": "gt_boxes",
        "detector_mode": "cache",
        "visual_dim": 128,
        "detector_token_dim": 128,
        "max_detector_tokens": 256,
        "online_detector_batch_size": 1,
        "online_detector_workers": 0,
        "qformer_hidden_size": 128,
        "num_queries": 16,
        "num_attention_heads": 4,
        "num_track_queries": 64,
        "track_embed_dim": 128,
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
            "Generate the current 3D track set with object boxes, classes, and temporally consistent identities."
        ),
    },
    "evaluator": {
        "enabled": False,
        "iou_threshold": 0.5,
        "metrics_path": "outputs/tracklm_rs/tracking_metrics.json",
    },
    "loss": {
        "compute_det_loss": False,
        "lambda_cls": 1.0,
        "lambda_box": 5.0,
        "lambda_embed": 1.0,
        "no_object_weight": 0.1,
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
        "score_thresh": 0.3,
        "embedding_match_threshold": 0.5,
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
