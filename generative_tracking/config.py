from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .chapter3_features import expected_detector_token_dim

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "chapter3_root": "/media/ana-4090/LY/chapter3",
    "detector_cfg_file": "/path/to/voxelnext_a1.yaml",
    "detector_ckpt": "",
    "dataset": {
        "name": "xian",
        "split": "train",
        "datasets": ["xian", "v2x_seq", "v2x_real"],
        "info_paths": {
            "xian": {
                "train": "/media/ana-4090/LY/chapter3/data/v2x_xian/tracking_infos_train.pkl",
                "val": "/media/ana-4090/LY/chapter3/data/v2x_xian/tracking_infos_val.pkl",
            },
            "v2x_real": {
                "train": "/media/ana-4090/LY/chapter3/data/v2x_real/tracking_infos_train.pkl",
                "val": "/media/ana-4090/LY/chapter3/data/v2x_real/tracking_infos_val.pkl",
            },
            "v2x_seq": {
                "train": "/media/ana-4090/LY/chapter3/data/v2x_xian/tracking_infos_train.pkl",
                "val": "/media/ana-4090/LY/chapter3/data/v2x_xian/tracking_infos_val.pkl",
            },
        },
        "K": 3,
        "stride": 1,
        "max_objects": 64,
        "class_names": ["Car"],
        "box_center_range": [0.0, -20.0, -4.5, 120.0, 30.0, 10.5],
        "box_size_scale": [10.0, 10.0, 5.0],
        "box_yaw_scale": 3.141592653589793,
    },
    "model": {
        "use_mock_visual": True,
        "use_mock_llm": True,
        "feature_source": "gt_boxes",
        "detector_mode": "cache",
        "visual_dim": 128,
        "detector_query_feature_dim": 128,
        "detector_token_dim": expected_detector_token_dim(128),
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
        "lambda_obj": 1.0,
        "lambda_center": 2.0,
        "lambda_size": 1.0,
        "lambda_yaw": 1.0,
        "lambda_embed": 1.0,
        "no_object_weight": 0.1,
        "matching_obj_cost": 1.0,
        "matching_center_cost": 2.0,
        "matching_size_cost": 1.0,
        "matching_yaw_cost": 1.0,
        "ignore_index": -1,
        "lambda_quality": 1.0,
    },
    "detection_cache": {
        "root": "outputs/nova_qwen05b_a1/detection_cache",
        "paths": {},
        "score_thresh": 0.05,
        "max_dets_per_frame": 100,
        "class_name": "Car",
    },
    "nova": {
        "history_len": 3,
        "history_stride": 1,
        "det_gt_iou_threshold": 0.5,
        "association_threshold": 0.5,
        "max_lost_frames": 2,
        "geometry_hidden_size": 128,
        "negative_positive_ratio": 3.0,
        "box_token": "<box>",
        "yes_token": "Yes",
        "no_token": "No",
        "use_lora": False,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
    },
    "train": {
        "batch_size": 2,
        "num_workers": 0,
        "lr": 0.0001,
        "epochs": 1,
        "max_iters": 0,
        "log_interval": 1,
        "eval_interval_epochs": 3,
        "save_interval_epochs": 3,
        "eval_on_final": True,
        "eval_max_frames": 0,
        "best_metric": "mota",
        "best_mode": "max",
        "save_trainable_only": True,
        "resume": "",
    },
    "eval": {
        "batch_size": 1,
        "max_lost_frames": 2,
        "score_default": 1.0,
        "score_thresh": 0.3,
        "embedding_match_threshold": 0.5,
        "checkpoint": "",
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
