from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .chapter3_features import expected_detector_token_dim, extract_detector_frame_tokens
from .config import Config


class OnlineChapter3Detector:
    """Frozen chapter3 detector used as an online visual feature extractor."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.chapter3_root = Path(cfg.chapter3_root)
        self.max_tokens = int(cfg.model.max_detector_tokens)
        self.dataset = None
        self.model = None
        self.load_data_to_gpu = None
        self._key_to_index: dict[tuple[str, str], int] = {}
        self._build()

    def _build(self) -> None:
        tools_dir = self.chapter3_root / "tools"
        for path in (str(tools_dir), str(self.chapter3_root)):
            if path not in sys.path:
                sys.path.insert(0, path)

        from pcdet.config import cfg as det_cfg
        from pcdet.config import cfg_from_yaml_file
        from pcdet.datasets import build_dataloader
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils

        if not str(self.cfg.detector_ckpt):
            raise ValueError("Set detector_ckpt before using model.detector_mode=online.")

        old_cwd = Path.cwd()
        os.chdir(self.chapter3_root)
        try:
            det_cfg.clear()
            det_cfg.ROOT_DIR = self.chapter3_root.resolve()
            det_cfg.LOCAL_RANK = 0
            cfg_from_yaml_file(str(self.cfg.detector_cfg_file), det_cfg)
            all_info_paths = []
            for paths in det_cfg.DATA_CONFIG.INFO_PATH.values():
                for path in paths:
                    if path not in all_info_paths:
                        all_info_paths.append(path)
            det_cfg.DATA_CONFIG.INFO_PATH["test"] = all_info_paths
        finally:
            os.chdir(old_cwd)

        logger = common_utils.create_logger(log_file=None, rank=0)
        dataset, _loader, _sampler = build_dataloader(
            dataset_cfg=det_cfg.DATA_CONFIG,
            class_names=det_cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            root_path=self.chapter3_root / det_cfg.DATA_CONFIG.DATA_PATH,
            workers=0,
            logger=logger,
            training=False,
        )
        model = build_network(det_cfg.MODEL, num_class=len(det_cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=str(self.cfg.detector_ckpt), logger=logger, to_cpu=False)
        model.cuda()
        model.eval()
        if bool(self.cfg.model.freeze_detector):
            for param in model.parameters():
                param.requires_grad_(False)

        self.det_cfg = det_cfg
        self.dataset = dataset
        self.model = model
        self.load_data_to_gpu = load_data_to_gpu
        self._key_to_index = self._build_index(dataset)

    def extract_window_tokens(
        self,
        window_sequence_ids: list[list[str]],
        window_frame_ids: list[list[str]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.dataset is not None and self.model is not None and self.load_data_to_gpu is not None
        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        token_dim = int(self.cfg.model.get("detector_token_dim", expected_detector_token_dim(int(self.cfg.model.get("detector_query_feature_dim", 128)))))
        with torch.no_grad():
            for seq_ids, frame_ids in zip(window_sequence_ids, window_frame_ids):
                sample_tokens: list[np.ndarray] = []
                for seq, frame in zip(seq_ids, frame_ids):
                    if not seq or not frame:
                        sample_tokens.append(np.zeros((0, token_dim), dtype=np.float32))
                        continue
                    index = self._lookup_index(seq, frame)
                    raw = self.dataset[index]
                    batch_dict = self.dataset.collate_batch([raw])
                    self.load_data_to_gpu(batch_dict)
                    self.model(batch_dict)
                    tokens = extract_detector_frame_tokens(batch_dict, self.max_tokens)
                    cur = tokens[0] if tokens else np.zeros((0, token_dim), dtype=np.float32)
                    if cur.ndim == 1:
                        cur = cur.reshape(1, -1)
                    if cur.shape[-1] != token_dim:
                        raise ValueError(
                            f"Online detector token dim {cur.shape[-1]} does not match model.detector_token_dim={token_dim}"
                        )
                    sample_tokens.append(cur[: self.max_tokens])
                row, mask = _pad_concat_tokens(sample_tokens, self.max_tokens, token_dim)
                rows.append(row)
                masks.append(mask)
        return torch.stack(rows, dim=0).to(device), torch.stack(masks, dim=0).to(device)

    def _lookup_index(self, sequence_id: str, frame_id: str) -> int:
        key = (str(sequence_id), str(frame_id))
        if key not in self._key_to_index:
            raise KeyError(f"Frame {key} not found in chapter3 detector dataset")
        return self._key_to_index[key]

    @staticmethod
    def _build_index(dataset: Any) -> dict[tuple[str, str], int]:
        infos = getattr(dataset, "det_infos", None) or getattr(dataset, "tracking_infos", None)
        if infos is None:
            raise AttributeError("chapter3 dataset does not expose det_infos/tracking_infos for frame lookup")
        mapping: dict[tuple[str, str], int] = {}
        for idx, info in enumerate(infos):
            seq = str(info.get("sequence_id", ""))
            frame = str(info.get("frame_id", info.get("frame_idx", "")))
            mapping[(seq, frame)] = idx
            mapping[(seq, f"{seq}_{frame}")] = idx
        return mapping


def _pad_concat_tokens(tokens_per_frame: list[np.ndarray], max_tokens: int, token_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    total = max(1, len(tokens_per_frame) * max_tokens)
    out = torch.zeros((total, token_dim), dtype=torch.float32)
    mask = torch.zeros((total,), dtype=torch.bool)
    for frame_idx, tokens in enumerate(tokens_per_frame):
        n = min(len(tokens), max_tokens)
        if n == 0:
            continue
        start = frame_idx * max_tokens
        out[start:start + n] = torch.as_tensor(tokens[:n], dtype=torch.float32)
        mask[start:start + n] = True
    return out, mask
