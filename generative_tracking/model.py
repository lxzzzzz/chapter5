from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .config import Config


class MockVisualFeatureExtractor(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.class_embed = nn.Embedding(max(num_classes, 1), 16)
        self.track_embed = nn.Embedding(4096, 16)
        self.mlp = nn.Sequential(
            nn.Linear(7 + 16 + 16, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )

    def _encode(self, boxes: torch.Tensor, class_ids: torch.Tensor, track_ids: torch.Tensor | None = None) -> torch.Tensor:
        class_ids = class_ids.clamp_min(0).clamp_max(self.class_embed.num_embeddings - 1)
        if track_ids is None:
            track_ids = torch.zeros_like(class_ids)
        track_ids = track_ids.clamp_min(0).remainder(self.track_embed.num_embeddings)
        x = torch.cat([boxes, self.class_embed(class_ids), self.track_embed(track_ids)], dim=-1)
        return self.mlp(x)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frame_tokens = self._encode(batch["window_boxes"], batch["window_class_ids"], batch["window_track_ids"])
        current_tokens = self._encode(batch["current_boxes"], batch["current_class_ids"], batch["current_track_ids"])
        history_tokens = self._encode(batch["history_boxes"], batch["history_class_ids"], batch["history_track_ids"])
        frame_tokens = frame_tokens * batch["window_valid_mask"].unsqueeze(-1)
        current_tokens = current_tokens * batch["current_valid_mask"].unsqueeze(-1)
        history_tokens = history_tokens * batch["history_valid_mask"].unsqueeze(-1)
        return {
            "frame_tokens": frame_tokens,
            "current_tokens": current_tokens,
            "history_tokens": history_tokens,
        }


class Chapter3VisualFeatureExtractor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.box_encoder = MockVisualFeatureExtractor(len(cfg.dataset.class_names), int(cfg.model.visual_dim))
        token_dim = int(cfg.model.get("detector_token_dim", cfg.model.visual_dim))
        self.detector_proj = nn.Linear(token_dim, int(cfg.model.visual_dim)) if token_dim != int(cfg.model.visual_dim) else nn.Identity()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.box_encoder(batch)
        if "detector_frame_tokens" not in batch:
            return features
        detector_tokens = self.detector_proj(batch["detector_frame_tokens"])
        detector_mask = batch.get("detector_frame_valid_mask")
        if detector_mask is None:
            detector_mask = detector_tokens.abs().sum(dim=-1).gt(0)
        features["frame_tokens"] = detector_tokens
        batch["window_valid_mask"] = detector_mask
        return features


class TrackQFormer(nn.Module):
    def __init__(self, visual_dim: int, hidden_size: int, num_queries: int, num_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)
        self.key_proj = nn.Linear(visual_dim, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, frame_tokens: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        bsz = frame_tokens.shape[0]
        if frame_tokens.dim() == 4:
            memory_in = frame_tokens.flatten(1, 2)
            mask_in = frame_mask.flatten(1, 2)
        else:
            memory_in = frame_tokens
            mask_in = frame_mask
        memory = self.key_proj(memory_in)
        key_padding_mask = ~mask_in
        if key_padding_mask.all(dim=1).any():
            # MultiheadAttention cannot attend to an all-masked row. Leave one zero token visible.
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[key_padding_mask.all(dim=1), 0] = False
        query = self.query.unsqueeze(0).expand(bsz, -1, -1)
        attended, _ = self.attn(query, memory, memory, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm(query + attended)
        return self.norm(x + self.ffn(x))


class Adapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MockFrozenTextLM(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int, freeze: bool):
        super().__init__()
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        if freeze:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs_embeds)


class RealFrozenTextLM(nn.Module):
    def __init__(self, cfg: Config, hidden_size: int, freeze: bool):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required when model.use_mock_llm=false") from exc
        model_name_or_path = str(cfg.model.llm_model_name_or_path)
        kwargs: dict[str, Any] = {
            "trust_remote_code": bool(cfg.model.get("llm_trust_remote_code", True)),
            "low_cpu_mem_usage": bool(cfg.model.get("llm_low_cpu_mem_usage", True)),
        }
        dtype = _parse_torch_dtype(str(cfg.model.get("llm_torch_dtype", "float16")))
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        attn_impl = str(cfg.model.get("llm_attn_implementation", ""))
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=kwargs["trust_remote_code"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_size = hidden_size
        if freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)

    def encode_prompts(self, prompts: list[str], device: torch.device, max_length: int) -> torch.Tensor:
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        return self.model.get_input_embeddings()(input_ids)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        embed_weight = self.model.get_input_embeddings().weight
        inputs_embeds = inputs_embeds.to(dtype=embed_weight.dtype)
        output = self.model(inputs_embeds=inputs_embeds, use_cache=False, output_hidden_states=True)
        return output.hidden_states[-1]


class PointerIDHead(nn.Module):
    def __init__(self, token_dim: int, context_dim: int):
        super().__init__()
        self.current_proj = nn.Linear(token_dim, context_dim)
        self.history_proj = nn.Linear(token_dim, context_dim)
        self.context_proj = nn.Linear(context_dim, context_dim)
        self.new_head = nn.Sequential(nn.Linear(context_dim, context_dim), nn.GELU(), nn.Linear(context_dim, 1))

    def forward(
        self,
        current_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
        history_valid_mask: torch.Tensor,
        lm_context: torch.Tensor,
    ) -> torch.Tensor:
        lm_context = lm_context.to(dtype=current_tokens.dtype)
        context = lm_context.mean(dim=1).unsqueeze(1)
        current = self.current_proj(current_tokens) + self.context_proj(context)
        history = self.history_proj(history_tokens)
        logits_hist = torch.matmul(current, history.transpose(1, 2)) / math.sqrt(current.shape[-1])
        logits_hist = logits_hist.masked_fill(~history_valid_mask.unsqueeze(1), torch.finfo(logits_hist.dtype).min / 2)
        logits_new = self.new_head(current)
        return torch.cat([logits_hist, logits_new], dim=-1)


class TrackLMRS(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        num_classes = len(cfg.dataset.class_names)
        visual_dim = int(cfg.model.visual_dim)
        if cfg.model.use_mock_visual:
            self.visual = MockVisualFeatureExtractor(num_classes, visual_dim)
        else:
            self.visual = Chapter3VisualFeatureExtractor(cfg)
        q_hidden = int(cfg.model.qformer_hidden_size)
        self.qformer = TrackQFormer(
            visual_dim=visual_dim,
            hidden_size=q_hidden,
            num_queries=int(cfg.model.num_queries),
            num_heads=int(cfg.model.num_attention_heads),
        )
        llm_hidden = resolve_llm_hidden_size(cfg)
        self.adapter = Adapter(q_hidden, llm_hidden)
        if cfg.model.use_mock_llm:
            self.llm = MockFrozenTextLM(
                hidden_size=llm_hidden,
                num_heads=int(cfg.model.num_attention_heads),
                num_layers=int(cfg.model.mock_llm_layers),
                freeze=bool(cfg.model.freeze_llm),
            )
        else:
            self.llm = RealFrozenTextLM(cfg, llm_hidden, bool(cfg.model.freeze_llm))
        self.pointer_head = PointerIDHead(visual_dim, llm_hidden)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        features = self.visual(batch)
        q_tokens = self.qformer(features["frame_tokens"], batch["window_valid_mask"])
        lm_inputs = self.adapter(q_tokens)
        if bool(self.cfg.prompt.get("enabled", False)) and (not self.cfg.model.use_mock_llm) and "prompt_texts" in batch:
            prompt_embeds = self.llm.encode_prompts(
                batch["prompt_texts"],
                device=lm_inputs.device,
                max_length=int(self.cfg.prompt.max_length),
            )
            prompt_embeds = prompt_embeds.to(dtype=lm_inputs.dtype)
            lm_inputs = torch.cat([prompt_embeds, lm_inputs], dim=1)
        lm_context = self.llm(lm_inputs)
        logits = self.pointer_head(
            features["current_tokens"],
            features["history_tokens"],
            batch["history_valid_mask"],
            lm_context,
        )
        out: dict[str, Any] = {"logits": logits, "L_det": logits.new_zeros(())}
        if "pointer_labels" in batch:
            loss_id, metrics = pointer_loss_and_metrics(
                logits,
                batch["pointer_labels"],
                batch["current_valid_mask"],
                ignore_index=int(self.cfg.loss.ignore_index),
                lambda_id=float(self.cfg.loss.lambda_id),
            )
            out.update(metrics)
            out["L_id"] = loss_id
            out["loss"] = out["L_det"] + loss_id
        return out


def pointer_loss_and_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    current_valid_mask: torch.Tensor,
    ignore_index: int,
    lambda_id: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = current_valid_mask & labels.ne(ignore_index)
    if not valid.any():
        zero = logits.sum() * 0.0
        return zero, {"pointer_acc": zero.detach(), "new_acc": zero.detach()}
    valid_logits = logits[valid]
    valid_labels = labels[valid].clamp_max(logits.shape[-1] - 1)
    loss = F.cross_entropy(valid_logits, valid_labels) * lambda_id
    pred = valid_logits.argmax(dim=-1)
    pointer_acc = pred.eq(valid_labels).float().mean()
    new_index = logits.shape[-1] - 1
    new_mask = valid_labels.eq(new_index)
    if new_mask.any():
        new_acc = pred[new_mask].eq(valid_labels[new_mask]).float().mean()
    else:
        new_acc = logits.new_zeros(())
    return loss, {"pointer_acc": pointer_acc.detach(), "new_acc": new_acc.detach()}


def resolve_llm_hidden_size(cfg: Config) -> int:
    configured = int(cfg.model.get("llm_hidden_size", 0))
    if cfg.model.use_mock_llm:
        return configured if configured > 0 else int(cfg.model.mock_llm_hidden_size)
    if configured > 0:
        return configured
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required to infer llm_hidden_size when model.use_mock_llm=false") from exc
    llm_cfg = AutoConfig.from_pretrained(
        str(cfg.model.llm_model_name_or_path),
        trust_remote_code=bool(cfg.model.get("llm_trust_remote_code", True)),
    )
    hidden = getattr(llm_cfg, "hidden_size", None) or getattr(llm_cfg, "n_embd", None)
    if hidden is None:
        raise ValueError("Could not infer LLM hidden size; set model.llm_hidden_size explicitly.")
    return int(hidden)


def _parse_torch_dtype(name: str) -> torch.dtype | None:
    if not name or name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported llm_torch_dtype={name!r}")
    return mapping[name]
