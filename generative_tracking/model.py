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
        frame_tokens = frame_tokens * batch["window_valid_mask"].unsqueeze(-1)
        return {
            "frame_tokens": frame_tokens,
        }


class Chapter3VisualFeatureExtractor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.detector_mode = str(cfg.model.get("detector_mode", "cache"))
        self.box_encoder = MockVisualFeatureExtractor(len(cfg.dataset.class_names), int(cfg.model.visual_dim))
        token_dim = int(cfg.model.get("detector_token_dim", cfg.model.visual_dim))
        self.detector_proj = nn.Linear(token_dim, int(cfg.model.visual_dim)) if token_dim != int(cfg.model.visual_dim) else nn.Identity()
        self.online_detector = None
        if self.detector_mode == "online":
            from .chapter3_online import OnlineChapter3Detector
            self.online_detector = OnlineChapter3Detector(cfg)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.box_encoder(batch)
        if self.detector_mode == "online":
            if self.online_detector is None:
                raise RuntimeError("online detector was not initialized")
            detector_tokens, detector_mask = self.online_detector.extract_window_tokens(
                batch["window_sequence_ids"],
                batch["window_frame_ids"],
                device=batch["current_boxes"].device,
            )
            batch["detector_frame_tokens"] = detector_tokens
            batch["detector_frame_valid_mask"] = detector_mask
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


class GenerativeTrackHead(nn.Module):
    def __init__(self, context_dim: int, num_queries: int, num_classes: int, track_embed_dim: int, num_heads: int):
        super().__init__()
        self.track_queries = nn.Parameter(torch.randn(num_queries, context_dim) * 0.02)
        self.attn = nn.MultiheadAttention(context_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(context_dim)
        self.ffn = nn.Sequential(
            nn.Linear(context_dim, context_dim * 4),
            nn.GELU(),
            nn.Linear(context_dim * 4, context_dim),
        )
        self.class_head = nn.Linear(context_dim, num_classes + 1)
        self.box_head = nn.Sequential(nn.Linear(context_dim, context_dim), nn.GELU(), nn.Linear(context_dim, 7))
        self.embed_head = nn.Sequential(nn.Linear(context_dim, context_dim), nn.GELU(), nn.Linear(context_dim, track_embed_dim))
        self.score_head = nn.Linear(context_dim, 1)

    def forward(self, lm_context: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = lm_context.shape[0]
        query = self.track_queries.unsqueeze(0).expand(bsz, -1, -1).to(dtype=lm_context.dtype)
        attended, _ = self.attn(query, lm_context, lm_context, need_weights=False)
        x = self.norm(query + attended)
        x = self.norm(x + self.ffn(x))
        return {
            "pred_logits": self.class_head(x),
            "pred_boxes": self.box_head(x),
            "pred_track_embeds": F.normalize(self.embed_head(x), dim=-1),
            "pred_scores": self.score_head(x).sigmoid().squeeze(-1),
        }


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
        self.track_head = GenerativeTrackHead(
            context_dim=llm_hidden,
            num_queries=int(cfg.model.num_track_queries),
            num_classes=len(cfg.dataset.class_names),
            track_embed_dim=int(cfg.model.track_embed_dim),
            num_heads=int(cfg.model.num_attention_heads),
        )

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
        out = self.track_head(lm_context)
        out["L_det"] = out["pred_boxes"].new_zeros(())
        if "target_boxes" in batch:
            losses, metrics = generative_track_loss(out, batch, self.cfg)
            out.update(losses)
            out.update(metrics)
            out["loss"] = out["L_det"] + out["L_cls"] + out["L_box"] + out["L_embed"]
        return out


def generative_track_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: Config,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    pred_embeds = outputs["pred_track_embeds"]
    target_boxes = batch["target_boxes"].to(pred_boxes.device)
    target_classes = batch["target_class_ids"].to(pred_logits.device)
    target_track_ids = batch["target_track_ids"].to(pred_logits.device)
    target_valid = batch["target_valid_mask"].to(pred_logits.device)
    bsz, num_queries, num_classes_plus_noobj = pred_logits.shape
    no_object_class = num_classes_plus_noobj - 1

    cls_targets = torch.full((bsz, num_queries), no_object_class, dtype=torch.long, device=pred_logits.device)
    matched_pred_indices: list[torch.Tensor] = []
    matched_target_indices: list[torch.Tensor] = []
    all_matched_embeds: list[torch.Tensor] = []
    all_matched_track_ids: list[torch.Tensor] = []
    box_losses: list[torch.Tensor] = []
    matched_count = 0

    for bidx in range(bsz):
        valid_idx = torch.where(target_valid[bidx])[0]
        if len(valid_idx) == 0:
            matched_pred_indices.append(torch.empty(0, dtype=torch.long, device=pred_logits.device))
            matched_target_indices.append(torch.empty(0, dtype=torch.long, device=pred_logits.device))
            continue
        cur_tgt_boxes = target_boxes[bidx, valid_idx]
        cur_tgt_classes = target_classes[bidx, valid_idx]
        pred_prob = pred_logits[bidx].softmax(dim=-1)
        cls_cost = -pred_prob[:, cur_tgt_classes]
        box_cost = torch.cdist(pred_boxes[bidx], cur_tgt_boxes, p=1) / pred_boxes.shape[-1]
        cost = cls_cost + box_cost
        pred_idx, tgt_local_idx = greedy_min_cost_match(cost.detach())
        if len(pred_idx) == 0:
            continue
        tgt_idx = valid_idx[tgt_local_idx]
        cls_targets[bidx, pred_idx] = target_classes[bidx, tgt_idx]
        box_losses.append(F.l1_loss(pred_boxes[bidx, pred_idx], target_boxes[bidx, tgt_idx], reduction="mean"))
        all_matched_embeds.append(pred_embeds[bidx, pred_idx])
        all_matched_track_ids.append(target_track_ids[bidx, tgt_idx])
        matched_pred_indices.append(pred_idx)
        matched_target_indices.append(tgt_idx)
        matched_count += int(len(pred_idx))

    class_weights = pred_logits.new_ones((num_classes_plus_noobj,))
    class_weights[no_object_class] = float(cfg.loss.no_object_weight)
    l_cls = F.cross_entropy(pred_logits.flatten(0, 1), cls_targets.flatten(), weight=class_weights)
    l_box = torch.stack(box_losses).mean() if box_losses else pred_boxes.sum() * 0.0
    if all_matched_embeds:
        matched_embeds = torch.cat(all_matched_embeds, dim=0)
        matched_track_ids = torch.cat(all_matched_track_ids, dim=0)
        l_embed = track_embedding_consistency_loss(matched_embeds, matched_track_ids)
    else:
        l_embed = pred_embeds.sum() * 0.0

    losses = {
        "L_cls": l_cls * float(cfg.loss.lambda_cls),
        "L_box": l_box * float(cfg.loss.lambda_box),
        "L_embed": l_embed * float(cfg.loss.lambda_embed),
    }
    pred_classes = pred_logits.argmax(dim=-1)
    valid_cls = cls_targets.ne(no_object_class)
    if valid_cls.any():
        cls_acc = pred_classes[valid_cls].eq(cls_targets[valid_cls]).float().mean()
    else:
        cls_acc = pred_logits.new_zeros(())
    metrics = {
        "match_count": pred_logits.new_tensor(float(matched_count)),
        "track_cls_acc": cls_acc.detach(),
    }
    return losses, metrics


def greedy_min_cost_match(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cost.numel() == 0:
        device = cost.device
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
    num_queries, num_targets = cost.shape
    order = torch.argsort(cost.flatten())
    used_q: set[int] = set()
    used_t: set[int] = set()
    pred_indices: list[int] = []
    target_indices: list[int] = []
    for flat in order.tolist():
        q = flat // num_targets
        t = flat % num_targets
        if q in used_q or t in used_t:
            continue
        used_q.add(q)
        used_t.add(t)
        pred_indices.append(q)
        target_indices.append(t)
        if len(target_indices) >= min(num_queries, num_targets):
            break
    device = cost.device
    return torch.as_tensor(pred_indices, dtype=torch.long, device=device), torch.as_tensor(target_indices, dtype=torch.long, device=device)


def track_embedding_consistency_loss(embeds: torch.Tensor, track_ids: torch.Tensor) -> torch.Tensor:
    valid = track_ids.ge(0)
    embeds = embeds[valid]
    track_ids = track_ids[valid]
    if embeds.shape[0] < 2:
        return embeds.sum() * 0.0
    sim = embeds @ embeds.transpose(0, 1)
    same = track_ids[:, None].eq(track_ids[None, :])
    eye = torch.eye(len(track_ids), dtype=torch.bool, device=track_ids.device)
    pos = same & ~eye
    neg = (~same) & ~eye
    loss = embeds.sum() * 0.0
    terms = 0
    if pos.any():
        loss = loss + (1.0 - sim[pos]).mean()
        terms += 1
    if neg.any():
        loss = loss + F.relu(sim[neg] - 0.2).mean()
        terms += 1
    return loss / max(terms, 1)


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
