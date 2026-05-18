from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .config import Config
from .nova_data import BOX_TOKEN_DEFAULT, NOVAFormulator
from .model import Adapter, MockFrozenTextLM, RealFrozenTextLM, encode_boxes_for_loss, normalize_boxes, resolve_llm_hidden_size


class NOVAGeometryEncoder(nn.Module):
    def __init__(self, cfg: Config, hidden_size: int):
        super().__init__()
        self.cfg = cfg
        num_classes = max(1, len(cfg.dataset.class_names))
        self.class_embed = nn.Embedding(num_classes, 16)
        self.token_type_embed = nn.Embedding(2, 16)
        self.time_mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 16))
        in_dim = 8 + 1 + 16 + 16 + 16
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        time_values: torch.Tensor,
    ) -> torch.Tensor:
        boxes_norm = normalize_boxes(boxes, self.cfg)
        box_code = encode_boxes_for_loss(boxes_norm)
        scores = scores.reshape(*box_code.shape[:-1], 1).clamp_min(0.0)
        class_ids = class_ids.clamp_min(0).clamp_max(self.class_embed.num_embeddings - 1)
        token_type_ids = token_type_ids.clamp_min(0).clamp_max(1)
        time_embed = self.time_mlp(time_values.reshape(*box_code.shape[:-1], 1).to(dtype=box_code.dtype))
        features = torch.cat(
            [
                box_code,
                scores.to(dtype=box_code.dtype),
                self.class_embed(class_ids),
                self.token_type_embed(token_type_ids),
                time_embed,
            ],
            dim=-1,
        )
        return self.net(features)


class NOVAAssociationModel(nn.Module):
    """NOVA-style association via prompt + injected box embeddings.

    Real LLM mode predicts the next answer token and uses the logits of the
    configured No/Yes tokens as the association logits. Mock mode keeps the
    same geometry path but uses a small decision head for fast unit tests.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        geom_hidden = int(cfg.nova.get("geometry_hidden_size", cfg.model.get("qformer_hidden_size", 128)))
        self.geometry_encoder = NOVAGeometryEncoder(cfg, geom_hidden)
        llm_hidden = resolve_llm_hidden_size(cfg)
        self.adapter = Adapter(geom_hidden, llm_hidden)
        if cfg.model.use_mock_llm:
            self.llm = MockFrozenTextLM(
                hidden_size=llm_hidden,
                num_heads=int(cfg.model.num_attention_heads),
                num_layers=int(cfg.model.mock_llm_layers),
                freeze=bool(cfg.model.freeze_llm),
            )
            self.mock_prefix = nn.Parameter(torch.randn(1, 4, llm_hidden) * 0.02)
            self.mock_answer = nn.Parameter(torch.randn(1, 1, llm_hidden) * 0.02)
            self.decision_head = nn.Sequential(nn.Linear(llm_hidden, llm_hidden), nn.GELU(), nn.Linear(llm_hidden, 2))
            self.tokenizer = None
            self.box_token_id = None
            self.yes_token_id = None
            self.no_token_id = None
        else:
            self.llm = RealFrozenTextLM(cfg, llm_hidden, bool(cfg.model.freeze_llm))
            self._configure_real_llm_tokens()
            self._maybe_enable_lora()
        self.norm = nn.LayerNorm(llm_hidden)
        self.quality_head = nn.Sequential(nn.Linear(llm_hidden, llm_hidden), nn.GELU(), nn.Linear(llm_hidden, 1))
        self.formulator = NOVAFormulator(cfg)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        history_boxes = batch["track_history_boxes"]
        history_scores = batch["track_history_scores"]
        history_class_ids = batch["track_history_class_ids"]
        history_mask = batch["track_history_mask"]
        candidate_box = batch["candidate_box"].unsqueeze(1)
        candidate_score = batch["candidate_score"].unsqueeze(1)
        candidate_class_id = batch["candidate_class_id"].unsqueeze(1)

        bsz, history_len, _ = history_boxes.shape
        device = history_boxes.device
        dtype = history_boxes.dtype
        history_type = torch.zeros((bsz, history_len), dtype=torch.long, device=device)
        candidate_type = torch.ones((bsz, 1), dtype=torch.long, device=device)
        if history_len > 1:
            history_time = torch.linspace(-1.0, -1.0 / history_len, history_len, device=device, dtype=dtype)
        else:
            history_time = torch.full((1,), -1.0, device=device, dtype=dtype)
        history_time = history_time.unsqueeze(0).expand(bsz, -1)
        candidate_time = torch.zeros((bsz, 1), dtype=dtype, device=device)

        hist_tokens = self.geometry_encoder(
            history_boxes,
            history_scores,
            history_class_ids,
            history_type,
            history_time,
        )
        cand_tokens = self.geometry_encoder(
            candidate_box,
            candidate_score,
            candidate_class_id,
            candidate_type,
            candidate_time,
        )
        hist_tokens = hist_tokens * history_mask.unsqueeze(-1).to(dtype=hist_tokens.dtype)
        box_mask = torch.cat(
            [history_mask, torch.ones((bsz, 1), dtype=torch.bool, device=device)],
            dim=1,
        )
        box_embeds = self.adapter(torch.cat([hist_tokens, cand_tokens], dim=1))
        if bool(self.cfg.model.use_mock_llm):
            match_logits, pair_context = self._mock_decision(box_embeds)
        else:
            match_logits, pair_context = self._real_llm_decision(batch, box_embeds, box_mask)
        quality = self.quality_head(pair_context).squeeze(-1).sigmoid()
        out: dict[str, Any] = {
            "match_logits": match_logits,
            "quality": quality,
            "match_prob": match_logits.softmax(dim=-1)[:, 1],
        }
        if "match_label" in batch:
            losses, metrics = nova_association_loss(out, batch, self.cfg)
            out.update(losses)
            out.update(metrics)
            out["loss"] = out["L_match"] + out["L_quality"]
        return out

    def _mock_decision(self, box_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = box_embeds.shape[0]
        prefix = self.mock_prefix.expand(bsz, -1, -1).to(dtype=box_embeds.dtype, device=box_embeds.device)
        answer = self.mock_answer.expand(bsz, -1, -1).to(dtype=box_embeds.dtype, device=box_embeds.device)
        lm_inputs = torch.cat([prefix, box_embeds, answer], dim=1)
        lm_context = self.llm(lm_inputs)
        pair_context = self.norm(lm_context[:, -1])
        return self.decision_head(pair_context), pair_context

    def _real_llm_decision(
        self,
        batch: dict[str, torch.Tensor],
        box_embeds: torch.Tensor,
        box_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_texts = batch.get("prompt_texts")
        if not prompt_texts:
            prompt_texts = self._fallback_prompts(batch)
        tokenizer = self.llm.tokenizer
        device = box_embeds.device
        encoded = tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=int(self.cfg.prompt.get("max_length", 128)),
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        embedding_layer = self.llm.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)
        inputs_embeds = self._inject_box_embeddings(input_ids, inputs_embeds, box_embeds, box_mask)
        output = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        answer_pos = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        row_idx = torch.arange(input_ids.shape[0], device=device)
        logits = output.logits[row_idx, answer_pos]
        match_logits = torch.stack([logits[:, int(self.no_token_id)], logits[:, int(self.yes_token_id)]], dim=-1)
        hidden = output.hidden_states[-1][row_idx, answer_pos]
        return match_logits.to(dtype=self.norm.weight.dtype), self.norm(hidden.to(dtype=self.norm.weight.dtype))

    def _inject_box_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        box_embeds: torch.Tensor,
        box_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.box_token_id is None:
            raise RuntimeError("box_token_id is not configured")
        out = inputs_embeds.clone()
        for bidx in range(input_ids.shape[0]):
            positions = torch.where(input_ids[bidx].eq(int(self.box_token_id)))[0]
            valid_embeds = box_embeds[bidx, box_mask[bidx].bool()]
            expected = valid_embeds.shape[0]
            if len(positions) != expected:
                raise ValueError(
                    f"Prompt has {len(positions)} box tokens but {expected} box embeddings. "
                    "Increase prompt.max_length or check NOVAFormulator."
                )
            if expected:
                out[bidx, positions[:expected]] = valid_embeds.to(dtype=out.dtype)
        return out

    def _fallback_prompts(self, batch: dict[str, torch.Tensor]) -> list[str]:
        history_mask = batch["track_history_mask"].detach().cpu().numpy()
        history_class_ids = batch["track_history_class_ids"].detach().cpu().numpy()
        candidate_class_ids = batch["candidate_class_id"].detach().cpu().numpy()
        track_ids = batch.get("track_id")
        if isinstance(track_ids, torch.Tensor):
            track_ids_np = track_ids.detach().cpu().numpy()
        else:
            track_ids_np = [-1] * len(candidate_class_ids)
        return [
            self.formulator.build_prompt(
                track_id=int(track_ids_np[idx]),
                history_mask=history_mask[idx],
                history_class_ids=history_class_ids[idx],
                candidate_class_id=int(candidate_class_ids[idx]),
            )
            for idx in range(len(candidate_class_ids))
        ]

    def _configure_real_llm_tokens(self) -> None:
        tokenizer = self.llm.tokenizer
        box_token = str(self.cfg.nova.get("box_token", BOX_TOKEN_DEFAULT))
        added = tokenizer.add_special_tokens({"additional_special_tokens": [box_token]})
        if added:
            self.llm.model.resize_token_embeddings(len(tokenizer))
            if bool(self.cfg.model.freeze_llm):
                for param in self.llm.model.get_input_embeddings().parameters():
                    param.requires_grad_(False)
                output_embeddings = self.llm.model.get_output_embeddings()
                if output_embeddings is not None:
                    for param in output_embeddings.parameters():
                        param.requires_grad_(False)
        self.box_token_id = int(tokenizer.convert_tokens_to_ids(box_token))
        self.yes_token_id = self._resolve_answer_token_id(str(self.cfg.nova.get("yes_token", "Yes")))
        self.no_token_id = self._resolve_answer_token_id(str(self.cfg.nova.get("no_token", "No")))

    def _resolve_answer_token_id(self, token_text: str) -> int:
        tokenizer = self.llm.tokenizer
        for text in (f" {token_text}", token_text):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                return int(ids[-1])
        raise ValueError(f"Could not resolve answer token id for {token_text!r}")

    def _maybe_enable_lora(self) -> None:
        if not bool(self.cfg.nova.get("use_lora", False)):
            return
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("peft is required when nova.use_lora=true. Install peft in the training environment.") from exc
        target_modules = list(self.cfg.nova.get("lora_target_modules", ["q_proj", "v_proj"]))
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(self.cfg.nova.get("lora_r", 8)),
            lora_alpha=int(self.cfg.nova.get("lora_alpha", 16)),
            lora_dropout=float(self.cfg.nova.get("lora_dropout", 0.05)),
            target_modules=target_modules,
        )
        self.llm.model = get_peft_model(self.llm.model, lora_cfg)


def nova_association_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: Config,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    labels = batch["match_label"].to(outputs["match_logits"].device).long()
    target_iou = batch["target_iou"].to(outputs["quality"].device).float()
    quality_valid = batch.get("quality_valid", labels.eq(1)).to(outputs["quality"].device).bool()
    l_match = F.cross_entropy(outputs["match_logits"], labels)
    if quality_valid.any():
        l_quality = F.smooth_l1_loss(outputs["quality"][quality_valid], target_iou[quality_valid], reduction="mean")
    else:
        l_quality = outputs["quality"].sum() * 0.0
    pred = outputs["match_logits"].argmax(dim=-1)
    acc = pred.eq(labels).float().mean() if labels.numel() else outputs["match_logits"].new_zeros(())
    positives = labels.eq(1)
    positive_recall = pred[positives].eq(1).float().mean() if positives.any() else outputs["match_logits"].new_zeros(())
    losses = {
        "L_match": l_match,
        "L_quality": l_quality * float(cfg.loss.get("lambda_quality", 1.0)),
    }
    metrics = {
        "match_acc": acc.detach(),
        "positive_recall": positive_recall.detach(),
        "positive_count": positives.float().sum().detach(),
    }
    return losses, metrics
