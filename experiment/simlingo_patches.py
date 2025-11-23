"""Runtime monkey patches for Sim-Lingo language models.

이 모듈은 서브모듈 코드를 직접 수정하지 않고도 텍스트 토큰/로짓을
수집할 수 있도록 `greedy_sample`을 덮어씌웁니다. 모델 로드 전에
`patch_simlingo()`를 호출하면 됩니다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


def _patched_greedy_sample_llm(
    self,
    input_embeds: Tensor,
    inputs_mask: Optional[Tensor] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    cache_offset: int = 0,
    input_embed_matrix: Optional[Tensor] = None,
    logit_matrix: Optional[Tensor] = None,
    restrict_tokens: Optional[Tuple[int, int]] = None,
    attention_mask=None,
    position_ids=None,
):
    """Copy of LLM.greedy_sample with token/logit capture."""
    if input_embed_matrix is None:
        if self.embed_tokens is None:
            raise ValueError(
                "No input embeddings available because the model doesn't define a vocab. "
                "Please provide input_embed_matrix. "
            )
        input_embed_matrix = self.embed_tokens.weight
    if logit_matrix is None:
        if self.lm_head is None:
            raise ValueError(
                "No logit matrix available because the model doesn't define a vocab. "
                "Please provide logit_matrix. "
            )
        logit_matrix = self.lm_head.weight

    sampled_tokens = torch.empty(
        (input_embeds.size(0), max_new_tokens), device=input_embeds.device, dtype=torch.long
    )
    if eos_token_id is not None:
        sampled_tokens.fill_(eos_token_id)

    incomplete_seq_mask = torch.ones(input_embeds.size(0), dtype=torch.bool, device=input_embeds.device)
    token_score_seq = []
    for i in range(max_new_tokens):
        features, logits = self.forward(
            embeddings=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        last_hidden_state = features[:, -1]
        logits = F.linear(last_hidden_state, logit_matrix)
        next_token = self.sample_categorical(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, restrict_tokens=restrict_tokens
        )
        token_score = logits.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
        x = F.embedding(next_token.unsqueeze(1), input_embed_matrix)

        input_embeds = torch.cat([input_embeds, x], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((input_embeds.size(0), 1), device=input_embeds.device)], dim=1
        )

        sampled_tokens[incomplete_seq_mask, i] = next_token[incomplete_seq_mask]
        token_score_seq.append(token_score)

        if eos_token_id is not None:
            incomplete_seq_mask = sampled_tokens[:, i] != eos_token_id
            if not incomplete_seq_mask.any():
                sampled_tokens = sampled_tokens[:, : i + 1]
                break

    if token_score_seq:
        gen_len = len(token_score_seq)
        token_scores = torch.stack(token_score_seq, dim=1)
        tokens_trimmed = sampled_tokens[:, :gen_len]
        self.last_sampled_tokens = tokens_trimmed.detach()
        self.last_sampled_token_scores = token_scores.detach()
    else:
        self.last_sampled_tokens = None
        self.last_sampled_token_scores = None

    return sampled_tokens, input_embeds


def _patched_greedy_sample_llama(
    self,
    input_embeds: Tensor,
    inputs_mask: Optional[Tensor] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    cache_offset: int = 0,
    input_embed_matrix: Optional[Tensor] = None,
    logit_matrix: Optional[Tensor] = None,
    restrict_tokens: Optional[Tuple[int, int]] = None,
):
    """Copy of base-training Llama.greedy_sample with token/logit capture."""
    if input_embed_matrix is None:
        if self.embed_tokens is None:
            raise ValueError(
                "No input embeddings available because the model doesn't define a vocab. "
                "Please provide input_embed_matrix. "
            )
        input_embed_matrix = self.embed_tokens.weight
    if logit_matrix is None:
        if self.lm_head is None:
            raise ValueError(
                "No logit matrix available because the model doesn't define a vocab. "
                "Please provide logit_matrix. "
            )
        logit_matrix = self.lm_head.weight

    sampled_tokens = torch.empty(
        (input_embeds.size(0), max_new_tokens), device=input_embeds.device, dtype=torch.long
    )
    if eos_token_id is not None:
        sampled_tokens.fill_(eos_token_id)

    incomplete_seq_mask = torch.ones(input_embeds.size(0), dtype=torch.bool, device=input_embeds.device)
    token_score_seq = []
    for i in range(max_new_tokens):
        outputs = self.forward(embeddings=input_embeds)
        last_hidden_state = outputs[:, -1]

        logits = F.linear(last_hidden_state, logit_matrix)
        next_token = self.sample_categorical(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, restrict_tokens=restrict_tokens
        )
        token_score = logits.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
        x = F.embedding(next_token.unsqueeze(1), input_embed_matrix)

        input_embeds = torch.cat([input_embeds, x], dim=1)

        sampled_tokens[incomplete_seq_mask, i] = next_token[incomplete_seq_mask]
        token_score_seq.append(token_score)

        if eos_token_id is not None:
            incomplete_seq_mask = sampled_tokens[:, i] != eos_token_id
            if not incomplete_seq_mask.any():
                sampled_tokens = sampled_tokens[:, : i + 1]
                break

    if token_score_seq:
        gen_len = len(token_score_seq)
        token_scores = torch.stack(token_score_seq, dim=1)
        tokens_trimmed = sampled_tokens[:, :gen_len]
        self.last_sampled_tokens = tokens_trimmed.detach()
        self.last_sampled_token_scores = token_scores.detach()
    else:
        self.last_sampled_tokens = None
        self.last_sampled_token_scores = None

    return sampled_tokens, input_embeds


def patch_simlingo() -> None:
    """Apply monkey patches to Sim-Lingo language models."""
    try:
        from simlingo_training.models.language_model.llm import LLM
    except Exception:
        LLM = None
    if LLM is not None:
        LLM.greedy_sample = _patched_greedy_sample_llm

    try:
        from simlingo_base_training.models.language_model.llama import Llama
    except Exception:
        Llama = None
    if Llama is not None:
        Llama.greedy_sample = _patched_greedy_sample_llama

