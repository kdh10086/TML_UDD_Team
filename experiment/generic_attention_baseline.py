"""
Generic Attention Explainability utilities adapted for Sim-Lingo VLA experiments.

이 모듈은 Transformer-MM-Explainability 레포지토리(VisualBERT 백엔드)의
`ExplanationGenerator` 구현을 참고하여, Sim-Lingo 추론 기록물에 저장된
어텐션 맵과 그래디언트를 받아 GAE/rollout 방식의 토큰 관련성 맵 및
이미지 히트맵을 생성하는 베이스라인을 제공합니다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2


def compute_rollout_attention(
    all_layer_matrices: Sequence[torch.Tensor],
    start_layer: int = 0,
    add_residual: bool = True,
) -> torch.Tensor:
    """멀티 레이어 어텐션 행렬을 residual 포함하여 roll-out."""
    if not all_layer_matrices:
        raise ValueError("Empty attention matrix list passed to rollout.")

    matrices = []
    for mat in all_layer_matrices:
        if mat.dim() != 3:
            raise ValueError(f"Expected [B,S,S] matrix, got shape {mat.shape}")
        if add_residual:
            eye = torch.eye(mat.size(-1), device=mat.device, dtype=mat.dtype)
            mat = mat + eye.unsqueeze(0)
        matrices.append(mat)

    joint = matrices[start_layer]
    for idx in range(start_layer + 1, len(matrices)):
        joint = matrices[idx].bmm(joint)
    return joint


def _normalize_rows(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = matrix.sum(dim=-1, keepdim=True).clamp_min(eps)
    return matrix / denom


def _prepare_single_matrix(
    attn: torch.Tensor,
    grad: Optional[torch.Tensor],
    clamp: bool = True,
    average_heads: bool = True,
) -> torch.Tensor:
    """어텐션/그래디언트 텐서를 [B,S,S] 형태로 변환."""
    if attn.dim() == 4:
        # assume [B, H, S, S]
        bsz = attn.size(0)
    elif attn.dim() == 3:
        bsz = 1
        attn = attn.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected attention tensor shape {attn.shape}")

    if grad is not None:
        if grad.shape != attn.shape:
            if grad.dim() == attn.dim() - 1:
                grad = grad.unsqueeze(0)
            else:
                raise ValueError(f"Gradient shape {grad.shape} incompatible with attn {attn.shape}")
        grad = grad.to(attn.device)
        attn = attn * grad

    if clamp:
        attn = attn.clamp(min=0)

    if average_heads:
        attn = attn.mean(dim=1, keepdim=False)

    return attn  # [B, S, S]


def prepare_attention_matrices(
    attention_records: Dict[str, Iterable[Dict[str, torch.Tensor]]],
    layer_filter: Optional[str] = None,
    sort_keys: bool = True,
) -> List[torch.Tensor]:
    """Recorder에서 저장한 dict를 GAE 입력 행렬로 변환."""
    items = attention_records.items()
    if sort_keys:
        items = sorted(items, key=lambda kv: kv[0])

    matrices: List[torch.Tensor] = []
    for name, records in items:
        if layer_filter is not None and layer_filter not in name:
            continue
        layer_tensors: List[torch.Tensor] = []
        for entry in records:
            attn = entry["attn"]
            grad = entry.get("grad")
            layer_tensors.append(_prepare_single_matrix(attn, grad))
        if not layer_tensors:
            continue
        layer_matrix = torch.stack(layer_tensors, dim=0).mean(dim=0)
        matrices.append(layer_matrix)
    if not matrices:
        raise ValueError("No attention matrices found after filtering.")
    return matrices


def build_token_relevance(
    attention_records: Dict[str, Iterable[Dict[str, torch.Tensor]]],
    output_token_index: int = 0,
    layer_filter: Optional[str] = "vision_block",
    start_layer: int = 0,
    residual_alpha: float = 0.5,
) -> torch.Tensor:
    """Gradient-weighted 어텐션을 사용해 토큰 관련성을 계산."""
    matrices = prepare_attention_matrices(attention_records, layer_filter=layer_filter)
    processed: List[torch.Tensor] = []
    for mat in matrices:
        mat = _normalize_rows(mat)
        alpha = residual_alpha
        eye = torch.eye(mat.size(-1), device=mat.device, dtype=mat.dtype).unsqueeze(0)
        mat = alpha * eye + (1 - alpha) * mat
        processed.append(mat)
    rollout = compute_rollout_attention(processed, start_layer=start_layer, add_residual=False)
    if output_token_index >= rollout.size(1):
        raise IndexError(f"Token index {output_token_index} out of range for rollout matrix.")
    relevance = rollout[:, output_token_index, :]
    return relevance.squeeze(0)


def extract_image_token_scores(
    token_relevance: torch.Tensor,
    meta: Dict[str, int],
    cls_has_index_zero: bool = True,
) -> torch.Tensor:
    """토큰 관련성에서 이미지 토큰 구간만 추출."""
    num_image_tokens = meta.get("num_total_image_tokens")
    if num_image_tokens is None:
        raise KeyError("meta dict must contain 'num_total_image_tokens'.")
    start = 1 if cls_has_index_zero else 0
    end = start + num_image_tokens
    if end > token_relevance.numel():
        raise ValueError(
            f"Token relevance length {token_relevance.numel()} smaller than requested slice {end}."
        )
    return token_relevance[start:end]


def tokens_to_heatmap(
    image_token_scores: torch.Tensor,
    meta: Dict[str, int],
    normalize: bool = True,
) -> torch.Tensor:
    """이미지 토큰 관련성을 원본 해상도 히트맵으로 반환."""
    num_tokens = image_token_scores.numel()
    grid_size = int(math.sqrt(num_tokens))
    if grid_size * grid_size != num_tokens:
        raise ValueError(
            f"Number of image tokens ({num_tokens}) is not a perfect square; "
            "cannot reshape into 2D grid automatically."
        )
    heatmap = image_token_scores.reshape(1, 1, grid_size, grid_size)
    H = int(meta["original_height"])
    W = int(meta["original_width"])
    heatmap = F.interpolate(heatmap, size=(H, W), mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze(0).squeeze(0)
    if normalize:
        max_val = heatmap.max().clamp_min(1e-6)
        heatmap = heatmap / max_val
    return heatmap


def save_heatmap_overlay(
    heatmap: torch.Tensor,
    image_path: Path,
    output_path: Path,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> None:
    """원본 이미지를 읽어 히트맵을 합성한 뒤 저장."""
    heatmap_np = heatmap.detach().cpu().numpy()
    heatmap_uint8 = np.uint8(255 * heatmap_np)
    color = cv2.applyColorMap(heatmap_uint8, getattr(cv2, f"COLORMAP_{colormap.upper()}"))
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    image = np.array(Image.open(image_path).convert("RGB"))
    overlay = (alpha * color + (1 - alpha) * image).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)


class GenericAttentionBaseline:
    """Sim-Lingo GAE-style relevancy generator built on saved inference payloads."""

    def __init__(
        self,
        layer_filter: str = "vision_block",
        start_layer: int = 0,
        residual_alpha: float = 0.5,
        cls_index: int = 0,
    ) -> None:
        self.layer_filter = layer_filter
        self.start_layer = start_layer
        self.residual_alpha = residual_alpha
        self.cls_index = cls_index

    def compute_heatmap_from_payload(
        self,
        payload: Dict[str, Any],
        output_dir: Path,
        image_token_slice: Optional[Tuple[int, int]] = None,
        suffix: str = "gae",
    ) -> Path:
        """단일 추론 결과(.pt) dict에서 히트맵을 계산하고 저장."""
        attention = payload["attention"]
        meta = payload["meta"]
        token_relevance = build_token_relevance(
            attention,
            output_token_index=self.cls_index,
            layer_filter=self.layer_filter,
            start_layer=self.start_layer,
            residual_alpha=self.residual_alpha,
        )
        if image_token_slice is None:
            image_scores = extract_image_token_scores(token_relevance, meta, cls_has_index_zero=True)
        else:
            start, end = image_token_slice
            image_scores = token_relevance[start:end]
        heatmap = tokens_to_heatmap(image_scores, meta)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(payload["image_path"])
        output_path = output_dir / f"{image_path.stem}_{suffix}.png"
        save_heatmap_overlay(heatmap, image_path, output_path)
        return output_path
