#!/usr/bin/env python3
"""오프라인 실험을 위한 경량 Sim-Lingo 추론 베이스라인.

이 스크립트는 고정된 장면 디렉토리를 입력으로 받아 사전 학습된 Sim-Lingo InternVL2
모델을 불러오고, 추론 과정에서 등장하는 모든 어텐션 헤드의 어텐션 맵을 저장하는
실험 골격을 제공합니다. 아직 완성된 배치 파이프라인은 아니지만, 이후
Vision-Language-Action 설명 실험을 빠르게 확장할 수 있도록 구성했습니다.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import types
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
import importlib.util

# 어떤 위치에서 실행해도 external/simlingo 모듈을 불러올 수 있도록 경로 추가
REPO_ROOT = Path(__file__).resolve().parents[1]
SIMLINGO_SRC = REPO_ROOT / "external" / "simlingo"
if SIMLINGO_SRC.exists() and str(SIMLINGO_SRC) not in sys.path:
    sys.path.insert(0, str(SIMLINGO_SRC))
# Use locally patched InternVL2 module cache (copied to experiment/InternVL2-1B)
LOCAL_TFMM_ROOT = REPO_ROOT / "experiment" / "InternVL2-1B" / "transformers_modules"
os.environ["HF_MODULES_CACHE"] = str(LOCAL_TFMM_ROOT.resolve())

# Force-import patched InternVL2 modules so AutoModel uses them instead of downloading
# pre-load package stubs for transformers dynamic imports
def _ensure_pkg(name: str, path: Path):
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg

_ensure_pkg("transformers_modules", LOCAL_TFMM_ROOT)
_ensure_pkg("transformers_modules.OpenGVLab", LOCAL_TFMM_ROOT / "OpenGVLab")

# [CRITICAL] Find the hash directory containing modeling_internvl_chat.py
# We MUST use the hashed directory because AutoModel.from_pretrained with trust_remote_code=True
# expects the specific hash present in the config or downloads it.
# To force usage of our local custom code, we map the package to this hash dir.
internvl_root = LOCAL_TFMM_ROOT / "OpenGVLab" / "InternVL2-1B"
hash_dir = None
if internvl_root.exists():
    for child in internvl_root.iterdir():
        if child.is_dir() and (child / "modeling_internvl_chat.py").exists():
            hash_dir = child
            break

if hash_dir:
    print(f"[SimLingo] Found custom module hash dir: {hash_dir.name}")
    _ensure_pkg("transformers_modules.OpenGVLab.InternVL2-1B", hash_dir)
    # ALSO register the hashed package name to resolve relative imports within the module
    _ensure_pkg(f"transformers_modules.OpenGVLab.InternVL2-1B.{hash_dir.name}", hash_dir)
else:
    print("[SimLingo] Warning: Could not find hash dir with modeling_internvl_chat.py, using root")
    _ensure_pkg("transformers_modules.OpenGVLab.InternVL2-1B", internvl_root)

# Patch Qwen2Attention forward to stash attn weights for all layers/heads
_orig_qwen2_attn_forward = Qwen2Attention.forward


def _patched_qwen2_attn_forward(self, *args, **kwargs):
    kwargs["output_attentions"] = True
    result = _orig_qwen2_attn_forward(self, *args, **kwargs)
    attn_weights = None
    if isinstance(result, tuple):
        if len(result) >= 3:
            attn_weights = result[1]
        elif len(result) >= 2:
            attn_weights = result[1]
    # attn_weights shape: [batch, heads, tgt_len, src_len]
    self.attn_map = attn_weights
    if attn_weights is not None and torch.is_tensor(attn_weights):
        try:
            attn_weights.retain_grad()
        except Exception:
            pass
        self.attn_map_grad = attn_weights.grad
    else:
        self.attn_map_grad = None
    return result


Qwen2Attention.forward = _patched_qwen2_attn_forward

from simlingo_training.models.driving import DrivingModel
from simlingo_training.models.encoder import internvl2_model as ivl
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
from team_code.simlingo_utils import get_camera_extrinsics, get_camera_intrinsics, project_points
from experiment.simlingo_patches import patch_simlingo

patch_simlingo()

# Ensure InternVL vision encoder forwards with output_attentions=True so vision hooks can capture attn
_orig_extract_feature = getattr(ivl.LingoInternVLModel, "extract_feature", None)


def _patched_extract_feature(self, pixel_values, **kwargs):
    outputs = self.model(
        pixel_values,
        output_attentions=True,
        return_dict=True,
    )
    # keep return shape identical to original (hidden states only)
    # keep return shape identical to original (hidden states only)
    if hasattr(outputs, "last_hidden_state"):
        # Store vision attentions on the model instance for later retrieval
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            # We attach it to the model instance (self.model) so we can access it later
            self.model.vision_attentions = outputs.attentions
            # Also retain grad for all vision attentions
            for attn in outputs.attentions:
                if attn is not None and torch.is_tensor(attn):
                    try:
                        attn.retain_grad()
                    except Exception:
                        pass
        return outputs.last_hidden_state
    # fallback to original behavior if unexpected return
    if _orig_extract_feature is not None:
        return _orig_extract_feature(self, pixel_values, **kwargs)
    return outputs


if _orig_extract_feature is not None:
    ivl.LingoInternVLModel.extract_feature = _patched_extract_feature

# Patch AutoModel.extract_feature at runtime so AutoModel forward is called with output_attentions=True
def _install_extract_feature_patch(auto_model) -> None:
    orig = getattr(auto_model, "extract_feature", None)
    if orig is None:
        return

    def _patched(self, pixel_values, **kwargs):
        # Use original extract_feature to preserve internal kwargs (image_flags etc.).
        # Rely on model.config.output_attentions=True set upstream.
        return orig(pixel_values, **kwargs)

    auto_model.extract_feature = types.MethodType(_patched, auto_model)


def _install_interleaver_logging(driving_model) -> None:
    """Patch InternVLChatModel forward/extract_feature to stash interleaver tensors."""
    # Target the VLM used for vision encoding (InternVLChatModel)
    # It is located at driving_model.vision_model.image_encoder.model
    lm = None
    if hasattr(driving_model, "vision_model") and hasattr(driving_model.vision_model, "image_encoder"):
        lm = getattr(driving_model.vision_model.image_encoder, "model", None)
    
    if lm is None:
        # Fallback to language model if vision path fails (though unlikely for SimLingo)
        lm = getattr(driving_model, "language_model", None)

    if lm is None:
        return
        
    core = lm
    if getattr(core, "_interleaver_logging_installed", False):
        return

    def _patch_extract_feature(self, pixel_values):
        """Copy of InternVLChatModel.extract_feature with logging of intermediate tensors."""
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.select_layer]
        # drop CLS
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        pixel_shuffled = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        mlp_input = pixel_shuffled.reshape(pixel_shuffled.shape[0], -1, pixel_shuffled.shape[-1])
        mlp_output = self.mlp1(mlp_input)

        # Retain grads on intermediates for interleaver relevance
        for t in (pixel_shuffled, mlp_input, mlp_output):
            if hasattr(t, "retain_grad"):
                try:
                    t.retain_grad()
                except Exception:
                    pass

        # Derive patch count per image (assumes uniform num_image_token per patch)
        # Derive patch count per image (assumes uniform num_image_token per patch)
        num_tokens = mlp_input.shape[1]
        # Use actual output shape as tokens_per_patch
        tokens_per_patch = num_tokens
        num_patches_total = None
        if tokens_per_patch and tokens_per_patch > 0:
            num_patches_total = int(num_tokens // tokens_per_patch)
        num_patches_list = [num_patches_total] if num_patches_total is not None else None

        cache = getattr(self, "_interleaver_cache", {}) or {}
        cache.update(
            {
                "pixel_shuffle_out": pixel_shuffled,
                "mlp1_input": mlp_input,
                "mlp1_output": mlp_output,
                "num_patches_list": num_patches_list,
                "tokens_per_patch": tokens_per_patch,
                "mlp1_weight": self.mlp1[1].weight.detach().to("cpu") if hasattr(self, "mlp1") else None,
                "mlp1_bias": self.mlp1[1].bias.detach().to("cpu") if hasattr(self, "mlp1") else None,
            }
        )
        self._interleaver_cache = cache
        return mlp_output

    def _patch_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        lm_core = getattr(self, "language_model", None)
        if lm_core is None:
            lm_core = getattr(self, "model", self)

        # If inputs_embeds are provided (e.g., downstream calls), bypass interleaver logging
        if inputs_embeds is not None or pixel_values is None:
            outputs = lm_core(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            image_flags = image_flags.squeeze(-1) if image_flags is not None else None
            input_embeds = lm_core.get_input_embeddings()(input_ids).clone()

            vit_embeds = self.extract_feature(pixel_values)
            vit_embeds = vit_embeds[image_flags == 1] if image_flags is not None else vit_embeds
            vit_batch_size = pixel_values.shape[0]

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                    f"vit_embeds.shape={vit_embeds.shape}"
                )
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

            input_embeds = input_embeds.reshape(B, N, C)

            cache = getattr(self, "_interleaver_cache", {}) or {}
            cache.update(
                {
                    "selected_mask": selected.reshape(B, N),
                    "vit_embeds": vit_embeds,
                    "vit_batch_size": vit_batch_size,
                    "input_shape": (B, N, C),
                }
            )
            self._interleaver_cache = cache

            outputs = lm_core(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        # Use the attentions returned by the model directly
        if outputs.attentions is not None:
            # Stash for external retrieval (e.g. by SimLingoInferenceBaseline)
            self.all_attentions = outputs.attentions
            for attn in outputs.attentions:
                if attn is not None and torch.is_tensor(attn):
                    try:
                        attn.retain_grad()
                    except Exception:
                        pass
        else:
            self.all_attentions = None

        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    core.extract_feature = types.MethodType(_patch_extract_feature, core)
    core.forward = types.MethodType(_patch_forward, core)
    core._interleaver_logging_installed = True

DEFAULT_CONFIG_PATH = Path("checkpoints/simlingo/simlingo/.hydra/config.yaml")  # 기본 Hydra config
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt")  # 기본 ckpt
DEFAULT_SCENE_DIR = Path("data/sample/01")  # 기본 입력 시나리오 디렉토리(하위 frames/speed 사용)
DEFAULT_OUTPUT_DIR = Path("experiment_outputs/simlingo_inference")  # 기본 출력 디렉토리
DEFAULT_EXPLAIN_MODE = os.environ.get("SIMLINGO_EXPLAIN_MODE", "action")  # action/text 모드 기본값
DEFAULT_KINEMATIC_METRIC = "curv_energy"  # 운동학 함수 기본값
DEFAULT_IMAGE_SIZE = 224  # 입력 리사이즈 (기본 448 : 메모리 절감 필요 시 336 / 228 - (우선순위 1))
DEFAULT_MAX_PATCHES = 2  # dynamic_preprocess max_num (기본 2 : 메모리 절감 필요 시 1 - (우선순위 2))
DEFAULT_FRAMES_SUBDIR = "video_garmin"
DEFAULT_SPEED_SUBDIR = "video_garmin_speed"
DEFAULT_USE_SPLINE = True
DEFAULT_SPLINE_SMOOTHING = 0.0  # 2차 차분(곡률) 페널티 강도 λ
DEFAULT_SPLINE_NUM_SAMPLES = 0  # 0이면 입력 포인트 개수 유지

EPS = 1e-6
DELTA_S = 1.0
DELTA_T = 0.25


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x.float()


def _build_second_diff_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """2차 차분 행렬 D (shape: (n-2, n))을 생성합니다."""
    if n < 3:
        return torch.zeros((0, n), device=device, dtype=dtype)
    main = torch.full((n - 2,), -2.0, device=device, dtype=dtype)
    D = torch.zeros((n - 2, n), device=device, dtype=dtype)
    row_idx = torch.arange(n - 2, device=device)
    D[row_idx, row_idx] = 1.0
    D[row_idx, row_idx + 1] = main
    D[row_idx, row_idx + 2] = 1.0
    # off-diagonals already placed via indexing; 'off' variable unused but kept for clarity
    return D


def _parametric_cubic_smoothing_resample(
    points: torch.Tensor,
    smoothing: float = 0.0,
    num_samples: Optional[int] = None,
    param: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """입력 포인트(배치 가능)를 매끄럽게 한 뒤 파라메트릭 cubic Hermite spline으로 리샘플합니다.

    - smoothing: 0이면 원본 유지, 양수이면 2차 차분 페널티를 준 평활화(Whittaker 형태).
    - num_samples: 0/None이면 입력 포인트 개수를 유지합니다.
    - param: 길이 N의 1D (또는 B×N) 파라미터 축을 직접 지정할 수 있습니다. None이면
      호길이 기반 정규화 파라미터를 사용합니다.
    모든 연산을 torch에서 수행해 그래디언트가 끊기지 않습니다.
    """
    points = _ensure_batch(points)
    orig_dtype = points.dtype
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    points = points.to(work_dtype)
    B, N, _ = points.shape
    if N < 2:
        return points
    device = points.device
    dtype = points.dtype
    target_samples = num_samples if num_samples and num_samples > 0 else N

    # 파라메터 t: 외부 파라미터(시간 등) 또는 호 길이 정규화 [0, 1]
    if param is not None:
        param = param.to(device=device, dtype=dtype)
        if param.dim() == 1:
            param = param.unsqueeze(0).expand(B, -1)
        t = param - param[:, :1]
    else:
        diffs = points[:, 1:, :] - points[:, :-1, :]
        seglen = diffs.norm(dim=-1)
        t = torch.zeros((B, N), device=device, dtype=dtype)
        t[:, 1:] = torch.cumsum(seglen, dim=1)
    total = t[:, -1:].clamp_min(EPS)
    t = t / total

    smoothed = points
    if smoothing > 0.0 and N >= 3:
        D = _build_second_diff_matrix(N, device=device, dtype=dtype)
        penalty = D.transpose(0, 1) @ D  # (N, N)
        A = torch.eye(N, device=device, dtype=dtype) + smoothing * penalty
        # broadcast solve over batch (A: (N,N) -> (B,N,N))
        A_batch = A.unsqueeze(0).expand(B, -1, -1)
        smoothed = torch.linalg.solve(A_batch, points)

    # 접선 벡터 (Catmull-Rom 스타일)
    tangents = torch.zeros_like(smoothed)
    dt_forward = (t[:, 2:] - t[:, :-2]).clamp_min(EPS)
    tangents[:, 1:-1, :] = (smoothed[:, 2:, :] - smoothed[:, :-2, :]) / dt_forward.unsqueeze(-1)
    tangents[:, 0, :] = (smoothed[:, 1, :] - smoothed[:, 0, :]) / (t[:, 1] - t[:, 0]).clamp_min(EPS).unsqueeze(-1)
    tangents[:, -1, :] = (smoothed[:, -1, :] - smoothed[:, -2, :]) / (t[:, -1] - t[:, -2]).clamp_min(EPS).unsqueeze(-1)

    # 타깃 샘플 위치
    t_query = torch.linspace(0, 1, target_samples, device=device, dtype=dtype)
    t_query = t_query.unsqueeze(0).expand(B, -1)  # (B, M)

    # 각 샘플이 속하는 구간 인덱스 찾기
    idx = torch.searchsorted(t, t_query, right=True)
    idx = idx.clamp(1, N - 1)
    idx_prev = idx - 1

    def _gather(batch_tensor: torch.Tensor, gather_idx: torch.Tensor) -> torch.Tensor:
        gather_idx_expanded = gather_idx.unsqueeze(-1).expand(-1, -1, batch_tensor.size(-1))
        return torch.gather(batch_tensor, 1, gather_idx_expanded)

    t0 = torch.gather(t, 1, idx_prev)
    t1 = torch.gather(t, 1, idx)
    p0 = _gather(smoothed, idx_prev)
    p1 = _gather(smoothed, idx)
    m0 = _gather(tangents, idx_prev)
    m1 = _gather(tangents, idx)

    h = (t1 - t0).clamp_min(EPS).unsqueeze(-1)  # (B, M, 1)
    u = ((t_query - t0) / (t1 - t0).clamp_min(EPS)).unsqueeze(-1)  # (B, M, 1)

    u2 = u * u
    u3 = u2 * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2

    resampled = (
        h00 * p0
        + h10 * h * m0
        + h01 * p1
        + h11 * h * m1
    )
    return resampled.to(dtype=orig_dtype)


# pred_route -> 곡률 제곱합 (조향 강도)
def compute_curvature_energy(route: torch.Tensor, delta_s: Optional[float] = None) -> torch.Tensor:
    route = _ensure_batch(route)
    diffs = route[:, 1:, :] - route[:, :-1, :]
    seglen = diffs.norm(dim=-1).clamp_min(EPS)
    tangent = diffs / seglen.unsqueeze(-1)
    headings = torch.atan2(tangent[..., 1], tangent[..., 0])
    heading_diff = headings[:, 1:] - headings[:, :-1]
    if delta_s is not None:
        curvature = heading_diff / delta_s
    else:
        ds = (seglen[:, 1:] + seglen[:, :-1]) * 0.5
        curvature = heading_diff / ds.clamp_min(EPS)
    return (curvature ** 2).sum()


# pred_route -> 곡률 변화 제곱합 (조향 부드러움/jerk)
def compute_curvature_diff(route: torch.Tensor, delta_s: Optional[float] = None) -> torch.Tensor:
    route = _ensure_batch(route)
    diffs = route[:, 1:, :] - route[:, :-1, :]
    seglen = diffs.norm(dim=-1).clamp_min(EPS)
    tangent = diffs / seglen.unsqueeze(-1)
    headings = torch.atan2(tangent[..., 1], tangent[..., 0])
    if delta_s is not None:
        curvature = (headings[:, 1:] - headings[:, :-1]) / delta_s
    else:
        ds = (seglen[:, 1:] + seglen[:, :-1]) * 0.5
        curvature = (headings[:, 1:] - headings[:, :-1]) / ds.clamp_min(EPS)
    curvature_diff = curvature[:, 1:] - curvature[:, :-1]
    return (curvature_diff ** 2).sum()


# pred_speed -> 전진 거리 (종방향 진행량)
def compute_longitudinal_progress(speed_wps: torch.Tensor) -> torch.Tensor:
    speed_wps = _ensure_batch(speed_wps)
    diffs = speed_wps[:, 1:, :] - speed_wps[:, :-1, :]
    base_dir = diffs[:, :1, :]
    norm = base_dir.norm(dim=-1, keepdim=True).clamp_min(EPS)
    direction = base_dir / norm
    proj = (diffs * direction).sum(dim=-1)
    return torch.relu(proj).sum()


# pred_speed -> 속도/가속도/jerk를 종방향 축으로 투영해 계산 (내부 헬퍼 함수)
def _derive_longitudinal_terms(speed_wps: torch.Tensor, delta_t: float = DELTA_T):
    speed_wps = _ensure_batch(speed_wps)
    diffs = speed_wps[:, 1:, :] - speed_wps[:, :-1, :]
    base_dir = diffs[:, :1, :]
    norm = base_dir.norm(dim=-1, keepdim=True).clamp_min(EPS)
    direction = base_dir / norm
    velocities = (diffs * direction).sum(dim=-1) / max(delta_t, EPS)
    accelerations = (velocities[:, 1:] - velocities[:, :-1]) / max(delta_t, EPS)
    jerks = (accelerations[:, 1:] - accelerations[:, :-1]) / max(delta_t, EPS)
    return velocities, accelerations, jerks


# pred_speed -> 평균 전진 속도 (양수 성분)
def compute_forward_speed(speed_wps: torch.Tensor, delta_t: float = DELTA_T) -> torch.Tensor:
    velocities, _, _ = _derive_longitudinal_terms(speed_wps, delta_t=delta_t)
    return torch.relu(velocities).mean()


# pred_speed -> 가속도 에너지 (승차감/동적 강도)
def compute_acceleration_energy(speed_wps: torch.Tensor, delta_t: float = DELTA_T) -> torch.Tensor:
    _, accelerations, _ = _derive_longitudinal_terms(speed_wps, delta_t=delta_t)
    return (accelerations ** 2).sum()


# pred_speed -> 제동 에너지 (감속 위험)
def compute_brake_energy(speed_wps: torch.Tensor, delta_t: float = DELTA_T) -> torch.Tensor:
    _, accelerations, _ = _derive_longitudinal_terms(speed_wps, delta_t=delta_t)
    braking = torch.relu(-accelerations)
    return (braking ** 2).sum()


# pred_speed -> jerk 에너지 (가속도 변화량)
def compute_jerk_energy(speed_wps: torch.Tensor, delta_t: float = DELTA_T) -> torch.Tensor:
    _, _, jerks = _derive_longitudinal_terms(speed_wps, delta_t=delta_t)
    return (jerks ** 2).sum()


# kinematic_metric 이름 -> 사용 토큰(source)/함수/설명 매핑
KINEMATIC_METRICS = {
    "curv_energy": {"source": "route", "fn": compute_curvature_energy, "description": "곡률 제곱합"},
    "curv_diff": {
        "source": "route",
        "fn": compute_curvature_diff,
        "description": "곡률 변화 제곱합",
    },
    "longitudinal_progress": {
        "source": "speed",
        "fn": compute_longitudinal_progress,
        "description": "종방향 전진 거리",
    },
    "forward_speed": {
        "source": "speed",
        "fn": compute_forward_speed,
        "description": "평균 전진 속도",
    },
    "acc_energy": {
        "source": "speed",
        "fn": compute_acceleration_energy,
        "description": "종방향 가속도 에너지",
    },
    "brake_energy": {
        "source": "speed",
        "fn": compute_brake_energy,
        "description": "제동(감속) 에너지",
    },
    "jerk_energy": {
        "source": "speed",
        "fn": compute_jerk_energy,
        "description": "종방향 jerk 에너지",
    },
    "none": {"source": None, "fn": None, "description": "절댓값 합을 사용하는 예비 설정"},
}
TEXT_TOKEN_STRATEGIES = ("max", "last", "index")

class AttentionRecorder:
    """ViT/LLaMA 블록의 어텐션 출력을 훅으로 수집하고 가중치·그래디언트를 저장합니다."""

    def __init__(self, store_on_cpu: bool = True, keep_last_only: bool = False) -> None:
        self.store_on_cpu = store_on_cpu
        self.keep_last_only = keep_last_only
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.records: List[Dict[str, Any]] = []
        self._active_buffers: Optional[dict] = None
        self._current_tag: Optional[str] = None

    def register_module(self, module: torch.nn.Module, name: str, record_grad: bool = True) -> None:
        def hook(mod, __, output):
            if self._active_buffers is None:
                return
            grad_tensor = None
            # 먼저 모듈에 이미 저장된 attn_map을 확인 (커스텀 패치에서 설정)
            attn = getattr(mod, "attn_map", None)
            grad_tensor = getattr(mod, "attn_map_grad", None) if hasattr(mod, "attn_map_grad") else None
            if attn is None:
                attn = self._extract_attention(output)
            if attn is None:
                return
            if record_grad:
                attn.retain_grad()
                grad_tensor = attn.grad if attn.grad is not None else grad_tensor
                if self.keep_last_only:
                    self._active_buffers[name] = (attn, grad_tensor)
                else:
                    self._active_buffers.setdefault(name, []).append((attn, grad_tensor))
            else:
                if self.keep_last_only:
                    self._active_buffers[name] = (attn.detach(), grad_tensor)
                else:
                    self._active_buffers.setdefault(name, []).append((attn.detach(), grad_tensor))

        handle = module.register_forward_hook(hook)
        self.handles.append(handle)

    @staticmethod
    def _extract_attention(output) -> Optional[torch.Tensor]:
        """Recursively pull the first available 4D attention tensor or stack of them."""

        def _collect(obj):
            if torch.is_tensor(obj) and obj.dim() == 4:
                yield obj
                return
            if hasattr(obj, "attentions"):
                yield from _collect(obj.attentions)
            if isinstance(obj, Mapping):
                for val in obj.values():
                    yield from _collect(val)
            elif isinstance(obj, (list, tuple)):
                for val in obj:
                    yield from _collect(val)

        tensors = list(_collect(output))
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]
        try:
            return torch.stack(tensors, dim=0)  # [n, B, H, S, S] if shapes match
        except Exception:
            return tensors[0]

    def start_recording(self, tag: str) -> None:
        self._current_tag = tag
        self._active_buffers = defaultdict(list) if not self.keep_last_only else {}

    def stop_recording(self) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        assert self._active_buffers is not None, "Recording was not started."
        aggregated: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        for name, tensors in self._active_buffers.items():
            tensor_list = [tensors] if self.keep_last_only else tensors
            aggregated[name] = []
            for pair in tensor_list:
                if isinstance(pair, tuple):
                    tensor, grad_tensor = pair
                else:
                    tensor, grad_tensor = pair, None
                attn_tensor = tensor.detach()
                if grad_tensor is None and tensor.grad is not None:
                    grad_tensor = tensor.grad
                grad_tensor = grad_tensor.detach() if grad_tensor is not None else None
                if self.store_on_cpu:
                    attn_tensor = attn_tensor.to("cpu")
                    if grad_tensor is not None:
                        grad_tensor = grad_tensor.to("cpu")
                aggregated[name].append(
                    {
                        "attn": attn_tensor,
                        "grad": grad_tensor,
                        "shape": tuple(tensor.shape),
                    }
                )
                tensor.grad = None
        self.records.append({"tag": self._current_tag, "maps": aggregated})
        if not aggregated and os.environ.get("ATTN_DEBUG"):
            print(f"[AttentionRecorder] No attentions captured for tag={self._current_tag}")
        self._active_buffers = None
        self._current_tag = None
        return aggregated

    def dump(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for entry in self.records:
            tag = entry["tag"]
            torch.save(entry["maps"], output_dir / f"{tag}_attn.pt")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class SimLingoInferenceBaseline:
    """Sim-Lingo 모델을 오프라인으로 실행하며 모든 어텐션 맵과 추론 결과를 수집합니다."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: Optional[str] = None,
        target_mode: str = "auto",
        explain_mode: str = DEFAULT_EXPLAIN_MODE,
        text_token_strategy: str = "max",
        text_token_index: int = -1,
        kinematic_metric: str = DEFAULT_KINEMATIC_METRIC,
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_patches: int = DEFAULT_MAX_PATCHES,
        frames_subdir: str = DEFAULT_FRAMES_SUBDIR,
        speed_subdir: str = DEFAULT_SPEED_SUBDIR,
        use_spline: bool = DEFAULT_USE_SPLINE,
        spline_smoothing: float = DEFAULT_SPLINE_SMOOTHING,
        spline_num_samples: int = DEFAULT_SPLINE_NUM_SAMPLES,
        enable_vision_hooks: Optional[bool] = None,
        enable_language_hooks: Optional[bool] = None,
        skip_backward: bool = False,
    ) -> None:
        config_path = config_path or DEFAULT_CONFIG_PATH
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at {self.config_path}")
        checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_mode = target_mode
        self.explain_mode = explain_mode.lower()
        if self.explain_mode not in {"action", "text"}:
            raise ValueError(f"Unsupported explain mode: {explain_mode}")
        self.text_token_strategy = text_token_strategy
        if self.text_token_strategy not in TEXT_TOKEN_STRATEGIES:
            raise ValueError(f"text_token_strategy must be one of {TEXT_TOKEN_STRATEGIES}")
        self.text_token_index = text_token_index
        kinematic_metric = kinematic_metric or DEFAULT_KINEMATIC_METRIC
        if kinematic_metric not in KINEMATIC_METRICS:
            raise ValueError(f"Unsupported kinematic metric: {kinematic_metric}")
        self.kinematic_metric = kinematic_metric
        self.image_size = image_size
        self.max_patches = max_patches
        self.frames_subdir = frames_subdir
        self.speed_subdir = speed_subdir
        self.use_spline = use_spline
        self.spline_smoothing = max(0.0, float(spline_smoothing))
        self.spline_num_samples = spline_num_samples if spline_num_samples and spline_num_samples > 0 else None
        
        self.debug_info = {}  # Store debug info for .pt file
        self.debug = False # Initialize debug flag

        # ViT/LLM 모두 어텐션을 수집 (기본 켜짐)
        if enable_vision_hooks is None:
            enable_vision_hooks = True
        if enable_language_hooks is None:
            enable_language_hooks = True
        self.enable_vision_hooks = enable_vision_hooks
        self.enable_language_hooks = enable_language_hooks
        self.skip_backward = skip_backward
        self.cfg = OmegaConf.load(self.config_path)
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model.vision_model.variant, trust_remote_code=True
        )
        if hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<WAYPOINTS>",
                    "<WAYPOINTS_DIFF>",
                    "<ORG_WAYPOINTS_DIFF>",
                    "<ORG_WAYPOINTS>",
                    "<WAYPOINT_LAST>",
                    "<ROUTE>",
                    "<ROUTE_DIFF>",
                    "<TARGET_POINT>",
                ]
            }
        )
        self.tokenizer.padding_side = "left"
        self.transform = build_transform(input_size=self.image_size)
        self.T = 1
        self.num_image_token = self._compute_num_image_tokens(
            self.cfg.model.vision_model.variant, image_size_override=self.image_size
        )
        self.model = self._build_model()
        # 최종 forward의 어텐션/grad만 유지 (레이어/헤드별)
        self.recorder = AttentionRecorder(keep_last_only=True)
        self._register_attention_hooks()
        self._speed_cache: Dict[str, Dict[str, float]] = {}

    def _compute_num_image_tokens(self, encoder_variant: str, image_size_override: Optional[int] = None) -> int:
        """InternVL2 비전 인코더 설정에서 이미지 토큰(패치 수)을 계산합니다."""
        cfg = AutoConfig.from_pretrained(encoder_variant, trust_remote_code=True)
        image_size = image_size_override or cfg.force_image_size or cfg.vision_config.image_size
        patch_size = cfg.vision_config.patch_size
        downsample = getattr(cfg, "downsample_ratio", getattr(cfg.vision_config, "downsample_ratio", 1.0))
        return int((image_size // patch_size) ** 2 * (downsample ** 2))

    def _build_model(self) -> DrivingModel:
        """Hydra 설정으로 DrivingModel을 만들고 체크포인트를 불러온 뒤 어텐션 출력을 활성화합니다."""
        cache_dir = Path("pretrained") / self.cfg.model.vision_model.variant.split("/")[1]
        cache_dir = Path(to_absolute_path(str(cache_dir)))
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        model = hydra.utils.instantiate(
            self.cfg.model,
            cfg_data_module=self.cfg.data_module,
            processor=self.processor,
            cache_dir=str(cache_dir),
            _recursive_=False,
        ).to(self.device)
        torch.set_default_dtype(default_dtype)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        # 어텐션 텐서를 추출할 수 있도록 설정 플래그 활성화
        model.language_model.model.config.output_attentions = True
        model.language_model.model.config.output_hidden_states = True
        model.vision_model.image_encoder.model.config.output_attentions = True
        # patch extract_feature on AutoModel so forward runs with output_attentions=True
        _install_extract_feature_patch(model.vision_model.image_encoder.model)
        # install interleaver logging on language model
        _install_interleaver_logging(model)
        return model

    def _register_attention_hooks(self) -> None:
        """비전/언어 트랜스포머 블록 전체에 레코더 훅을 연결합니다."""
        # 비전 인코더 블록 훅 등록 (어텐션만, grad는 비활성화해 메모리 절약)
        if self.enable_vision_hooks:
            vision_container = getattr(self.model.vision_model, "image_encoder", None)
            vision_model = getattr(vision_container, "model", None)
            if vision_model is not None:
                # top-level AutoModel (captures BaseModelOutput.attentions if provided)
                self.recorder.register_module(vision_model, "vision_model_top", record_grad=False)
                # inner vision encoder (e.g., vision_model.encoder.layers)
                core = getattr(vision_model, "vision_model", vision_model)
                if hasattr(core, "encoder"):
                    for idx, block in enumerate(core.encoder.layers):
                        self.recorder.register_module(block, f"vision_block_{idx}", record_grad=False)
                # attention submodules directly (to force per-layer attn capture)
                for sub_name, sub_module in vision_model.named_modules():
                    cls = sub_module.__class__.__name__.lower()
                    if "attention" in cls or "attn" in sub_name.lower():
                        safe_name = sub_name.replace(".", "_")
                        self.recorder.register_module(sub_module, f"vision_attn_{safe_name}", record_grad=False)
        # 언어 모델 블록 훅 등록
        if self.enable_language_hooks:
            lm = self.model.language_model.model
            
            # Unwrap PeftModel if present
            # PeftModel -> base_model -> model (usually)
            if "PeftModel" in str(type(lm)):
                 # print(f"[DEBUG] Unwrapping PeftModel: {type(lm)}")
                 if hasattr(lm, "base_model") and hasattr(lm.base_model, "model"):
                     lm = lm.base_model.model
                 elif hasattr(lm, "base_model"):
                     lm = lm.base_model
            
            # top-level CausalLM/LLM module (captures BaseModelOutput.attentions if provided)
            self.recorder.register_module(lm, "language_model_top", record_grad=True)
            
            layers = []
            if hasattr(lm, "layers"):
                layers = lm.layers
            elif hasattr(lm, "model") and hasattr(lm.model, "layers"):
                layers = lm.model.layers
            elif hasattr(lm, "h"): # Common in some models
                layers = lm.h
            elif hasattr(lm, "block"): # Common in others
                layers = lm.block
            
            # print(f"[DEBUG] Found {len(layers)} LLM layers for hooking")
            self.debug_info["llm_layers_found_for_hooks"] = len(layers)
            self.debug_info["llm_model_type"] = str(type(lm))
            
            for idx, block in enumerate(layers):
                self.recorder.register_module(block, f"language_block_{idx}", record_grad=True)
                # register inner attention modules explicitly
                for sub_name, sub_module in block.named_modules():
                    cls = sub_module.__class__.__name__.lower()
                    if "attention" in cls or "attn" in sub_name.lower():
                        safe_name = f"language_block_{idx}_" + sub_name.replace(".", "_")
                        self.recorder.register_module(sub_module, safe_name, record_grad=True)

    def _build_scenario_name(self, scenario_dir: Path) -> str:
        """시나리오 경로에서 정렬/원본, city 번호, 시나리오 번호를 추출해 접두어를 만듭니다."""
        scenario_dir = scenario_dir.resolve()
        scenario_id = scenario_dir.name
        city_id = scenario_dir.parent.name if scenario_dir.parent else "unknown"
        index_type = scenario_dir.parent.parent.name if scenario_dir.parent and scenario_dir.parent.parent else "scene"
        return f"{index_type}_{city_id}_{scenario_id}"

    def _prepare_output_subdir(self, output_root: Path, scenario_dir: Path, mode_suffix: str) -> Path:
        """정규화된 시나리오 이름과 모드, 타겟 설정, 타임스탬프 규칙으로 하위 디렉토리를 만듭니다."""
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        scenario_name = self._build_scenario_name(scenario_dir)
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        detail = self.kinematic_metric if mode_suffix == "action" else self.text_token_strategy
        base_name = f"{scenario_name}_{mode_suffix}_{detail}_{timestamp}"
        candidate = output_root / base_name
        counter = 1
        while candidate.exists():
            candidate = output_root / f"{base_name}_{counter}"
            counter += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _prepare_scenario_subdirs(self, scenario_output_dir: Path) -> Dict[str, Path]:
        """시나리오 단위 하위 디렉토리(파일 유형별)를 모두 생성합니다."""
        subdirs = {
            "pt": scenario_output_dir / "pt",
            "route_overlay": scenario_output_dir / "route_overlay",
            "speed_overlay": scenario_output_dir / "speed_overlay",
            "text_output": scenario_output_dir / "text_output",
            "pred_route": scenario_output_dir / "pred_route",
            "pred_speed_wps": scenario_output_dir / "pred_speed_wps",
            "input_images": scenario_output_dir / "input_images",
        }
        for path in subdirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return subdirs

    def _project_and_draw_points(
        self,
        points: Optional[torch.Tensor],
        image_hw: Tuple[int, int],
        color: Tuple[int, int, int, int],
        radius: int,
        output_path: Path,
    ) -> None:
        """예측 궤적을 원본 해상도에 맞춰 투영한 뒤 투명 배경 PNG로 저장합니다."""
        if points is None:
            return
        points = points.detach().to("cpu")
        if points.numel() == 0:
            return
        if points.dim() == 3:
            points = points.squeeze(0)
        if points.dim() != 2 or points.shape[1] != 2:
            return
        points = points.float()
        H, W = image_hw
        K = get_camera_intrinsics(W, H, 110)
        if torch.is_tensor(K):
            K_np = K.detach().cpu().numpy()
        else:
            K_np = np.asarray(K)
        try:
            projected = project_points(points.numpy(), K_np)
        except Exception:
            return
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for coord in projected:
            x, y = float(coord[0]), float(coord[1])
            if not np.isfinite([x, y]).all():
                continue
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                draw.ellipse((xi - radius, yi - radius, xi + radius, yi + radius), fill=color)
        overlay.save(output_path)

    @staticmethod
    def _write_tensor_txt(tensor: Optional[torch.Tensor], output_path: Path) -> None:
        """텐서를 CPU로 옮겨 txt로 저장합니다."""
        if tensor is None:
            return
        tensor_cpu = tensor.detach().to("cpu")
        if tensor_cpu.dim() == 3:
            tensor_cpu = tensor_cpu.squeeze(0)
        data = tensor_cpu.tolist()
        lines: List[str] = []
        for row in data:
            if isinstance(row, (list, tuple)):
                lines.append(" ".join(f"{float(x):.6f}" for x in row))
            else:
                lines.append(f"{float(row):.6f}")
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_text_outputs(
        self,
        output_path: Path,
        text_features: Optional[Dict[str, Any]],
        language_fallback: Optional[Any] = None,
    ) -> None:
        """생성 텍스트를 txt로 저장합니다. 기본은 모델이 생성한 문장, 없으면 fallback."""
        decoded = ""
        tokens: List[str] = []
        if text_features is not None:
            decoded = text_features.get("decoded_text") or ""
            tokens = text_features.get("token_strings") or []
        if not decoded and language_fallback:
            if isinstance(language_fallback, (list, tuple)):
                decoded = " ".join(str(x) for x in language_fallback if x)
            else:
                decoded = str(language_fallback)
        text_lines = []
        if decoded:
            text_lines.append(decoded)
        if tokens:
            text_lines.append("")
            text_lines.append("Tokens: " + " ".join(tokens))
        output_path.write_text("\n".join(text_lines), encoding="utf-8")

    def _load_speed_table(self, speed_dir: Path) -> Dict[str, float]:
        """speed 텍스트 디렉토리에서 stem->m/s 매핑을 로드합니다."""
        key = str(speed_dir.resolve())
        if key in self._speed_cache:
            return self._speed_cache[key]
        table: Dict[str, float] = {}
        for txt in sorted(speed_dir.glob("*.txt")):
            try:
                val = float(txt.read_text().strip().split()[0])
                table[txt.stem] = val
            except Exception:
                continue
        self._speed_cache[key] = table
        return table

    def run_scene(self, scene_dir: Path, output_dir: Path) -> None:
        """시나리오 디렉토리(하위 images/)의 모든 이미지를 순회하며 추론 결과를 저장합니다."""
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        # frames 디렉토리 우선 사용, 없으면 images 폴백
        candidate_dirs = []
        if self.frames_subdir:
            candidate_dirs.append(scene_dir / self.frames_subdir)
        candidate_dirs.append(scene_dir / "images")
        images_dir = None
        for cdir in candidate_dirs:
            if cdir.exists():
                images_dir = cdir
                break
        if images_dir is None:
            raise FileNotFoundError(f"No frames directory found under {scene_dir} (tried {candidate_dirs})")
        speed_dir = scene_dir / self.speed_subdir if self.speed_subdir else None
        speed_table: Dict[str, float] = {}
        if speed_dir and speed_dir.exists():
            speed_table = self._load_speed_table(speed_dir)
        mode_suffix = "action" if self.explain_mode == "action" else "text"
        scenario_output_dir = self._prepare_output_subdir(output_dir, scene_dir, mode_suffix)
        scenario_subdirs = self._prepare_scenario_subdirs(scenario_output_dir)
        image_paths = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
    def _process_single_frame(
        self,
        image_path: Path,
        speed_table: Dict[str, float],
        scenario_subdirs: Dict[str, Path],
        scenario_output_dir: Path,
    ) -> None:
        record_tag = image_path.stem
        current_speed = speed_table.get(record_tag, 0.0)
        self.recorder.start_recording(record_tag)
        driving_input, meta = self._prepare_driving_input(image_path, current_speed=current_speed)
        outputs = self.model(driving_input)
        pred_speed_wps, pred_route, _ = outputs
        text_features = self._gather_text_features()
        target_scalar, target_meta = self._compute_target_scalar(outputs, text_features)
        if not self.skip_backward:
            if target_scalar is None:
                raise RuntimeError("Target scalar could not be computed for relevance backprop.")
            target_scalar.backward(retain_graph=False)

        attention_maps = self.recorder.stop_recording()
        # 추가: 언어 모델 outputs.attentions를 레이어별로 저장 (grad 포함)
        # DrivingModel이 attentions를 반환하지 않으므로, InternVLChatModel에 stash된 값을 가져옵니다.
        attn_seq = None
        # Try to find stashed attentions in the model hierarchy
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "all_attentions"):
                attn_seq = self.model.language_model.all_attentions
                # print(f"[DEBUG] Retrieved {len(attn_seq) if attn_seq else 0} layers from self.model.language_model.all_attentions")
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "model") and hasattr(self.model.language_model.model, "all_attentions"):
                attn_seq = self.model.language_model.model.all_attentions
                # print(f"[DEBUG] Retrieved {len(attn_seq) if attn_seq else 0} layers from self.model.language_model.model.all_attentions")
        
        # Fallback: check outputs just in case
        if attn_seq is None:
            if hasattr(outputs, "attentions"):
                attn_seq = outputs.attentions
                self.debug_info["attn_source"] = "outputs.attentions"
            elif isinstance(outputs, tuple):
                # Try to find the tuple element that looks like attentions (tuple of tensors)
                for item in outputs:
                    if isinstance(item, (list, tuple)) and len(item) > 0 and torch.is_tensor(item[0]):
                        # Assume this is the attentions tuple (usually length=num_layers)
                        # Qwen2 has 24 layers.
                        if len(item) >= 10: # Heuristic check
                            attn_seq = item
                            self.debug_info["attn_source"] = f"outputs_tuple_item_len_{len(item)}"
                            break
                # If still not found, maybe it's the last element?
                if attn_seq is None and len(outputs) > 0:
                        # Last element is often attentions if output_attentions=True
                        last_item = outputs[-1]
                        if isinstance(last_item, (list, tuple)):
                            attn_seq = last_item
                            self.debug_info["attn_source"] = "outputs_last_item"

            # print(f"[DEBUG] Fallback to outputs search: {len(attn_seq) if attn_seq else 0} layers")
        else:
            self.debug_info["attn_source"] = "stashed_all_attentions"

        self.debug_info["outputs_type"] = str(type(outputs))
        if isinstance(outputs, tuple):
            self.debug_info["outputs_len"] = len(outputs)

        if attn_seq:
            for idx, attn in enumerate(attn_seq):
                if attn is None:
                    continue
                if not torch.is_tensor(attn):
                        # Skip non-tensor elements (e.g. if we accidentally picked up a tuple of strings)
                        continue
                attn_tensor = attn.detach().to("cpu")
                grad_tensor = attn.grad.detach().to("cpu") if attn.grad is not None else None
                
                # Force save to attention_maps with correct key
                entry = {"attn": attn_tensor, "grad": grad_tensor, "shape": tuple(attn.shape)}
                key = f"language_block_{idx}_manual"
                # Overwrite or append? The user wants ALL layers. 
                # If recorder captured something, it might be partial. We trust this explicit capture more.
                attention_maps[key] = [entry]
        
        # 추가: 비전 모델 attentions 저장 (from _patched_extract_feature)
        # We access it via the vision model instance where we attached it
        vision_model_instance = self.model.vision_model.image_encoder.model
        if hasattr(vision_model_instance, "vision_attentions"):
            vision_attentions = vision_model_instance.vision_attentions
            if vision_attentions:
                for idx, attn in enumerate(vision_attentions):
                    if attn is None:
                        continue
                    attn_tensor = attn.detach().to("cpu")
                    grad_tensor = attn.grad.detach().to("cpu") if attn.grad is not None else None
                    # Use a distinct key to avoid collision with hooks if any
                    attention_maps[f"vision_attn_{idx}_output"] = [{"attn": attn_tensor, "grad": grad_tensor, "shape": tuple(attn.shape)}]
            # Clear it to avoid stale data
            vision_model_instance.vision_attentions = None

        # 추가: interleaver 구간 로깅 (mask, mlp1 입출력 등)
        interleaver_info = None
        # Retrieve from the VLM wrapper where we installed the hook
        # driving_model.vision_model.image_encoder.model
        lm_core = None
        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "image_encoder"):
             lm_core = getattr(self.model.vision_model.image_encoder, "model", None)
        
        if lm_core is None:
             # Fallback
             lm_core = getattr(self.model.language_model, "model", None)
        if lm_core is not None:
            cache = getattr(lm_core, "_interleaver_cache", None)
            interleaver_info = self._serialize_interleaver_cache(cache)
            # Clear cache to avoid carrying over between frames
            lm_core._interleaver_cache = {}

        if not self.skip_backward:
            self.model.zero_grad(set_to_none=True)

        payload = {
            "tag": record_tag,
            "image_path": str(image_path),
            "input_speed_mps": current_speed,
            "meta": meta,
            "target_scalar": target_scalar.detach().to("cpu") if target_scalar is not None else None,
            "target_info": target_meta,
            "outputs": self._serialize_outputs(outputs),
            "text_outputs": self._serialize_text_outputs(text_features),
            "mode": self.explain_mode,
            "attention": attention_maps,
            "interleaver": interleaver_info,
            "debug_info": self.debug_info,
        }
        payload_path = scenario_subdirs["pt"] / f"{record_tag}.pt"
        torch.save(payload, payload_path)
        image_hw = (meta["original_height"], meta["original_width"])
        self._project_and_draw_points(
            pred_route,
            image_hw,
            (255, 0, 0, 255),
            radius=3,
            output_path=scenario_subdirs["route_overlay"] / f"{record_tag}.png",
        )
        self._project_and_draw_points(
            pred_speed_wps,
            image_hw,
            (0, 200, 0, 255),
            radius=2,
            output_path=scenario_subdirs["speed_overlay"] / f"{record_tag}.png",
        )
        self._write_text_outputs(
            scenario_subdirs["text_output"] / f"{record_tag}.txt",
            text_features,
            language_fallback=outputs[-1],
        )
        self._write_tensor_txt(pred_route, scenario_subdirs["pred_route"] / f"{record_tag}.txt")
        self._write_tensor_txt(
            pred_speed_wps, scenario_subdirs["pred_speed_wps"] / f"{record_tag}.txt"
        )
        # 입력 원본 이미지도 정리용으로 복사
        shutil.copy2(image_path, scenario_subdirs["input_images"] / image_path.name)

        # Clear recorder records to prevent memory accumulation
        self.recorder.records.clear()
        
        # Explicit cleanup
        del outputs, payload, attention_maps, text_features, driving_input
        if not self.skip_backward:
            del target_scalar
        
        # Force GC and empty cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def run_batch(self, image_paths: List[Path], output_dir: Path, scene_dir: Path) -> Path:
        """Run inference on a batch of images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup speed table
        speed_dir = scene_dir / self.speed_subdir if self.speed_subdir else None
        speed_table: Dict[str, float] = {}
        if speed_dir and speed_dir.exists():
            speed_table = self._load_speed_table(speed_dir)
            
        # Setup output dirs
        mode_suffix = "action" if self.explain_mode == "action" else "text"
        scenario_output_dir = self._prepare_output_subdir(output_dir, scene_dir, mode_suffix)
        scenario_subdirs = self._prepare_scenario_subdirs(scenario_output_dir)
        
        for image_path in tqdm(image_paths, desc=f"Batch ({self.explain_mode})", unit="img"):
            self._process_single_frame(image_path, speed_table, scenario_subdirs, scenario_output_dir)
            
        return scenario_output_dir

    def run_scene(self, scene_dir: Path, output_dir: Path) -> None:
        """시나리오 디렉토리(하위 images/)의 모든 이미지를 순회하며 추론 결과를 저장합니다."""
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        # frames 디렉토리 우선 사용, 없으면 images 폴백
        candidate_dirs = []
        if self.frames_subdir:
            candidate_dirs.append(scene_dir / self.frames_subdir)
        candidate_dirs.append(scene_dir / "images")
        images_dir = None
        for cdir in candidate_dirs:
            if cdir.exists():
                images_dir = cdir
                break
        if images_dir is None:
            raise FileNotFoundError(f"No frames directory found under {scene_dir} (tried {candidate_dirs})")
        
        image_paths = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        
        self.run_batch(image_paths, output_dir, scene_dir)

    def _prepare_driving_input(
        self,
        image_path: Path,
        current_speed: float = 0.0,
        append_response: str = "",
        target_points: Optional[torch.Tensor] = None,
    ) -> Tuple[DrivingInput, Dict[str, Any]]:
        """단일 이미지를 전처리하여 DrivingInput과 메타 정보를 생성합니다."""
        processed_image, num_patches, orig_hw = self._preprocess_image(image_path)
        placeholder_batch_list: List[dict] = []
        target_point_tensor = torch.zeros(1, 2, dtype=torch.float32, device=self.device)

        if target_points is not None and target_points.numel() > 0:
            tp = target_points.detach()
            if tp.dim() == 2:
                tp = tp.unsqueeze(0)
            tp = tp.to(device=self.device, dtype=torch.float32)
            target_point_tensor = tp[:, 0, :2]
            token_id = self.tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
            placeholder_batch_list.append({token_id: tp[0, :, :2].to("cpu").numpy()})
        prompt_str = (
            f"Current speed: {current_speed:.1f} m/s. What should the ego vehicle do next? Provide a short commentary."
        )
        lang_label = self._build_language_label(
            prompt_str, placeholder_batch_list, num_patches, assistant_response=append_response
        )
        if self.debug:
            print(f"[DEBUG] _prepare_driving_input: append_response='{append_response}'")
            print(f"[DEBUG] _prepare_driving_input: lang_label.phrase_ids.shape={lang_label.phrase_ids.shape}")
        
        # Build DrivingInput
        # prompt_token_ids = self._extract_prompt_token_ids(lang_label) # This line was not in the original code, but was in the instruction. Removing it.
        camera_intrinsics = get_camera_intrinsics(orig_hw[1], orig_hw[0], 110).unsqueeze(0).unsqueeze(0)
        camera_extrinsics = get_camera_extrinsics().unsqueeze(0).unsqueeze(0)
        driving_input = DrivingInput(
            camera_images=processed_image.to(self.device).bfloat16(),
            image_sizes=torch.tensor([[orig_hw[0], orig_hw[1]]], dtype=torch.float32).to(self.device),
            camera_intrinsics=camera_intrinsics.to(self.device),
            camera_extrinsics=camera_extrinsics.to(self.device),
            vehicle_speed=torch.tensor([[current_speed]], dtype=torch.float32, device=self.device),
            target_point=target_point_tensor,
            prompt=lang_label,
            prompt_inference=lang_label,
        )
        meta = {
            "original_height": orig_hw[0],
            "original_width": orig_hw[1],
            "num_patch_views": num_patches,
            "num_image_tokens_per_patch": self.num_image_token,
            "num_total_image_tokens": self.num_image_token * num_patches,
            "frames": self.T,
        }
        return driving_input, meta

    def _compute_target_scalar(
        self,
        outputs,
        text_features: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if self.explain_mode == "text":
            return self._compute_text_target(text_features)
        return self._compute_action_target(outputs)

    def _apply_parametric_spline(
        self, tensor: Optional[torch.Tensor], head: Optional[str] = None
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """필요 시 파라메트릭 cubic smoothing spline으로 리샘플합니다."""
        meta = {"applied": False}
        if tensor is None or not self.use_spline:
            return tensor, meta
        orig_len = tensor.shape[1] if tensor.dim() >= 2 else 0
        duration = None
        param = None
        if head == "speed" and orig_len > 1:
            duration = DELTA_T * (orig_len - 1)
            param = torch.linspace(
                0.0,
                duration,
                steps=orig_len,
                device=tensor.device,
                dtype=tensor.dtype if tensor.is_floating_point() else torch.float32,
            )
        resampled = _parametric_cubic_smoothing_resample(
            tensor,
            smoothing=self.spline_smoothing,
            num_samples=self.spline_num_samples,
            param=param,
        )
        effective_dt = None
        if duration is not None and resampled is not None and resampled.shape[1] > 1:
            effective_dt = duration / (resampled.shape[1] - 1)
        meta.update(
            {
                "applied": True,
                "smoothing": self.spline_smoothing,
                "num_samples": resampled.shape[1] if resampled is not None else None,
                "duration": duration,
                "effective_dt": effective_dt,
            }
        )
        return resampled, meta

    def _compute_action_target(self, outputs) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """선택한 운동학 함수로 pred_route/pred_speed에서 스칼라 y_t를 계산합니다."""
        pred_speed_wps, pred_route, _ = outputs
        metric_cfg = KINEMATIC_METRICS.get(self.kinematic_metric, KINEMATIC_METRICS["none"])
        meta: Dict[str, Any] = {
            "type": "action",
            "target_mode": self.target_mode,
            "kinematic_metric": self.kinematic_metric,
            "description": metric_cfg.get("description"),
            "head": None,
        }
        source = metric_cfg["source"]
        metric_fn = metric_cfg["fn"]
        tensor = None
        if source == "route":
            tensor = pred_route
            meta["head"] = "route"
        elif source == "speed":
            tensor = pred_speed_wps
            meta["head"] = "speed"

        tensor, spline_meta = self._apply_parametric_spline(tensor, head=meta["head"])
        meta["spline"] = spline_meta

        if metric_fn is not None:
            if tensor is None:
                return None, meta
            metric_kwargs = {}
            if meta["head"] == "speed":
                dt = spline_meta.get("effective_dt")
                if dt is not None:
                    metric_kwargs["delta_t"] = float(dt)
            return metric_fn(tensor, **metric_kwargs), meta

        if self.target_mode in ("auto", "route") and meta.get("head") == "route" and tensor is not None:
            meta["head"] = "route"
            return tensor.float().abs().sum(), meta
        if self.target_mode in ("auto", "speed") and meta.get("head") == "speed" and tensor is not None:
            meta["head"] = "speed"
            return tensor.float().abs().sum(), meta
        return None, meta

    def _compute_text_target(
        self,
        text_features: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """텍스트 모드에서 생성 토큰 로짓을 선택해 스칼라 타깃으로 사용합니다."""
        if text_features is None:
            raise RuntimeError("Text features are unavailable but text explain mode was requested.")
        token_scores = text_features["token_scores"]
        if token_scores.numel() == 0:
            raise RuntimeError("No generated token logits were found for text explain mode.")
        strategy = self.text_token_strategy
        if strategy == "max":
            value, index = torch.max(token_scores, dim=0)
            scalar = value
            token_idx = int(index.item())
        elif strategy == "last":
            token_idx = token_scores.numel() - 1
            scalar = token_scores[token_idx]
        else:  # index 전략
            requested = self.text_token_index
            if requested < 0:
                requested = 0
            token_idx = min(requested, token_scores.numel() - 1)
            scalar = token_scores[token_idx]
        token_ids = text_features["token_ids"]
        token_strings = text_features["token_strings"]
        chosen_token_id = int(token_ids[token_idx].item()) if token_ids.numel() else -1
        token_string = token_strings[token_idx] if 0 <= token_idx < len(token_strings) else ""
        meta = {
            "type": "text",
            "token_strategy": strategy,
            "token_index": token_idx,
            "token_id": chosen_token_id,
            "token_string": token_string,
        }
        return scalar, meta

    def _gather_text_features(self) -> Optional[Dict[str, Any]]:
        """모델이 방금 생성한 토큰 ID·로짓·문자열을 추출합니다."""
        token_id_seq = getattr(self.model, "text_token_ids", None)
        token_logit_seq = getattr(self.model, "text_token_logits", None)
        if (not token_id_seq or not token_logit_seq) and hasattr(self.model, "language_model"):
            lm = getattr(self.model, "language_model", None)
            last_tokens = getattr(lm, "last_sampled_tokens", None)
            last_scores = getattr(lm, "last_sampled_token_scores", None)
            if last_tokens is not None and last_scores is not None:
                token_id_seq = [last_tokens]
                token_logit_seq = [last_scores]
        if not token_id_seq or not token_logit_seq:
            return None
        token_ids = token_id_seq[0]
        token_scores = token_logit_seq[0]
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        if token_scores.dim() > 1:
            token_scores = token_scores.squeeze(0)
        token_ids = token_ids.to(torch.long)
        decoded_text = ""
        if hasattr(self.model, "language") and self.model.language:
            decoded_text = self.model.language[0]
        token_strings = (
            self.tokenizer.convert_ids_to_tokens(token_ids.tolist()) if token_ids.numel() > 0 else []
        )
        return {
            "token_ids": token_ids,
            "token_scores": token_scores,
            "token_strings": token_strings,
            "decoded_text": decoded_text,
        }

    @staticmethod
    # 행동 요약은 더 이상 사용하지 않음 (텍스트 출력은 모델 생성/펜싱 fallback만 사용)

    def _serialize_text_outputs(text_features: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """text_features 텐서를 CPU 리스트로 변환해 저장 가능하도록 만듭니다."""
        if text_features is None:
            return None
        token_ids = text_features["token_ids"].detach().to("cpu").tolist()
        token_scores = text_features["token_scores"].detach().to("cpu").tolist()
        return {
            "token_ids": token_ids,
            "token_scores": token_scores,
            "token_strings": text_features["token_strings"],
            "decoded_text": text_features["decoded_text"],
        }

    def _preprocess_image(self, image_path: Path):
        """Sim-Lingo용 동적 전처리를 수행해 패치 텐서를 반환합니다."""
        image = Image.open(image_path).convert("RGB")
        use_global_img = getattr(self.cfg.model.vision_model, "use_global_img", True)
        images = dynamic_preprocess(
            image,
            image_size=self.image_size,
            use_thumbnail=use_global_img,
            max_num=self.max_patches,
        )
        pixel_values = torch.stack([self.transform(img) for img in images])
        pixel_values = pixel_values.unsqueeze(0)
        num_patches = pixel_values.shape[1]
        C = pixel_values.shape[2]
        H = pixel_values.shape[3]
        W = pixel_values.shape[4]
        processed_image = pixel_values.view(1, self.T, num_patches, C, H, W)
        return processed_image, num_patches, (image.height, image.width)

    def _run_teacher_forcing_pass(
        self, image_path: Path, current_speed: float, generated_text: str, original_outputs
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Generated text를 포함한 입력을 구성해 Teacher Forcing으로 그래디언트를 계산합니다."""
        target_points = None
        tf_response = generated_text
        if self.explain_mode == "action":
            _, pred_route, _ = original_outputs
            if pred_route is not None and pred_route.numel() > 0:
                tp = pred_route.detach()
                if tp.dim() == 2:
                    tp = tp.unsqueeze(0)
                coords = tp[..., :2]
                coords_count = min(coords.shape[1], 2)
                if coords_count > 0:
                    target_points = coords[:, :coords_count]
                    tf_response = "<TARGET_POINT>" * coords_count

        # 1. Re-build input with full text
        tf_input, _ = self._prepare_driving_input(
            image_path, current_speed, append_response=tf_response, target_points=target_points
        )
        
        # 2. Get adaptors (inference=True to prepare embeddings, but we use them for TF)
        adaptor_dict = self.model.adaptors(tf_input, inference=True)
        
        # 3. Manual Forward Pass (bypassing generation)
        # forward_model returns (adaptor_features, adaptor_logits)
        features, logits = self.model.forward_model(tf_input, adaptor_dict)
        
        # 4. Compute Target Scalar
        meta = {"pass": "teacher_forcing"}
        
        if self.explain_mode == "action":
            # Decode action from features
            # SimLingo logic: driving features are at the end
            # We need to know the length of driving tokens? 
            # Actually, `self.adaptors.driving` handles the splitting if we pass the full features?
            # No, `DrivingModel.forward` splits explicitly:
            # len_driving = inputs_driving["inputs"].size(1)
            # driving_features = features[:, -len_driving:]
            
            # We need `inputs_driving` to know the length.
            inputs_driving = self.model.adaptors.driving(tf_input)
            len_driving = inputs_driving["inputs"].size(1)
            
            driving_features = features[:, -len_driving:]
            driving_logits = logits[:, -len_driving:]
            
            predictions = self.model.adaptors.driving.get_predictions(driving_features, driving_logits)
            
            # Reconstruct outputs tuple for _compute_action_target
            # (pred_speed_wps, pred_route, language)
            # We only care about action here.
            pred_speed_wps = predictions.get("speed_wps")
            pred_route = predictions.get("route")
            
            # Use the newly computed action predictions for target calculation
            tf_outputs = (pred_speed_wps, pred_route, None)
            return self._compute_action_target(tf_outputs)
            
        elif self.explain_mode == "text":
            # Use logits directly
            # We need to select the target token.
            # Strategy: "max" (max logit in the generated part) or "last" (last token logit)
            
            # We need to identify which logits correspond to the *generated* part.
            # The input sequence is [Prompt + Generated].
            # We need to know the length of the Prompt.
            
            # Re-calculate prompt length without response
            # This is expensive but safe.
            prompt_only_input, _ = self._prepare_driving_input(image_path, current_speed, append_response="")
            prompt_len = prompt_only_input.prompt.phrase_ids.shape[1]
            if self.debug:
                print(f"[DEBUG] Prompt len: {prompt_len}, Full seq len: {logits.shape[1]}")
            
            # Identify generated tokens: those after prompt_len
            gen_logits = logits[:, prompt_len:, :] # [B, GenLen, Vocab]
            if self.debug:
                print(f"[DEBUG] Gen logits shape: {gen_logits.shape}")
            
            if gen_logits.shape[1] == 0:
                # Fallback if no text generated
                if self.debug:
                    print(f"[DEBUG] No text generated (gen_logits is empty).")
                return None, {"error": "no_text_generated"}
                
            # Construct text_features dict for _compute_text_target
            # We need token_scores (logits of selected tokens? No, just logits)
            # _compute_text_target expects "token_scores" to be [GenLen] (scores of chosen tokens) 
            # OR [GenLen, Vocab]?
            # Let's check _compute_text_target.
            # It does: value, index = torch.max(token_scores, dim=0) if strategy="max"
            # So it expects `token_scores` to be 1D (sequence of scores) or 2D?
            # `_gather_text_features` returns `token_scores` from `last_sampled_token_scores`.
            # `last_sampled_token_scores` is usually [B, GenLen] (log probs or logits of *sampled* tokens).
            
            # So we need to gather the logits of the *actual* tokens in the sequence.
            # The actual tokens are in `tf_input.prompt.phrase_ids[:, prompt_len:]`.
            
            full_ids = tf_input.prompt.phrase_ids
            gen_ids = full_ids[:, prompt_len:] # [B, GenLen]
            if self.debug:
                print(f"[DEBUG] full_ids shape: {full_ids.shape}, gen_ids shape: {gen_ids.shape}")
            
            # Gather logits for these ids
            # gen_logits: [B, GenLen, Vocab]
            # We want [B, GenLen] scores.
            
            # torch.gather
            # gen_logits.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
            
            token_scores = torch.gather(gen_logits, 2, gen_ids.unsqueeze(-1)).squeeze(-1)
            if self.debug:
                print(f"[DEBUG] token_scores shape after gather: {token_scores.shape}")
            # If we have multiple generated tokens, we might want to keep them all or pick one
            # Logic for text_token_strategy is handled in _compute_text_target, but we need token_scores populated
            
            if token_scores.dim() > 1:
                token_scores = token_scores.squeeze(0) # [GenLen]
            if self.debug:
                print(f"[DEBUG] token_scores shape final: {token_scores.shape}")
            
            # Also need token_ids, token_strings for meta
            token_ids = gen_ids.squeeze(0)
            token_strings = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
            
            text_features = {
                "token_scores": token_scores,
                "token_ids": token_ids,
                "token_strings": token_strings,
                "decoded_text": generated_text
            }
            
            return self._compute_text_target(text_features)
            
        return None, meta

    @staticmethod
    def _to_cpu(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.detach().to("cpu")

    def _serialize_outputs(self, outputs):
        """모델 출력 텐서를 CPU로 옮겨 직렬화 가능한 형태로 만듭니다."""
        pred_speed_wps, pred_route, language = outputs
        return {
            "pred_speed_wps": self._to_cpu(pred_speed_wps),
            "pred_route": self._to_cpu(pred_route),
            "language": language,
        }

    def _serialize_interleaver_cache(self, cache: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Interleaver 구간 텐서 및 grad를 직렬화합니다."""
        if not cache:
            return None

        def _pack(t: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
            if t is None:
                return None
            grad = t.grad.detach().to("cpu") if t.grad is not None else None
            return {
                "value": t.detach().to("cpu"),
                "grad": grad,
                "shape": tuple(t.shape),
            }

        selected = cache.get("selected_mask")
        data: Dict[str, Any] = {
            "selected_mask": selected.detach().to("cpu") if selected is not None else None,
            "vit_batch_size": cache.get("vit_batch_size"),
            "input_shape": cache.get("input_shape"),
            "num_patches_list": cache.get("num_patches_list"),
            "tokens_per_patch": cache.get("tokens_per_patch"),
            "vit_embeds": _pack(cache.get("vit_embeds")),
            "pixel_shuffle_out": _pack(cache.get("pixel_shuffle_out")),
            "mlp1_input": _pack(cache.get("mlp1_input")),
            "mlp1_output": _pack(cache.get("mlp1_output")),
            "mlp1_weight": cache.get("mlp1_weight"),
            "mlp1_bias": cache.get("mlp1_bias"),
        }
        return data

    def _build_language_label(
        self,
        prompt: str,
        placeholder_batch_list: List[dict],
        num_patches_all: int,
        assistant_response: str = "",
    ) -> LanguageLabel:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Waypoints:" + assistant_response}],
            },
        ]
        conv_batch_list = [conversation]
        questions: List[str] = []
        for conv in conv_batch_list:
            for item in conv:
                questions.append(item["content"][0]["text"])
                item["content"] = item["content"][0]["text"]
        conv_module = self._load_conversation_template_module()
        prompt_batch_list: List[str] = []
        for idx, conv in enumerate(conv_batch_list):
            question = questions[idx]
            if "<image>" not in question:
                question = "<image>\n" + question
            template = conv_module.get_conv_template("internlm2-chat")
            for conv_part_idx, conv_part in enumerate(conv):
                if conv_part["role"] == "assistant":
                    # If we have content for assistant, append it.
                    # Otherwise append None (which usually means "start generation here")
                    # But here we want to FORCE the content if provided.
                    content = conv_part.get("content")
                    # In the loop above, we flattened content to string.
                    # Wait, the loop above:
                    # for item in conv:
                    #     questions.append(item["content"][0]["text"]) <-- This is wrong for assistant!
                    #     item["content"] = item["content"][0]["text"]
                    
                    # The loop above seems to assume everything is user question?
                    # Let's look at lines 1446-1449.
                    
                    # Actually, let's just fix the logic here.
                    # We want to append the content if it's not None/Empty.
                    # If assistant_response was passed, it is in conv_part["content"] (as a string now).
                    
                    # However, if we want to prompt the model to *continue*, we might need to handle it carefully.
                    # But for teacher forcing, we want the FULL conversation.
                    
                    val = conv_part["content"]
                    if val == "Waypoints:": # Empty response case -> Start generation
                         template.append_message(template.roles[1], None)
                    else:
                         # Teacher forcing case -> Force the content
                         template.append_message(template.roles[1], val)

                elif conv_part["role"] == "user":
                    if conv_part_idx == 0 and "<image>" not in conv_part["content"]:
                        conv_part["content"] = "<image>\n" + conv_part["content"]
                    template.append_message(template.roles[0], conv_part["content"])
                else:
                    raise ValueError(f"Unsupported role {conv_part['role']}")
            query = template.get_prompt()
            system_prompt = template.system_template.replace(
                "{system_message}", template.system_message
            ) + template.sep
            query = query.replace(system_prompt, "")
            IMG_START_TOKEN = "<img>"
            IMG_END_TOKEN = "</img>"
            IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches_all
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            prompt_batch_list.append(query)

        prompt_tokenized = self.tokenizer(
            prompt_batch_list,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        prompt_tokenized_ids = prompt_tokenized["input_ids"]
        prompt_tokenized_valid = prompt_tokenized["input_ids"] != self.tokenizer.pad_token_id
        prompt_tokenized_mask = prompt_tokenized_valid
        ll = LanguageLabel(
            phrase_ids=prompt_tokenized_ids.to(self.device),
            phrase_valid=prompt_tokenized_valid.to(self.device),
            phrase_mask=prompt_tokenized_mask.to(self.device),
            placeholder_values=placeholder_batch_list,
            language_string=prompt_batch_list,
            loss_masking=None,
        )
        return ll

    def _load_conversation_template_module(self):
        cache_dir = Path("pretrained") / self.cfg.model.vision_model.variant.split("/")[1]
        cache_dir = Path(to_absolute_path(str(cache_dir)))
        model_path = cache_dir / "conversation.py"
        if not model_path.exists():
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=self.cfg.model.vision_model.variant,
                local_dir=str(cache_dir),
            )
        spec = importlib.util.spec_from_file_location("get_conv_template", model_path)
        conv_module = importlib.util.module_from_spec(spec)
        sys.modules["get_conv_template"] = conv_module
        spec.loader.exec_module(conv_module)
        return conv_module


def parse_args():
    """CLI 인자를 정의해 추론 설정을 덮어쓸 수 있도록 합니다."""
    parser = argparse.ArgumentParser(description="Sim-Lingo inference baseline runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to Hydra config.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Model checkpoint path (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=DEFAULT_SCENE_DIR,
        help=f"Scenario directory containing frames/speed subdirs (default: {DEFAULT_SCENE_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--device", type=str, default=None, help="Device identifier, e.g., cuda:0")
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=["auto", "speed", "route"],
        default="auto",
        help="Which prediction head to use for scalar target construction.",
    )
    parser.add_argument(
        "--explain_mode",
        type=str,
        choices=["action", "text"],
        default=DEFAULT_EXPLAIN_MODE,
        help="Determines whether gradients target action heads or generated text logits.",
    )
    parser.add_argument(
        "--text_token_strategy",
        type=str,
        choices=list(TEXT_TOKEN_STRATEGIES),
        default="max",
        help="Strategy for selecting which generated token logit to backpropagate in text mode.",
    )
    parser.add_argument(
        "--text_token_index",
        type=int,
        default=-1,
        help="Explicit generated token index to use when --text_token_strategy=index.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints.",
    )
    parser.add_argument(
        "--kinematic_metric",
        type=str,
        choices=list(KINEMATIC_METRICS.keys()),
        default=DEFAULT_KINEMATIC_METRIC,
        help="Kinematic scalar function to apply on action outputs before backprop.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Input resize for dynamic_preprocess (reduce for lower GPU memory).",
    )
    parser.add_argument(
        "--max_patches",
        type=int,
        default=DEFAULT_MAX_PATCHES,
        help="Max number of patches (dynamic_preprocess max_num). Use 1 to reduce GPU memory.",
    )
    parser.add_argument(
        "--frames_subdir",
        type=str,
        default=DEFAULT_FRAMES_SUBDIR,
        help="Subdirectory under scene_dir containing frames (default: video_garmin). Fallback to images/ if missing.",
    )
    parser.add_argument(
        "--speed_subdir",
        type=str,
        default=DEFAULT_SPEED_SUBDIR,
        help="Subdirectory under scene_dir containing per-frame speed txt (m/s) (default: video_garmin_speed).",
    )
    parser.add_argument(
        "--use_spline",
        action="store_true",
        help="Enable parametric cubic smoothing spline on route/speed waypoints before kinematic metrics.",
    )
    parser.add_argument(
        "--spline_smoothing",
        type=float,
        default=DEFAULT_SPLINE_SMOOTHING,
        help="Non-negative smoothing strength (second-derivative penalty). 0 keeps raw points.",
    )
    parser.add_argument(
        "--spline_num_samples",
        type=int,
        default=DEFAULT_SPLINE_NUM_SAMPLES,
        help="Number of samples to evaluate on the spline. 0 keeps the original waypoint count.",
    )
    return parser.parse_args()


def main():
    """CLI 인자를 바탕으로 러너를 생성하고 장면 추론을 실행합니다."""
    args = parse_args()
    runner = SimLingoInferenceBaseline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        target_mode=args.target_mode,
        explain_mode=args.explain_mode,
        text_token_strategy=args.text_token_strategy,
        text_token_index=args.text_token_index,
        kinematic_metric=args.kinematic_metric,
        image_size=args.image_size,
        max_patches=args.max_patches,
        frames_subdir=args.frames_subdir,
        speed_subdir=args.speed_subdir,
        use_spline=args.use_spline,
        spline_smoothing=args.spline_smoothing,
        spline_num_samples=args.spline_num_samples,
        debug=args.debug,
    )
    runner.run_scene(args.scene_dir, args.output_dir)


if __name__ == "__main__":
    main()
