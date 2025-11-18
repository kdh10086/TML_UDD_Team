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
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoConfig, AutoProcessor

# 어떤 위치에서 실행해도 external/simlingo 모듈을 불러올 수 있도록 경로 추가
REPO_ROOT = Path(__file__).resolve().parents[1]
SIMLINGO_SRC = REPO_ROOT / "external" / "simlingo"
if SIMLINGO_SRC.exists() and str(SIMLINGO_SRC) not in sys.path:
    sys.path.insert(0, str(SIMLINGO_SRC))

from simlingo_training.models.driving import DrivingModel
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
from team_code.simlingo_utils import get_camera_extrinsics, get_camera_intrinsics

DEFAULT_CONFIG_PATH = Path("checkpoints/simlingo/simlingo/.hydra/config.yaml")  # 기본 Hydra config
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt")  # 기본 ckpt
DEFAULT_SCENE_DIR = Path("data/scene01")  # 기본 입력 이미지 디렉토리
DEFAULT_OUTPUT_DIR = Path("experiment_outputs/simlingo_inference")  # 기본 출력 디렉토리
DEFAULT_EXPLAIN_MODE = os.environ.get("SIMLINGO_EXPLAIN_MODE", "action")  # action/text 모드 기본값
DEFAULT_KINEMATIC_METRIC = "curv_energy"  # 운동학 함수 기본값

EPS = 1e-6
DELTA_S = 1.0
DELTA_T = 0.25


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x.float()


# pred_route -> 곡률 제곱합 (조향 강도)
def compute_curvature_energy(route: torch.Tensor, delta_s: float = DELTA_S) -> torch.Tensor:
    route = _ensure_batch(route)
    diffs = route[:, 1:, :] - route[:, :-1, :]
    tangent = diffs / (diffs.norm(dim=-1, keepdim=True).clamp_min(EPS))
    headings = torch.atan2(tangent[..., 1], tangent[..., 0])
    heading_diff = headings[:, 1:] - headings[:, :-1]
    curvature = heading_diff / delta_s
    return (curvature ** 2).sum()


# pred_route -> 곡률 변화 제곱합 (조향 부드러움/jerk)
def compute_curvature_diff(route: torch.Tensor, delta_s: float = DELTA_S) -> torch.Tensor:
    route = _ensure_batch(route)
    diffs = route[:, 1:, :] - route[:, :-1, :]
    tangent = diffs / (diffs.norm(dim=-1, keepdim=True).clamp_min(EPS))
    headings = torch.atan2(tangent[..., 1], tangent[..., 0])
    curvature = (headings[:, 1:] - headings[:, :-1]) / delta_s
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
def _derive_longitudinal_terms(speed_wps: torch.Tensor):
    speed_wps = _ensure_batch(speed_wps)
    diffs = speed_wps[:, 1:, :] - speed_wps[:, :-1, :]
    base_dir = diffs[:, :1, :]
    norm = base_dir.norm(dim=-1, keepdim=True).clamp_min(EPS)
    direction = base_dir / norm
    velocities = (diffs * direction).sum(dim=-1) / DELTA_T
    accelerations = (velocities[:, 1:] - velocities[:, :-1]) / DELTA_T
    jerks = (accelerations[:, 1:] - accelerations[:, :-1]) / DELTA_T
    return velocities, accelerations, jerks


# pred_speed -> 평균 전진 속도 (양수 성분)
def compute_forward_speed(speed_wps: torch.Tensor) -> torch.Tensor:
    velocities, _, _ = _derive_longitudinal_terms(speed_wps)
    return torch.relu(velocities).mean()


# pred_speed -> 가속도 에너지 (승차감/동적 강도)
def compute_acceleration_energy(speed_wps: torch.Tensor) -> torch.Tensor:
    _, accelerations, _ = _derive_longitudinal_terms(speed_wps)
    return (accelerations ** 2).sum()


# pred_speed -> 제동 에너지 (감속 위험)
def compute_brake_energy(speed_wps: torch.Tensor) -> torch.Tensor:
    _, accelerations, _ = _derive_longitudinal_terms(speed_wps)
    braking = torch.relu(-accelerations)
    return (braking ** 2).sum()


# pred_speed -> jerk 에너지 (가속도 변화량)
def compute_jerk_energy(speed_wps: torch.Tensor) -> torch.Tensor:
    _, _, jerks = _derive_longitudinal_terms(speed_wps)
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

    def __init__(self, store_on_cpu: bool = True) -> None:
        self.store_on_cpu = store_on_cpu
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.records: List[Dict[str, Any]] = []
        self._active_buffers: Optional[defaultdict] = None
        self._current_tag: Optional[str] = None

    def register_module(self, module: torch.nn.Module, name: str) -> None:
        def hook(_, __, output):
            if self._active_buffers is None:
                return
            attn = self._extract_attention(output)
            if attn is None:
                return
            attn.retain_grad()
            self._active_buffers.setdefault(name, []).append(attn)

        handle = module.register_forward_hook(hook)
        self.handles.append(handle)

    @staticmethod
    def _extract_attention(output) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            return output
        if isinstance(output, (list, tuple)):
            for elem in output:
                if torch.is_tensor(elem) and elem.dim() == 4:
                    return elem
        return None

    def start_recording(self, tag: str) -> None:
        self._current_tag = tag
        self._active_buffers = defaultdict(list)

    def stop_recording(self) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        assert self._active_buffers is not None, "Recording was not started."
        aggregated: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        for name, tensors in self._active_buffers.items():
            aggregated[name] = []
            for tensor in tensors:
                attn_tensor = tensor.detach()
                grad_tensor = tensor.grad.detach() if tensor.grad is not None else None
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
                # 불필요한 참조를 끊어 메모리 누수를 방지
                tensor.grad = None
        self.records.append({"tag": self._current_tag, "maps": aggregated})
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
        self.transform = build_transform(input_size=448)
        self.T = 1
        self.num_image_token = self._compute_num_image_tokens(self.cfg.model.vision_model.variant)
        self.model = self._build_model()
        self.recorder = AttentionRecorder()
        self._register_attention_hooks()

    def _compute_num_image_tokens(self, encoder_variant: str) -> int:
        """InternVL2 비전 인코더 설정에서 이미지 토큰(패치 수)을 계산합니다."""
        cfg = AutoConfig.from_pretrained(encoder_variant, trust_remote_code=True)
        image_size = cfg.force_image_size or cfg.vision_config.image_size
        patch_size = cfg.vision_config.patch_size
        return int((image_size // patch_size) ** 2 * (cfg.downsample_ratio ** 2))

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
        return model

    def _register_attention_hooks(self) -> None:
        """비전/언어 트랜스포머 블록 전체에 레코더 훅을 연결합니다."""
        # 비전 인코더 블록 훅 등록
        vision_model = getattr(self.model.vision_model.image_encoder.model, "vision_model", None)
        if vision_model is not None and hasattr(vision_model, "encoder"):
            for idx, block in enumerate(vision_model.encoder.layers):
                self.recorder.register_module(block, f"vision_block_{idx}")
        # 언어 모델 블록 훅 등록
        lm = self.model.language_model.model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            layers = lm.model.layers
        elif hasattr(lm, "layers"):
            layers = lm.layers
        else:
            layers = []
        for idx, block in enumerate(layers):
            self.recorder.register_module(block, f"language_block_{idx}")

    def _prepare_output_subdir(self, output_root: Path, scenario_name: str, mode_suffix: str) -> Path:
        """scene-모드-타임스탬프 규칙으로 하위 디렉토리를 만들고 중복 시 숫자 suffix를 붙입니다."""
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        base_name = f"{scenario_name}_{mode_suffix}_{timestamp}"
        candidate = output_root / base_name
        counter = 1
        while candidate.exists():
            candidate = output_root / f"{base_name}_{counter}"
            counter += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def run_scene(self, scene_dir: Path, output_dir: Path) -> None:
        """scene_dir의 모든 이미지를 순회하며 추론 결과를 payload 형태로 저장합니다."""
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        scenario_name = scene_dir.name
        mode_suffix = "action" if self.explain_mode == "action" else "text"
        scenario_output_dir = self._prepare_output_subdir(output_dir, scenario_name, mode_suffix)
        image_paths = sorted(
            [p for p in scene_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        for image_path in image_paths:
            record_tag = image_path.stem
            self.recorder.start_recording(record_tag)
            driving_input, meta = self._prepare_driving_input(image_path)
            outputs = self.model(driving_input)
            text_features = self._gather_text_features()
            target_scalar, target_meta = self._compute_target_scalar(outputs, text_features)
            if target_scalar is None:
                raise RuntimeError("Target scalar could not be computed for relevance backprop.")
            target_scalar.backward(retain_graph=False)
            attention_maps = self.recorder.stop_recording()
            self.model.zero_grad(set_to_none=True)
            payload = {
                "tag": record_tag,
                "image_path": str(image_path),
                "meta": meta,
                "target_scalar": target_scalar.detach().to("cpu"),
                "target_info": target_meta,
                "outputs": self._serialize_outputs(outputs),
                "text_outputs": self._serialize_text_outputs(text_features),
                "mode": self.explain_mode,
                "attention": attention_maps,
            }
            output_path = scenario_output_dir / f"{image_path.stem}_{mode_suffix}.pt"
            torch.save(payload, output_path)

    def _prepare_driving_input(self, image_path: Path) -> Tuple[DrivingInput, Dict[str, Any]]:
        """단일 이미지를 전처리하여 DrivingInput과 메타 정보를 생성합니다."""
        processed_image, num_patches, orig_hw = self._preprocess_image(image_path)
        placeholder_batch_list: List[dict] = []
        prompt_str = "Current speed: 0 m/s. Predict the waypoints."
        lang_label = self._build_language_label(prompt_str, placeholder_batch_list, num_patches)
        camera_intrinsics = get_camera_intrinsics(orig_hw[1], orig_hw[0], 110).unsqueeze(0).unsqueeze(0)
        camera_extrinsics = get_camera_extrinsics().unsqueeze(0).unsqueeze(0)
        driving_input = DrivingInput(
            camera_images=processed_image.to(self.device).bfloat16(),
            image_sizes=torch.tensor([[orig_hw[0], orig_hw[1]]], dtype=torch.float32).to(self.device),
            camera_intrinsics=camera_intrinsics.to(self.device),
            camera_extrinsics=camera_extrinsics.to(self.device),
            vehicle_speed=torch.zeros(1, 1, dtype=torch.float32, device=self.device),
            target_point=torch.zeros(1, 2, dtype=torch.float32, device=self.device),
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

        if metric_fn is not None:
            if tensor is None:
                return None, meta
            return metric_fn(tensor), meta

        if self.target_mode in ("auto", "route") and pred_route is not None:
            meta["head"] = "route"
            return pred_route.float().abs().sum(), meta
        if self.target_mode in ("auto", "speed") and pred_speed_wps is not None:
            meta["head"] = "speed"
            return pred_speed_wps.float().abs().sum(), meta
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

    def _serialize_text_outputs(self, text_features: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
        images = dynamic_preprocess(
            image,
            image_size=448,
            use_thumbnail=self.cfg.model.vision_model.use_global_img,
            max_num=2,
        )
        pixel_values = torch.stack([self.transform(img) for img in images])
        pixel_values = pixel_values.unsqueeze(0)
        num_patches = pixel_values.shape[1]
        C = pixel_values.shape[2]
        H = pixel_values.shape[3]
        W = pixel_values.shape[4]
        processed_image = pixel_values.view(1, self.T, num_patches, C, H, W)
        return processed_image, num_patches, (image.height, image.width)

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

    def _build_language_label(
        self,
        prompt: str,
        placeholder_batch_list: List[dict],
        num_patches_all: int,
    ) -> LanguageLabel:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Waypoints:"}],
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
                    template.append_message(template.roles[1], None)
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
        help=f"Directory with input images (default: {DEFAULT_SCENE_DIR})",
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
        "--kinematic_metric",
        type=str,
        choices=list(KINEMATIC_METRICS.keys()),
        default=DEFAULT_KINEMATIC_METRIC,
        help="Kinematic scalar function to apply on action outputs before backprop.",
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
    )
    runner.run_scene(args.scene_dir, args.output_dir)


if __name__ == "__main__":
    main()
