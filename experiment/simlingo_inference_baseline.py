#!/usr/bin/env python3
"""Light-weight Sim-Lingo inference baseline for offline experiments.

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from simlingo_training.models.driving import DrivingModel
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
from team_code.simlingo_utils import get_camera_extrinsics, get_camera_intrinsics


class AttentionRecorder:
    """Registers hooks on attention blocks and stores weights + gradients."""

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
                # clear grad references to avoid leaks
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
    """Offline Sim-Lingo inference runner that captures every attention map."""

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: Optional[str] = None,
        target_mode: str = "auto",
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_mode = target_mode
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
        cfg = AutoConfig.from_pretrained(encoder_variant, trust_remote_code=True)
        image_size = cfg.force_image_size or cfg.vision_config.image_size
        patch_size = cfg.vision_config.patch_size
        return int((image_size // patch_size) ** 2 * (cfg.downsample_ratio ** 2))

    def _build_model(self) -> DrivingModel:
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
        # Ensure attentions are returned
        model.language_model.model.config.output_attentions = True
        model.language_model.model.config.output_hidden_states = True
        model.vision_model.image_encoder.model.config.output_attentions = True
        return model

    def _register_attention_hooks(self) -> None:
        # Vision encoder blocks
        vision_model = getattr(self.model.vision_model.image_encoder.model, "vision_model", None)
        if vision_model is not None and hasattr(vision_model, "encoder"):
            for idx, block in enumerate(vision_model.encoder.layers):
                self.recorder.register_module(block, f"vision_block_{idx}")
        # Language model blocks
        lm = self.model.language_model.model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            layers = lm.model.layers
        elif hasattr(lm, "layers"):
            layers = lm.layers
        else:
            layers = []
        for idx, block in enumerate(layers):
            self.recorder.register_module(block, f"language_block_{idx}")

    def run_scene(self, scene_dir: Path, output_dir: Path) -> None:
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(
            [p for p in scene_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        for image_path in image_paths:
            record_tag = image_path.stem
            self.recorder.start_recording(record_tag)
            driving_input, meta = self._prepare_driving_input(image_path)
            outputs = self.model(driving_input)
            target_scalar = self._compute_target_scalar(outputs)
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
                "outputs": self._serialize_outputs(outputs),
                "attention": attention_maps,
            }
            torch.save(payload, output_dir / f"{record_tag}_prediction.pt")

    def _prepare_driving_input(self, image_path: Path) -> Tuple[DrivingInput, Dict[str, Any]]:
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

    def _compute_target_scalar(self, outputs) -> Optional[torch.Tensor]:
        pred_speed_wps, pred_route, _ = outputs
        if self.target_mode in ("auto", "route") and pred_route is not None:
            return pred_route.float().abs().sum()
        if self.target_mode in ("auto", "speed") and pred_speed_wps is not None:
            return pred_speed_wps.float().abs().sum()
        return None

    def _preprocess_image(self, image_path: Path):
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
    parser = argparse.ArgumentParser(description="Sim-Lingo inference baseline runner")
    parser.add_argument("--config", type=Path, required=True, help="Path to Hydra config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--scene_dir", type=Path, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store outputs")
    parser.add_argument("--device", type=str, default=None, help="Device identifier, e.g., cuda:0")
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=["auto", "speed", "route"],
        default="auto",
        help="Which prediction head to use for scalar target construction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runner = SimLingoInferenceBaseline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        target_mode=args.target_mode,
    )
    runner.run_scene(args.scene_dir, args.output_dir)


if __name__ == "__main__":
    main()
