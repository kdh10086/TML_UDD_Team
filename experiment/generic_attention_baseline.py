#!/usr/bin/env python3
"""Generic Attention Explainability runner specialized for Sim-Lingo text mode.

이 스크립트는 Sim-Lingo InternVL2 추론 코드를 그대로 활용하여
텍스트 로짓에 대해 언어 모델 블록의 어텐션/그래디언트를 훅으로 수집하고,
Chefer의 Generic Attention 룰(논문의 rule 5/6)을 적용해 이미지 히트맵을 생성합니다.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from transformers import AutoProcessor

from experiment.overlay_utils import overlay_trajectories, resolve_overlay_dirs
from experiment.simlingo_inference_baseline import DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_SCENE_DIR, TEXT_TOKEN_STRATEGIES
from simlingo_training.utils.custom_types import LanguageLabel


def avg_heads(cam: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """rule 5 — head-wise gradient weighting."""
    if grad is None:
        raise RuntimeError("Gradient tensor is required to compute Generic Attention CAM.")
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def apply_self_attention_rules(R_ss: torch.Tensor, cam_ss: torch.Tensor) -> torch.Tensor:
    """rule 6 — propagate relevance with Ā · R."""
    return torch.matmul(cam_ss, R_ss)


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, colormap_code: int, alpha: float) -> np.ndarray:
    """원본 이미지를 [0,1] 범위로 받고 히트맵을 덮어씌워 시각화를 만든다."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap_code)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    cam = (1 - alpha) * np.float32(img) + alpha * heatmap
    cam = np.clip(cam, 0, 1)
    return cam


class GenericAttentionTextVisualizer:
    """Sim-Lingo 텍스트 모드 기반 Generic Attention 히트맵 생성기 (캐시 전용)."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        device: Optional[str] = None,
        text_token_strategy: str = "max",
        text_token_index: int = -1,
        colormap: str = "JET",
        alpha: float = 0.5,
        trajectory_overlay_root: Optional[Path] = None,
        payload_root: Optional[Path] = None,
    ) -> None:
        if payload_root is None:
            # Will attempt to derive from scene_dir later or raise error if scene_dir is also None
            pass
        else:
            self.payload_root = self._resolve_payload_root(payload_root, trajectory_overlay_root)
        if text_token_strategy not in TEXT_TOKEN_STRATEGIES:
            raise ValueError(f"text_token_strategy must be one of {TEXT_TOKEN_STRATEGIES}")
        self.text_token_strategy = text_token_strategy
        self.text_token_index = text_token_index
        self.config_path = Path(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = OmegaConf.load(self.config_path)
        self.processor = AutoProcessor.from_pretrained(self.cfg.model.vision_model.variant, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        self.tokenizer.padding_side = "left"
        self.num_image_token = None
        cmap_name = colormap.upper()
        attr_name = f"COLORMAP_{cmap_name}"
        if not hasattr(cv2, attr_name):
            raise ValueError(f"Unsupported OpenCV colormap: {colormap}")
        self.colormap_code = getattr(cv2, attr_name)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.trajectory_overlay_root = trajectory_overlay_root
        
        # payload_root resolution moved to generate_scene_heatmaps or handled lazily
        self.explicit_payload_root = payload_root
        self._payload_index = {} # Will be populated in generate_scene_heatmaps

    def generate_scene_heatmaps(
        self,
        scene_dir: Path,
        output_dir: Path,
        suffix: str = "generic_text",
        raw_output_dir: Optional[Path] = None,
    ) -> None:
        """scene_dir 내 pt 파일들에 대해 히트맵을 생성하고 저장한다."""
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        scenario_output_dir = self._prepare_output_subdir(output_dir, scene_dir, suffix)
        scenario_raw_output_dir = None
        if raw_output_dir:
            raw_output_dir = Path(raw_output_dir)
            raw_output_dir.mkdir(parents=True, exist_ok=True)
            scenario_raw_output_dir = self._prepare_output_subdir(raw_output_dir, scene_dir, "raw")
        
        # Resolve payload root
        if self.explicit_payload_root:
            self.payload_root = self._resolve_payload_root(self.explicit_payload_root, self.trajectory_overlay_root)
        else:
            # Default to scene_dir/pt
            potential_pt = scene_dir / "pt"
            if potential_pt.exists():
                self.payload_root = potential_pt
            else:
                # Fallback to scene_dir itself
                self.payload_root = scene_dir
        
        if not self.payload_root or not self.payload_root.exists():
             raise FileNotFoundError(f"Could not locate payload directory in {scene_dir} or specified root.")

        print(f"Loading payloads from: {self.payload_root}")
        self._payload_index = self._index_payloads(self.payload_root)
        
        if not self._payload_index:
            print(f"No .pt files found in {self.payload_root}")
            return

        # Robust image directory search
        candidates = [scene_dir / "input_images", scene_dir / "video_garmin", scene_dir / "images", scene_dir]
        image_root = scene_dir
        print(f"[DEBUG] Searching for images in candidates: {candidates}")
        for cand in candidates:
            print(f"[DEBUG] Checking candidate: {cand}")
            if cand.exists() and cand.is_dir():
                # Check if it has images
                has_images = False
                for p in cand.iterdir():
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        has_images = True
                        break
                print(f"[DEBUG] Candidate {cand} exists. Has images? {has_images}")
                if has_images:
                    image_root = cand
                    break
            else:
                print(f"[DEBUG] Candidate {cand} does not exist or is not a directory.")
        
        print(f"Using image root: {image_root}")
        route_dir, speed_dir = resolve_overlay_dirs(image_root, self.trajectory_overlay_root)

        # Iterate over payloads
        for tag, pt_path in sorted(self._payload_index.items()):
            # Find corresponding image
            image_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                cand = image_root / f"{tag}{ext}"
                if cand.exists():
                    image_path = cand
                    break
            
            if image_path is None:
                print(f"Warning: Image not found for tag {tag} in {image_root}. Skipping.")
                continue

            self._process_single_image(
                image_path, pt_path, scenario_output_dir, suffix, route_dir, speed_dir, scenario_raw_output_dir
            )

    @staticmethod
    def _prepare_output_subdir(output_root: Path, scene_dir: Path, suffix: str) -> Path:
        scenario_name = scene_dir.name
        base = f"{scenario_name}_{suffix}"
        candidate = output_root / base
        counter = 1
        while candidate.exists():
            candidate = output_root / f"{base}_{counter}"
            counter += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _process_single_image(
        self,
        image_path: Path,
        pt_path: Path,
        output_dir: Path,
        suffix: str,
        route_overlay_dir: Optional[Path],
        speed_overlay_dir: Optional[Path],
        raw_output_dir: Optional[Path] = None,
    ) -> Path:
        record_tag = image_path.stem
        # Load directly from pt_path
        cached_payload = torch.load(pt_path, map_location="cpu")
        
        # Validate payload
        if cached_payload.get("mode") != "text":
             print(f"[DEBUG] Skipping {pt_path.name}: mode is '{cached_payload.get('mode')}', expected 'text'")
             return None
        if not cached_payload.get("attention"):
             print(f"[DEBUG] Skipping {pt_path.name}: 'attention' key missing or empty")
             return None
        if cached_payload.get("text_outputs") is None:
             print(f"[DEBUG] Skipping {pt_path.name}: 'text_outputs' key missing")
             return None

        return self._process_cached_payload(
            cached_payload,
            image_path,
            output_dir,
            suffix,
            route_overlay_dir,
            speed_overlay_dir,
            raw_output_dir,
        )

    def _process_cached_payload(
        self,
        payload: Dict[str, Any],
        image_path: Path,
        output_dir: Path,
        suffix: str,
        route_overlay_dir: Optional[Path],
        speed_overlay_dir: Optional[Path],
        raw_output_dir: Optional[Path] = None,
    ) -> Path:
        attention_maps = payload.get("attention") or {}
        text_outputs = payload.get("text_outputs")
        target_info = payload.get("target_info", {})
        if not attention_maps or text_outputs is None or target_info.get("type") != "text":
            raise RuntimeError("Cached payload is missing attention/text info for text-mode Generic Attention.")
        meta = payload["meta"]
        self.num_image_token = int(meta["num_image_tokens_per_patch"])
        prompt_token_ids = self._rebuild_prompt_token_ids(meta, payload.get("input_speed_mps", 0.0))
        moved_attention = self._move_attention_maps_to_device(attention_maps, self.device)
        token_ids = torch.tensor(text_outputs["token_ids"], device=self.device)
        token_scores = torch.tensor(text_outputs["token_scores"], device=self.device)
        text_features = {
            "token_ids": token_ids,
            "token_scores": token_scores,
            "token_strings": text_outputs.get("token_strings", []),
            "decoded_text": text_outputs.get("decoded_text", ""),
        }
        relevance = self._compute_language_relevance(moved_attention)
        prompt_len = len(prompt_token_ids)
        generated_len = len(text_outputs.get("token_ids", []))
        seq_len = relevance.shape[0]
        if seq_len < prompt_len + generated_len:
            raise RuntimeError(
                f"Recorded sequence ({seq_len}) shorter than prompt+generated tokens ({prompt_len + generated_len})."
            )
        token_index = int(target_info.get("token_index", generated_len - 1))
        source_index = prompt_len + token_index
        if source_index >= seq_len:
            raise IndexError(
                f"Target token index ({source_index}) exceeds recorded attention sequence ({seq_len})."
            )
        image_token_positions = self._select_image_token_positions(
            prompt_token_ids, meta["num_total_image_tokens"]
        )
        token_scores = relevance[source_index, image_token_positions]
        heatmap = self._scores_to_heatmap(token_scores, meta)
        
        if raw_output_dir:
            heatmap_uint8 = np.uint8(255 * heatmap)
            raw_path = raw_output_dir / f"{image_path.stem}.png"
            Image.fromarray(heatmap_uint8).save(raw_path)

        # Save Overlay
        overlay = self._render_overlay(image_path, heatmap, image_path.stem, route_overlay_dir, speed_overlay_dir)
        output_path = output_dir / f"{image_path.stem}_{suffix}.png"
        Image.fromarray(overlay).save(output_path)
        return output_path

    def _prepare_driving_input_with_prompt(
        self, image_path: Path
    ) -> Tuple[DrivingInput, Dict[str, int], List[int]]:
        """언어 토큰 시퀀스를 추가로 회수하기 위한 전처리."""
        processed_image, num_patches, orig_hw = self._preprocess_image(image_path)
        placeholder_batch_list: List[dict] = []
        prompt_str = "Current speed: 0 m/s. Predict the waypoints."
        lang_label = self._build_language_label(prompt_str, placeholder_batch_list, num_patches)
        prompt_token_ids = self._extract_prompt_token_ids(lang_label)
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
        }
        return driving_input, meta, prompt_token_ids

    @staticmethod
    def _extract_prompt_token_ids(lang_label) -> List[int]:
        valid_mask = lang_label.phrase_valid[0].bool()
        return lang_label.phrase_ids[0][valid_mask].tolist()

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
        return LanguageLabel(
            phrase_ids=prompt_tokenized_ids.to(self.device),
            phrase_valid=prompt_tokenized_valid.to(self.device),
            phrase_mask=prompt_tokenized_mask.to(self.device),
            placeholder_values=placeholder_batch_list,
            language_string=prompt_batch_list,
            loss_masking=None,
        )

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

    def _compute_language_relevance(
        self, attention_maps: Dict[str, Sequence[Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        import re
        # Filter for language block attention keys
        # Supports keys like "language_block_0", "language_block_0_self_attn", etc.
        language_items = []
        for name, entries in attention_maps.items():
            if "language_block" in name:
                # Extract layer index
                match = re.search(r"language_block_(\d+)", name)
                if match:
                    layer_idx = int(match.group(1))
                    # Prefer _self_attn if available, or just the block
                    # We store tuple (index, is_self_attn, name, entries) to sort later
                    is_self_attn = 1 if "self_attn" in name else 0
                    language_items.append((layer_idx, is_self_attn, name, entries))

        if not language_items:
            raise RuntimeError("No language block attention maps were recorded.")
        
        # Sort by layer index, then by is_self_attn (prefer self_attn if duplicates exist for same layer?)
        # Actually we might want to filter out non-self-attn if self-attn exists.
        # But for now, let's just sort by layer.
        language_items.sort(key=lambda x: (x[0], x[1]))
        
        # If we have multiple entries for the same layer (e.g. block_0 and block_0_self_attn),
        # we should probably pick the most specific one (self_attn).
        # Let's group by layer index.
        from collections import defaultdict
        layer_map = defaultdict(list)
        for idx, is_self, name, entries in language_items:
            layer_map[idx].append((is_self, entries))
        
        sorted_layers = sorted(layer_map.keys())
        relevance: Optional[torch.Tensor] = None
        
        for idx in sorted_layers:
            # Pick the best entry for this layer. Max is_self_attn means we prefer self_attn.
            best_entry = max(layer_map[idx], key=lambda x: x[0])
            entries = best_entry[1]
            attn_list = [entry["attn"] for entry in entries if entry.get("attn") is not None]
            grad_list = [entry["grad"] for entry in entries if entry.get("grad") is not None]
            if not attn_list or not grad_list:
                continue
            attn = torch.stack(attn_list, dim=0).mean(dim=0)
            grad = torch.stack(grad_list, dim=0).mean(dim=0)
            cam = avg_heads(attn, grad)
            if relevance is None:
                num_tokens = cam.shape[-1]
                relevance = torch.eye(num_tokens, num_tokens, dtype=cam.dtype, device=cam.device)
            relevance = relevance + apply_self_attention_rules(relevance, cam)
        if relevance is None:
            raise RuntimeError("Unable to build relevance due to empty attention/gradient pairs.")
        return relevance

    def _select_image_token_positions(self, prompt_token_ids: List[int], expected: int) -> List[int]:
        positions = [idx for idx, token_id in enumerate(prompt_token_ids) if token_id == self.img_context_token_id]
        if len(positions) < expected:
            raise RuntimeError(
                f"Located {len(positions)} <IMG_CONTEXT> tokens, but expected {expected} image tokens."
            )
        return positions[:expected]

    @staticmethod
    def _move_attention_maps_to_device(
        attention: Dict[str, Sequence[Dict[str, torch.Tensor]]], device: Optional[torch.device] = None
    ) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        device = device or torch.device("cpu")
        moved: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        for name, entries in attention.items():
            moved[name] = []
            for entry in entries:
                moved_entry = {
                    "attn": entry["attn"].to(device) if entry.get("attn") is not None else None,
                    "grad": entry["grad"].to(device) if entry.get("grad") is not None else None,
                }
                moved[name].append(moved_entry)
        return moved

    def _rebuild_prompt_token_ids(self, meta: Dict[str, Any], current_speed: float) -> List[int]:
        """기존 추론과 동일한 프롬프트로 <IMG_CONTEXT> 위치를 복원한다."""
        placeholder_batch_list: List[dict] = []
        prompt_str = (
            f"Current speed: {current_speed:.1f} m/s. What should the ego vehicle do next? Provide a short commentary."
        )
        lang_label = self._build_language_label(prompt_str, placeholder_batch_list, meta["num_patch_views"])
        return self._extract_prompt_token_ids(lang_label)

    @staticmethod
    def _resolve_payload_root(
        explicit: Optional[Path], overlay_root: Optional[Path]
    ) -> Optional[Path]:
        candidates: List[Path] = []

        def add_base(path_like: Optional[Path]) -> None:
            if path_like is None:
                return
            base = Path(path_like)
            candidates.extend([base, base / "pt"])
            if base.name in {"route_overlay", "speed_overlay", "pt"}:
                candidates.extend([base.parent, base.parent / "pt"])

        add_base(explicit)
        add_base(overlay_root)

        for cand in candidates:
            if cand is None:
                continue
            if cand.exists() and cand.is_dir() and any(cand.glob("*.pt")):
                return cand
        return None

    @staticmethod
    def _index_payloads(payload_root: Optional[Path]) -> Dict[str, Path]:
        if payload_root is None:
            return {}
        return {p.stem: p for p in Path(payload_root).glob("*.pt")}

    def _load_cached_payload(self, tag: str) -> Optional[Dict[str, Any]]:
        if not self._payload_index:
            return None
        payload_path = self._payload_index.get(tag)
        if payload_path is None:
            return None
        payload = torch.load(payload_path, map_location="cpu")
        if payload.get("mode") != "text":
            return None
        if not payload.get("attention"):
            return None
        if payload.get("text_outputs") is None:
            return None
        return payload

    def _scores_to_heatmap(self, scores: torch.Tensor, meta: Dict[str, Any]) -> np.ndarray:
        total_tokens = meta["num_total_image_tokens"]
        orig_h = meta["original_height"]
        orig_w = meta["original_width"]
        
        # Attempt to find best H, W such that H*W = total_tokens and H/W ~ orig_h/orig_w
        best_h, best_w = int(math.sqrt(total_tokens)), int(math.sqrt(total_tokens))
        min_error = float("inf")
        target_ratio = orig_w / orig_h
        
        for h in range(1, int(math.sqrt(total_tokens)) + 1):
            if total_tokens % h == 0:
                w = total_tokens // h
                # Check pair (h, w)
                ratio = w / h
                error = abs(ratio - target_ratio)
                if error < min_error:
                    min_error = error
                    best_h, best_w = h, w
                
                # Check pair (w, h)
                ratio_inv = h / w
                error_inv = abs(ratio_inv - target_ratio)
                if error_inv < min_error:
                    min_error = error_inv
                    best_h, best_w = w, h

        heatmap = scores.reshape(best_h, best_w).detach().float().to("cpu").numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        heatmap = cv2.resize(heatmap, (orig_w, orig_h))
        return heatmap

    def _render_overlay(
        self,
        image_path: Path,
        heatmap: np.ndarray,
        record_tag: str,
        route_overlay_dir: Optional[Path],
        speed_overlay_dir: Optional[Path],
    ) -> np.ndarray:
        heatmap_np = heatmap
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        blended = show_cam_on_image(image, heatmap_np, self.colormap_code, self.alpha)
        blended = overlay_trajectories(blended, record_tag, route_overlay_dir, speed_overlay_dir)
        return np.uint8(255 * blended)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sim-Lingo text-mode Generic Attention heatmap generator (cache only)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Hydra config path.")
    parser.add_argument("--scene_dir", type=Path, default=None, help="Optional directory with scene images.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save heatmaps.")
    parser.add_argument(
        "--text_token_strategy",
        type=str,
        choices=list(TEXT_TOKEN_STRATEGIES),
        default="max",
        help="Strategy for selecting text logit to backpropagate.",
    )
    parser.add_argument(
        "--text_token_index",
        type=int,
        default=-1,
        help="Token index when --text_token_strategy=index.",
    )
    parser.add_argument("--colormap", type=str, default="JET", help="OpenCV colormap name (e.g., JET, VIRIDIS).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend ratio between original image and heatmap.")
    parser.add_argument("--suffix", type=str, default="generic_text", help="Output filename suffix.")
    parser.add_argument(
        "--trajectory_overlay_root",
        type=Path,
        default=None,
        help="Path to Sim-Lingo inference output (expects route_overlay/speed_overlay subdirs).",
    )
    parser.add_argument(
        "--payload_root",
        type=Path,
        default=None,
        help="Path to cached Sim-Lingo inference pt directory (optional, defaults to scene_dir/pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = GenericAttentionTextVisualizer(
        config_path=args.config,
        text_token_strategy=args.text_token_strategy,
        text_token_index=args.text_token_index,
        colormap=args.colormap,
        alpha=args.alpha,
        trajectory_overlay_root=args.trajectory_overlay_root,
        payload_root=args.payload_root,
    )
    runner.generate_scene_heatmaps(args.scene_dir, args.output_dir, suffix=args.suffix)


if __name__ == "__main__":
    main()
