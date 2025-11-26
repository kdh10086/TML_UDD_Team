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
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
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
        self.residual_alpha = 0.5  # Default from ours.py
        self.propagation_mode = "llm_to_vision"
        self.cam_softmax = False
        self.cam_temperature = 1.0
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
        final_output_dir: Optional[Path] = None,
        target_files: Optional[List[Path]] = None,
    ) -> None:
        """scene_dir 내 pt 파일들에 대해 히트맵을 생성하고 저장한다."""
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use final_output_dir if provided, otherwise default to output_dir/final
        if final_output_dir is None:
            final_output_dir = output_dir / "final"
            final_output_dir.mkdir(parents=True, exist_ok=True)
        if raw_output_dir:
            raw_output_dir = Path(raw_output_dir)
            raw_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Resolve payload root
        # Resolve payload root
        if self.payload_root and self.payload_root.exists():
            # Use already set payload_root (e.g. from pipeline)
            pass
        elif self.explicit_payload_root:
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
        
        if image_root is None:
             print(f"Warning: Could not find any image directory in {scene_dir}")
             return

        print(f"Using image root: {image_root}")
        route_dir, speed_dir = resolve_overlay_dirs(image_root, self.trajectory_overlay_root)

        # pt_log.txt goes in output_dir (method_dir)
        pt_log_path = output_dir / "pt_log.txt"

        # If target_files is provided, we iterate over them.
        # But wait, ViT scripts iterate over PAYLOADS, not images.
        # "for tag, payload_path in sorted(self._payload_index.items()):"
        # We need to filter payloads based on target_files (images).
        
        if target_files:
            target_stems = {p.stem for p in target_files}
            # Filter payload index
            filtered_index = {k: v for k, v in self._payload_index.items() if k in target_stems}
            items_to_process = sorted(filtered_index.items())
        else:
            items_to_process = sorted(self._payload_index.items())

        with open(pt_log_path, "a") as log_file:
            for tag, pt_path in tqdm(items_to_process, desc=f"GenericText ({scene_dir.name if scene_dir else '?'})", unit="img"):
                # Find corresponding image
                image_path = None
                # Try exact match first
                for cand_img in image_root.iterdir():
                    if cand_img.stem == tag and cand_img.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        image_path = cand_img
                        break
                
                if image_path is None:
                    # Try fuzzy match if needed (not implemented here for simplicity)
                    pass

                if image_path is None:
                    print(f"Warning: Image not found for payload {tag}. Skipping.")
                    continue
                
                # Log the PT file usage (simple format: image_name - pt_filename)
                log_file.write(f"{tag} - {pt_path.name}\n")
                
                self._process_single_image(
                    image_path, pt_path, final_output_dir, suffix, route_dir, speed_dir, raw_output_dir
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
        interleaver = payload.get("interleaver") or {}
        target_info = payload.get("target_info", {})
        if not attention_maps or text_outputs is None or target_info.get("type") != "text":
            raise RuntimeError("Cached payload is missing attention/text info for text-mode Generic Attention.")
        meta = payload["meta"]
        self.num_image_token = int(meta["num_image_tokens_per_patch"])
        prompt_token_ids = self._rebuild_prompt_token_ids(meta, payload.get("input_speed_mps", 0.0))
        moved_attention = self._move_attention_maps_to_device(attention_maps, self.device)
        
        # Compute LLM relevance
        relevance = self._compute_language_relevance(moved_attention)
        
        # Determine source token index
        prompt_len = len(prompt_token_ids)
        generated_len = len(text_outputs.get("token_ids", []))
        seq_len = relevance.shape[0]
        if seq_len < prompt_len + generated_len:
            print(
                f"[WARN] seq_len {seq_len} < prompt+gen {prompt_len + generated_len}; clipping indices for {image_path.name}"
            )
            prompt_len = min(prompt_len, seq_len)
            generated_len = max(0, min(generated_len, seq_len - prompt_len))
        token_index = int(target_info.get("token_index", generated_len - 1))
        token_index = max(0, min(token_index, max(generated_len - 1, 0)))
        source_index = min(prompt_len + token_index, seq_len - 1)
        
        image_token_positions = [
            pos for pos in self._select_image_token_positions(prompt_token_ids, meta["num_total_image_tokens"])
            if pos < seq_len
        ]
        if not image_token_positions:
            raise RuntimeError("No image token positions within recorded sequence.")
            
        # language-side relevance -> image tokens
        token_scores = relevance[source_index, image_token_positions].clone().detach().to(self.device)

        # If interleaver info exists, project back to ViT patch grid
        grid_hw: Optional[Tuple[int, int]] = None
        if interleaver:
            patch_scores_list = self._project_interleaver_relevance(token_scores, interleaver)
            if not patch_scores_list:
                print(f"[WARN] Interleaver projection failed for {image_path.name}; falling back to token scores.")
                patch_scores_list = [(token_scores, grid_hw)]
        else:
            patch_scores_list = [(token_scores, grid_hw)]

        # If vision attentions exist, propagate through vision stack
        outputs: List[Path] = []
        vision_attn = {k: v for k, v in moved_attention.items() if "vision_attn" in k or "vision_block" in k}

        for idx, (patch_scores, grid_hw_val) in enumerate(patch_scores_list):
            vision_scores = patch_scores
            if vision_attn and patch_scores is not None:
                try:
                    vision_scores = self._compute_vision_relevance(vision_attn, patch_scores)
                except Exception as exc:
                    print(f"[WARN] Vision relevance failed for {image_path.name} slice {idx}: {exc}")
                    vision_scores = patch_scores

            # Determine actual image size for alignment
            img_arr = cv2.imread(str(image_path))
            actual_size = (img_arr.shape[1], img_arr.shape[0]) if img_arr is not None else None

            heatmap_source = vision_scores.clone().detach().to(self.device) if vision_scores is not None else token_scores
            heatmap = self._scores_to_heatmap(heatmap_source, meta, grid_hw=grid_hw_val, target_size=actual_size)
            
            if raw_output_dir:
                heatmap_uint8 = np.uint8(255 * heatmap)
                raw_name = f"{image_path.stem}.png" if len(patch_scores_list) == 1 else f"{image_path.stem}_img{idx}.png"
                raw_path = raw_output_dir / raw_name
                Image.fromarray(heatmap_uint8).save(raw_path)

            overlay_name = image_path.stem if len(patch_scores_list) == 1 else f"{image_path.stem}_img{idx}"
            overlay = self._render_overlay(image_path, heatmap, overlay_name, route_overlay_dir, speed_overlay_dir)
            output_path = output_dir / f"{overlay_name}.png"
            Image.fromarray(overlay).save(output_path)
            outputs.append(output_path)

        return outputs[0] if outputs else output_dir / f"{image_path.stem}.png"

    def _compute_vision_relevance(
        self,
        attention_maps: Dict[str, Sequence[Dict[str, torch.Tensor]]],
        init_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate patch-level relevance through vision attention stack (with residual_alpha)."""
        import re
        vision_items = []
        for name, entries in attention_maps.items():
            if "vision_attn" in name or "vision_block" in name:
                match = re.search(r"vision_(?:attn|block)_?(\d+)", name)
                layer_idx = int(match.group(1)) if match else 0
                vision_items.append((layer_idx, name, entries))
        if not vision_items:
            return init_scores
        # Sort from last to first (top-down relevance)
        vision_items.sort(key=lambda x: x[0], reverse=True)
        relevance_vec = init_scores
        for _, _, entries in vision_items:
            attn_list = [e["attn"] for e in entries if e.get("attn") is not None]
            grad_list = [e["grad"] for e in entries if e.get("grad") is not None]
            if not attn_list or not grad_list:
                continue
            min_len = min(t.shape[-1] for t in attn_list)
            attn_list = [t[..., :min_len, :min_len] for t in attn_list]
            grad_list = [g[..., :min_len, :min_len] for g in grad_list]
            attn = torch.stack(attn_list, dim=0).mean(dim=0)
            grad = torch.stack(grad_list, dim=0).mean(dim=0)
            cam = avg_heads(attn, grad)  # [S, S]
            if self.cam_softmax:
                cam = torch.softmax(cam / max(self.cam_temperature, 1e-6), dim=-1)
            if relevance_vec.shape[-1] != cam.shape[-1]:
                # Trim/pad relevance vector
                target_len = cam.shape[-1]
                if relevance_vec.shape[-1] > target_len:
                    relevance_vec = relevance_vec[..., :target_len]
                else:
                    pad = torch.zeros(target_len - relevance_vec.shape[-1], device=relevance_vec.device, dtype=relevance_vec.dtype)
                    relevance_vec = torch.cat([relevance_vec, pad], dim=-1)
            relevance_vec = self.residual_alpha * relevance_vec + torch.matmul(cam, relevance_vec)
        return relevance_vec

    def _project_interleaver_relevance(
        self,
        token_scores: torch.Tensor,
        interleaver: Dict[str, Any],
    ) -> List[Tuple[torch.Tensor, Optional[Tuple[int, int]]]]:
        """Project image-token relevance back to ViT patch grid using interleaver metadata and grads."""
        # Move tensors to device
        def _to_device(entry):
            if entry is None:
                return None
            if isinstance(entry, dict) and "value" in entry:
                return entry["value"].to(self.device)
            if torch.is_tensor(entry):
                return entry.to(self.device)
            return entry

        pixel_shuffle_out = _to_device(interleaver.get("pixel_shuffle_out"))
        pixel_shuffle_grad = None
        if isinstance(interleaver.get("pixel_shuffle_out"), dict):
            pixel_shuffle_grad = _to_device(interleaver.get("pixel_shuffle_out").get("grad"))
        tokens_per_patch = interleaver.get("tokens_per_patch")
        num_patches_list = interleaver.get("num_patches_list")
        selected_mask = _to_device(interleaver.get("selected_mask"))
        mlp1_output_grad = None
        if isinstance(interleaver.get("mlp1_output"), dict):
            mlp1_output_grad = _to_device(interleaver.get("mlp1_output").get("grad"))
            # Flatten batch dimension if present (e.g. [B, N, C] -> [B*N, C])
            if mlp1_output_grad is not None and mlp1_output_grad.dim() == 3:
                mlp1_output_grad = mlp1_output_grad.reshape(-1, mlp1_output_grad.shape[-1])
        mlp1_weight = interleaver.get("mlp1_weight")
        mlp1_bias = interleaver.get("mlp1_bias")

        # If multiple images are present, split by selected_mask counts or num_patches_list.
        image_offsets: List[int] = []
        if selected_mask is not None and selected_mask.dim() == 2:
            # selected_mask: [B, N]; count image tokens per image
            image_token_counts = selected_mask.sum(dim=1).tolist()
            prefix = 0
            for cnt in image_token_counts:
                image_offsets.append(prefix)
                prefix += int(cnt)
        elif num_patches_list:
            # tokens_per_patch is needed to map patches->tokens
            if tokens_per_patch and tokens_per_patch > 0:
                prefix = 0
                for p in num_patches_list:
                    image_offsets.append(prefix)
                    prefix += int(p) * int(tokens_per_patch)
        else:
            image_offsets = [0]

        results: List[Tuple[torch.Tensor, Optional[Tuple[int, int]]]] = []
        num_images = len(image_offsets)
        for idx in range(num_images):
            start = image_offsets[idx]
            end = image_offsets[idx + 1] if idx + 1 < num_images else token_scores.numel()
            scores = token_scores[start:end]

            # If mlp1_output_grad and weight are available, backprop relevance to input space (Chefer-style)
            if mlp1_output_grad is not None and mlp1_weight is not None and mlp1_output_grad.shape[0] >= end:
                grad_slice = mlp1_output_grad[start:end]  # [T, D_out]
                # grad_slice @ W -> [T, D_in]; use norm as scalar weight per token
                try:
                    grad_w = torch.matmul(grad_slice, mlp1_weight.to(self.device))
                    token_weights = grad_w.norm(dim=-1)
                    token_weights = token_weights / (token_weights.max() + 1e-6)
                    if token_weights.numel() == scores.numel():
                        scores = scores * token_weights
                except Exception as exc:
                    print(f"[WARN] mlp1 grad-based weighting failed: {exc}")
            elif mlp1_output_grad is not None:
                # Fallback: grad norm only
                grad_slice = mlp1_output_grad[start:end]
                grad_norm = grad_slice.norm(dim=-1)
                if grad_norm.numel() == scores.numel():
                    grad_norm = grad_norm / (grad_norm.max() + 1e-6)
                    scores = scores * grad_norm

            # Map tokens -> patches by averaging tokens_per_patch
            if tokens_per_patch and tokens_per_patch > 0:
                total_tokens = scores.numel()
                num_patches = total_tokens // tokens_per_patch
                if num_patches == 0:
                    results.append((scores, None))
                    continue
                scores = scores[: num_patches * tokens_per_patch]
                scores = scores.reshape(num_patches, tokens_per_patch).mean(dim=1)
            else:
                if tokens_per_patch is None:
                    print(f"[WARN] tokens_per_patch missing; skipping token->patch aggregation for image {idx}")

            grid_hw = None
            if pixel_shuffle_out is not None:
                # pixel_shuffle_out shape: [B, H, W, C]; pick corresponding image if available
                img_idx = min(idx, pixel_shuffle_out.shape[0] - 1)
                h, w = pixel_shuffle_out.shape[1], pixel_shuffle_out.shape[2]
                grid_hw = (h, w)
                expected_patches = h * w
                if scores.numel() != expected_patches:
                    print(
                        f"[WARN] Image {idx}: patch relevance length ({scores.numel()}) != expected grid size ({expected_patches}); padding/trimming."
                    )
                    if scores.numel() < expected_patches:
                        pad = torch.zeros(expected_patches - scores.numel(), device=scores.device, dtype=scores.dtype)
                        scores = torch.cat([scores, pad], dim=0)
                    else:
                        scores = scores[:expected_patches]
                # If pixel_shuffle grad exists, weight patch relevance
                if pixel_shuffle_grad is not None:
                    grad_img = pixel_shuffle_grad[img_idx]
                    grad_norm = grad_img.norm(dim=-1).flatten()
                    if grad_norm.numel() == scores.numel():
                        grad_norm = grad_norm / (grad_norm.max() + 1e-6)
                        scores = scores * grad_norm
                    else:
                        print(f"[WARN] pixel_shuffle grad size mismatch for image {idx}: {grad_norm.numel()} vs {scores.numel()}")

            results.append((scores, grid_hw))

        return results

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
            # 토큰 길이가 미묘하게 다를 수 있어 스택 전에 최소 길이에 맞춰 자릅니다.
            min_len = min(t.shape[-1] for t in attn_list)
            attn_list = [t[..., :min_len, :min_len] for t in attn_list]
            grad_list = [g[..., :min_len, :min_len] for g in grad_list]
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

    def _scores_to_heatmap(
        self,
        scores: torch.Tensor,
        meta: Dict[str, Any],
        grid_hw: Optional[Tuple[int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        # Use actual token count (may be shorter than expected) and pad to a grid
        total_tokens = int(scores.numel())
        orig_h = meta["original_height"]
        orig_w = meta["original_width"]
        if target_size:
            tgt_w, tgt_h = target_size
            if (orig_w, orig_h) != (tgt_w, tgt_h):
                print(
                    f"[WARN] meta size ({orig_w}x{orig_h}) != actual image size ({tgt_w}x{tgt_h}); using actual size for final resize."
                )
                orig_w, orig_h = tgt_w, tgt_h

        if grid_hw is not None:
            grid_h, grid_w = grid_hw
            if grid_h * grid_w != total_tokens:
                pad = torch.zeros(grid_h * grid_w - total_tokens, device=scores.device, dtype=scores.dtype)
                scores = torch.cat([scores, pad], dim=0)
        else:
            # Approximate grid preserving aspect ratio, allowing padding if not divisible
            target_ratio = orig_h / max(orig_w, 1)
            grid_h = max(1, int(round(math.sqrt(total_tokens * target_ratio))))
            grid_w = max(1, int(math.ceil(total_tokens / grid_h)))
            if grid_h * grid_w < total_tokens:
                grid_h = int(math.ceil(total_tokens / grid_w))
            if grid_h * grid_w != total_tokens:
                pad = torch.zeros(grid_h * grid_w - total_tokens, device=scores.device, dtype=scores.dtype)
                scores = torch.cat([scores, pad], dim=0)

        heatmap = scores.reshape(grid_h, grid_w).detach().float().to("cpu").numpy()
        # Normalization/clipping for better contrast
        if heatmap.max() - heatmap.min() > 1e-9:
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
