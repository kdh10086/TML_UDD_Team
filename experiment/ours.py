#!/usr/bin/env python3
"""Generic Attention Explainability runner specialized for Sim-Lingo action mode.

Sim-Lingo InternVL2 추론 코드에서 액션 헤드(kinematic metric) 기준으로 backward된
어텐션/그래디언트를 활용해 Chefer rule5/6 Generic Attention을 적용, 이미지 히트맵을 생성한다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from experiment.overlay_utils import overlay_trajectories, resolve_overlay_dirs
from experiment.simlingo_inference_baseline import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENE_DIR,
    SimLingoInferenceBaseline,
)


def avg_heads(cam: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    if grad is None:
        raise RuntimeError("Gradient tensor is required to compute Generic Attention CAM.")
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def apply_self_attention_rules(R_ss: torch.Tensor, cam_ss: torch.Tensor) -> torch.Tensor:
    return torch.matmul(cam_ss, R_ss)


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, colormap_code: int, alpha: float) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap_code)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    cam = (1 - alpha) * np.float32(img) + alpha * heatmap
    cam = np.clip(cam, 0, 1)
    return cam


class GenericAttentionActionVisualizer(SimLingoInferenceBaseline):
    """Sim-Lingo 액션 모드 기반 Generic Attention 히트맵 생성기."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: Optional[str] = None,
        colormap: str = "JET",
        alpha: float = 0.5,
        trajectory_overlay_root: Optional[Path] = None,
        payload_root: Optional[Path] = None,
    ) -> None:
        super().__init__(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            target_mode="auto",
            explain_mode="action",
        )
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
        self.payload_root = self._resolve_payload_root(payload_root, trajectory_overlay_root)
        self._payload_index = self._index_payloads(self.payload_root)

    def generate_scene_heatmaps(self, scene_dir: Path, output_dir: Path, suffix: str = "generic_action") -> None:
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        scenario_output_dir = self._prepare_output_subdir(output_dir, scene_dir, suffix)
        images_dir = scene_dir / "images"
        image_root = images_dir if images_dir.exists() else scene_dir
        image_paths = sorted(
            [p for p in image_root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        route_dir, speed_dir = resolve_overlay_dirs(image_root, self.trajectory_overlay_root)
        for image_path in image_paths:
            self._process_single_image(image_path, scenario_output_dir, suffix, route_dir, speed_dir)

    def _process_single_image(
        self,
        image_path: Path,
        output_dir: Path,
        suffix: str,
        route_overlay_dir: Optional[Path],
        speed_overlay_dir: Optional[Path],
    ) -> Path:
        record_tag = image_path.stem
        cached_payload = self._load_cached_payload(record_tag)
        if cached_payload is not None:
            return self._process_cached_payload(
                cached_payload, image_path, output_dir, suffix, route_overlay_dir, speed_overlay_dir
            )

        self.recorder.start_recording(record_tag)
        driving_input, meta, prompt_token_ids = self._prepare_driving_input_with_prompt(image_path)
        outputs = self.model(driving_input)
        target_scalar, target_meta = self._compute_action_target(outputs)
        if target_scalar is None:
            raise RuntimeError("Failed to build scalar target for action mode Generic Attention.")
        target_scalar.backward(retain_graph=False)
        attention_maps = self.recorder.stop_recording()
        self.model.zero_grad(set_to_none=True)
        relevance = self._compute_language_relevance(attention_maps)
        source_index = len(prompt_token_ids) - 1
        seq_len = relevance.shape[0]
        if source_index >= seq_len:
            raise IndexError(f"Target token index ({source_index}) exceeds recorded attention sequence ({seq_len}).")
        image_token_positions = self._select_image_token_positions(prompt_token_ids, meta["num_total_image_tokens"])
        token_scores = relevance[source_index, image_token_positions]
        heatmap = self._scores_to_heatmap(token_scores, meta)
        overlay = self._render_overlay(image_path, heatmap, record_tag, route_overlay_dir, speed_overlay_dir)
        output_path = output_dir / f"{image_path.stem}_{suffix}.png"
        Image.fromarray(overlay).save(output_path)
        return output_path

    def _process_cached_payload(
        self,
        payload: Dict[str, Any],
        image_path: Path,
        output_dir: Path,
        suffix: str,
        route_overlay_dir: Optional[Path],
        speed_overlay_dir: Optional[Path],
    ) -> Path:
        attention_maps = payload.get("attention") or {}
        target_info = payload.get("target_info", {})
        if not attention_maps or target_info.get("type") != "action":
            raise RuntimeError("Cached payload is missing attention info for action-mode Generic Attention.")
        meta = payload["meta"]
        prompt_token_ids = self._rebuild_prompt_token_ids(meta, payload.get("input_speed_mps", 0.0))
        moved_attention = self._move_attention_maps_to_device(attention_maps, self.device)
        relevance = self._compute_language_relevance(moved_attention)
        source_index = len(prompt_token_ids) - 1
        seq_len = relevance.shape[0]
        if source_index >= seq_len:
            raise IndexError(f"Target token index ({source_index}) exceeds recorded attention sequence ({seq_len}).")
        image_token_positions = self._select_image_token_positions(prompt_token_ids, meta["num_total_image_tokens"])
        token_scores = relevance[source_index, image_token_positions]
        heatmap = self._scores_to_heatmap(torch.tensor(token_scores, device=self.device), meta)
        overlay = self._render_overlay(image_path, heatmap, image_path.stem, route_overlay_dir, speed_overlay_dir)
        output_path = output_dir / f"{image_path.stem}_{suffix}.png"
        Image.fromarray(overlay).save(output_path)
        return output_path

    def _scores_to_heatmap(self, scores: torch.Tensor, meta: Dict[str, Any]) -> np.ndarray:
        grid_size = int(math.sqrt(meta["num_total_image_tokens"]))
        heatmap = scores.reshape(grid_size, grid_size).detach().to("cpu").numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        heatmap = cv2.resize(heatmap, (meta["original_width"], meta["original_height"]))
        return heatmap

    def _render_overlay(
        self,
        image_path: Path,
        heatmap: np.ndarray,
        tag: str,
        route_overlay_dir: Optional[Path],
        speed_overlay_dir: Optional[Path],
    ) -> np.ndarray:
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        image_float = np.float32(image) / 255
        cam = show_cam_on_image(image_float, heatmap, self.colormap_code, self.alpha)
        cam = np.uint8(255 * cam)
        cam = overlay_trajectories(cam, tag, route_overlay_dir, speed_overlay_dir)
        return cam

    def _prepare_driving_input_with_prompt(
        self, image_path: Path
    ) -> tuple[DrivingInput, Dict[str, Any], List[int]]:
        driving_input, meta = self._prepare_driving_input(image_path)
        prompt_token_ids = self._extract_prompt_token_ids(driving_input.prompt)
        return driving_input, meta, prompt_token_ids

    @staticmethod
    def _extract_prompt_token_ids(lang_label) -> List[int]:
        valid_mask = lang_label.phrase_valid[0].bool()
        return lang_label.phrase_ids[0][valid_mask].tolist()

    def _compute_language_relevance(
        self, attention_maps: Dict[str, Sequence[Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        language_items = [(name, entries) for name, entries in attention_maps.items() if "language_block" in name]
        if not language_items:
            raise RuntimeError("No language block attention maps were recorded.")
        language_items.sort(key=lambda kv: int(kv[0].split("_")[-1]))
        relevance: Optional[torch.Tensor] = None
        for _, entries in language_items:
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
        return torch.load(payload_path, map_location=self.device)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sim-Lingo Generic Attention (action) heatmap generator.")
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=DEFAULT_SCENE_DIR,
        help="Scene directory containing frames or images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save heatmaps.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Hydra config path.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Checkpoint path.")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., cuda:0).")
    parser.add_argument("--colormap", type=str, default="JET", help="OpenCV colormap name (e.g., JET).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay blending factor.")
    parser.add_argument(
        "--trajectory_overlay_root",
        type=Path,
        default=None,
        help="Root containing route/speed overlay images (optional).",
    )
    parser.add_argument(
        "--payload_root",
        type=Path,
        default=None,
        help="Directory containing cached .pt payloads (optional).",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    viz = GenericAttentionActionVisualizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        colormap=args.colormap,
        alpha=args.alpha,
        trajectory_overlay_root=args.trajectory_overlay_root,
        payload_root=args.payload_root,
    )
    viz.generate_scene_heatmaps(args.scene_dir, args.output_dir, suffix="generic_action")


if __name__ == "__main__":
    main()
