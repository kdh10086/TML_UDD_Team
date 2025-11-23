#!/usr/bin/env python3
"""Sim-Lingo vision encoder raw-attention heatmap generator."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from experiment.overlay_utils import overlay_trajectories, resolve_overlay_dirs
from experiment.simlingo_inference_baseline import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENE_DIR,
    SimLingoInferenceBaseline,
)


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, colormap_code: int, alpha: float) -> np.ndarray:
    """Blend 입력 이미지와 히트맵을 결합한다."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap_code)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    cam = (1 - alpha) * img + alpha * heatmap
    return np.clip(cam, 0, 1)


class VisionRawAttention(SimLingoInferenceBaseline):
    """ViT 어텐션 가중치 자체를 이용한 히트맵 시각화."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: Optional[str] = None,
        layer_index: int = -1,
        head_strategy: str = "mean",
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
            enable_vision_hooks=True,
            enable_language_hooks=False,
            skip_backward=True,
        )
        self.layer_index = layer_index
        strategies = {"mean", "max"}
        if head_strategy not in strategies:
            raise ValueError(f"head_strategy must be one of {strategies}")
        self.head_strategy = head_strategy
        cmap_name = colormap.upper()
        attr_name = f"COLORMAP_{cmap_name}"
        if not hasattr(cv2, attr_name):
            raise ValueError(f"Unsupported OpenCV colormap: {colormap}")
        self.colormap_code = getattr(cv2, attr_name)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must lie between 0 and 1.")
        self.alpha = alpha
        self.trajectory_overlay_root = trajectory_overlay_root
        self.payload_root = self._resolve_payload_root(payload_root, trajectory_overlay_root)
        self._payload_index = self._index_payloads(self.payload_root)

    def generate_scene_heatmaps(self, scene_dir: Path, output_dir: Path, suffix: str = "vit_raw") -> None:
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
        driving_input, meta = self._prepare_driving_input(image_path)
        _ = self.model(driving_input)
        attention_maps = self.recorder.stop_recording()
        self.model.zero_grad(set_to_none=True)
        attention_tensor = self._select_layer_attention(attention_maps)
        token_scores = self._extract_image_scores(attention_tensor, meta["num_total_image_tokens"])
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
        meta = payload.get("meta")
        if not attention_maps or meta is None:
            raise RuntimeError("Cached payload is missing attention/meta for raw attention rendering.")
        moved = self._move_attention_maps_to_device(attention_maps, self.device)
        attention_tensor = self._select_layer_attention(moved)
        token_scores = self._extract_image_scores(attention_tensor, meta["num_total_image_tokens"])
        heatmap = self._scores_to_heatmap(token_scores, meta)
        overlay = self._render_overlay(image_path, heatmap, image_path.stem, route_overlay_dir, speed_overlay_dir)
        output_path = output_dir / f"{image_path.stem}_{suffix}.png"
        Image.fromarray(overlay).save(output_path)
        return output_path

    def _collect_vision_matrices(
        self, attention_maps: Dict[str, Sequence[Dict[str, torch.Tensor]]]
    ) -> List[torch.Tensor]:
        items = [(name, entries) for name, entries in attention_maps.items() if "vision_block" in name]
        if not items:
            raise RuntimeError("No vision block attention maps were recorded.")
        items.sort(key=lambda kv: int(kv[0].split("_")[-1]))
        matrices: List[torch.Tensor] = []
        for _, entries in items:
            tensors = [entry["attn"] for entry in entries if entry.get("attn") is not None]
            if not tensors:
                continue
            stacked = torch.stack(tensors, dim=0).mean(dim=0)
            if stacked.dim() == 3:
                stacked = stacked.unsqueeze(0)
            matrices.append(stacked)
        if not matrices:
            raise RuntimeError("Vision attention tensors were empty after aggregation.")
        return matrices

    def _select_layer_attention(
        self, attention_maps: Dict[str, Sequence[Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        matrices = self._collect_vision_matrices(attention_maps)
        idx = self.layer_index
        if idx < 0:
            idx = len(matrices) + idx
        if idx < 0 or idx >= len(matrices):
            raise ValueError(
                f"layer_index={self.layer_index} is out of range for {len(matrices)} recorded layers."
            )
        attn = matrices[idx]
        if attn.dim() != 4:
            raise ValueError(f"Expected attention tensor of shape [B,H,S,S], got {attn.shape}")
        if self.head_strategy == "mean":
            attn = attn.mean(dim=1)
        else:
            attn = attn.max(dim=1).values
        return attn

    def _extract_image_scores(self, attn: torch.Tensor, num_image_tokens: int) -> torch.Tensor:
        if attn.dim() != 3:
            raise ValueError(f"Expected tensor of shape [B,S,S], got {attn.shape}")
        attn = attn.squeeze(0)
        if attn.dim() != 2:
            raise ValueError("Unable to squeeze batch dimension from attention tensor.")
        if 1 + num_image_tokens > attn.size(-1):
            raise ValueError("Not enough tokens recorded to cover image patches.")
        scores = attn[0, 1 : 1 + num_image_tokens]
        scores = scores - scores.min()
        scores = scores / scores.max().clamp_min(1e-6)
        return scores

    def _scores_to_heatmap(self, token_scores: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
        num_views = int(meta["num_patch_views"])
        tokens_per_view = int(meta["num_image_tokens_per_patch"])
        total_tokens = int(meta["num_total_image_tokens"])
        if token_scores.numel() != total_tokens:
            raise RuntimeError(
                f"Token score length ({token_scores.numel()}) does not match meta ({total_tokens})."
            )
        grid = int(math.sqrt(tokens_per_view))
        if grid * grid != tokens_per_view:
            raise RuntimeError("Tokens per patch do not form a square grid.")
        scores = token_scores.view(num_views, 1, grid, grid).mean(dim=0)
        H = int(meta["original_height"])
        W = int(meta["original_width"])
        heatmap = F.interpolate(scores, size=(H, W), mode="bilinear", align_corners=False)
        return heatmap.squeeze(0).squeeze(0)

    def _render_overlay(
        self,
        image_path: Path,
        heatmap: torch.Tensor,
        record_tag: str,
        route_overlay_dir: Optional[Path],
        speed_overlay_dir: Optional[Path],
    ) -> np.ndarray:
        heatmap_np = heatmap.detach().cpu().numpy()
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        blended = show_cam_on_image(image, heatmap_np, self.colormap_code, self.alpha)
        blended = overlay_trajectories(blended, record_tag, route_overlay_dir, speed_overlay_dir)
        return np.uint8(255 * blended)

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
        if not payload.get("attention"):
            return None
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sim-Lingo ViT raw-attention generator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Hydra config path.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Model checkpoint path.")
    parser.add_argument("--scene_dir", type=Path, default=DEFAULT_SCENE_DIR, help="Directory with scene images.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save heatmaps.")
    parser.add_argument("--device", type=str, default=None, help="Device identifier, e.g., cuda:0.")
    parser.add_argument(
        "--layer_index",
        type=int,
        default=-1,
        help="ViT encoder layer index to visualize (supports negative indices).",
    )
    parser.add_argument(
        "--head_strategy",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="How to aggregate multi-head attention weights.",
    )
    parser.add_argument("--colormap", type=str, default="JET", help="OpenCV colormap name (JET, VIRIDIS 등).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay blend ratio.")
    parser.add_argument("--suffix", type=str, default="vit_raw", help="Output filename suffix.")
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
        help="Optional path to cached Sim-Lingo inference pt directory (or its parent).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = VisionRawAttention(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        layer_index=args.layer_index,
        head_strategy=args.head_strategy,
        colormap=args.colormap,
        alpha=args.alpha,
        trajectory_overlay_root=args.trajectory_overlay_root,
        payload_root=args.payload_root,
    )
    runner.generate_scene_heatmaps(args.scene_dir, args.output_dir, suffix=args.suffix)


if __name__ == "__main__":
    main()
