#!/usr/bin/env python3
"""Vision transformer attention-rollout heatmap generator for Sim-Lingo."""

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
    """입력 이미지를 [0,1]로 정규화했다고 가정하고 히트맵을 덮어 씌운다."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap_code)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    cam = (1 - alpha) * img + alpha * heatmap
    return np.clip(cam, 0, 1)


class VisionAttentionRollout(SimLingoInferenceBaseline):
    """Sim-Lingo의 ViT 어텐션을 이용해 CLS→패치 rollout relevance를 계산한다."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: Optional[str] = None,
        residual_alpha: float = 0.5,
        start_layer: int = 0,
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
        if not (0.0 <= residual_alpha <= 1.0):
            raise ValueError("residual_alpha must lie in [0, 1].")
        self.residual_alpha = residual_alpha
        self.start_layer = start_layer
        cmap_name = colormap.upper()
        attr_name = f"COLORMAP_{cmap_name}"
        if not hasattr(cv2, attr_name):
            raise ValueError(f"Unsupported OpenCV colormap: {colormap}")
        self.colormap_code = getattr(cv2, attr_name)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha
        self.trajectory_overlay_root = trajectory_overlay_root
        self.payload_root = self._resolve_payload_root(payload_root, trajectory_overlay_root)
        self._payload_index = self._index_payloads(self.payload_root)

    def generate_scene_heatmaps(self, scene_dir: Path, output_dir: Path, suffix: str = "vit_rollout") -> None:
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
        rollout = self._compute_rollout(attention_maps)
        token_scores = self._extract_image_scores(rollout, meta["num_total_image_tokens"])
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
            raise RuntimeError("Cached payload is missing attention/meta for rollout rendering.")
        moved = self._move_attention_maps_to_device(attention_maps, self.device)
        rollout = self._compute_rollout(moved)
        token_scores = self._extract_image_scores(rollout, meta["num_total_image_tokens"])
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
            stacked = torch.stack(tensors, dim=0).mean(dim=0)  # average over multiple calls if any
            if stacked.dim() == 3:
                stacked = stacked.unsqueeze(0)
            matrices.append(stacked)
        if len(matrices) <= self.start_layer:
            raise ValueError(
                f"start_layer={self.start_layer} is out of range for {len(matrices)} recorded blocks."
            )
        return matrices[self.start_layer :]

    def _compute_rollout(self, attention_maps: Dict[str, Sequence[Dict[str, torch.Tensor]]]) -> torch.Tensor:
        matrices = self._collect_vision_matrices(attention_maps)
        joint = None
        for mat in matrices:
            if mat.dim() != 4:
                raise ValueError(f"Expected attention tensor of shape [B,H,S,S], got {mat.shape}")
            bsz, _, seq_len, _ = mat.shape
            attn = mat.mean(dim=1)
            eye = torch.eye(seq_len, device=attn.device, dtype=attn.dtype).unsqueeze(0)
            attn = self.residual_alpha * eye + (1 - self.residual_alpha) * attn
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            if joint is None:
                joint = attn
            else:
                joint = torch.bmm(attn, joint)
        if joint is None:
            raise RuntimeError("Failed to compute rollout due to missing attention matrices.")
        return joint.squeeze(0)

    def _extract_image_scores(self, rollout: torch.Tensor, num_image_tokens: int) -> torch.Tensor:
        if rollout.dim() != 2:
            raise ValueError(f"Expected rollout matrix of shape [S,S], got {rollout.shape}")
        if 1 + num_image_tokens > rollout.size(-1):
            raise ValueError("Not enough tokens recorded to cover all image patches.")
        scores = rollout[0, 1 : 1 + num_image_tokens]
        scores = scores - scores.min()
        return scores / scores.max().clamp_min(1e-6)

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
    parser = argparse.ArgumentParser(description="Sim-Lingo ViT attention rollout generator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Hydra config path.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Model checkpoint path.")
    parser.add_argument("--scene_dir", type=Path, default=DEFAULT_SCENE_DIR, help="Directory with scene images.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save heatmaps.")
    parser.add_argument("--device", type=str, default=None, help="Device identifier, e.g., cuda:0.")
    parser.add_argument(
        "--residual_alpha",
        type=float,
        default=0.5,
        help="Mixing coefficient between identity connection and raw attention.",
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        default=0,
        help="Start rollout from this ViT encoder layer index.",
    )
    parser.add_argument("--colormap", type=str, default="JET", help="OpenCV colormap name (JET, VIRIDIS 등).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay blend ratio.")
    parser.add_argument("--suffix", type=str, default="vit_rollout", help="Output filename suffix.")
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
    runner = VisionAttentionRollout(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        residual_alpha=args.residual_alpha,
        start_layer=args.start_layer,
        colormap=args.colormap,
        alpha=args.alpha,
        trajectory_overlay_root=args.trajectory_overlay_root,
        payload_root=args.payload_root,
    )
    runner.generate_scene_heatmaps(args.scene_dir, args.output_dir, suffix=args.suffix)


if __name__ == "__main__":
    main()
