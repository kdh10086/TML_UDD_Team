#!/usr/bin/env python3
"""Vision transformer attention-rollout heatmap generator for Sim-Lingo."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
    ) -> None:
        super().__init__(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            target_mode="auto",
            explain_mode="action",
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

    def generate_scene_heatmaps(self, scene_dir: Path, output_dir: Path, suffix: str = "vit_rollout") -> None:
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        scenario_output_dir = self._prepare_output_subdir(output_dir, scene_dir.name, suffix)
        image_paths = sorted(
            [p for p in scene_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        for image_path in image_paths:
            self._process_single_image(image_path, scenario_output_dir, suffix)

    def _process_single_image(self, image_path: Path, output_dir: Path, suffix: str) -> Path:
        record_tag = image_path.stem
        self.recorder.start_recording(record_tag)
        driving_input, meta = self._prepare_driving_input(image_path)
        _ = self.model(driving_input)
        attention_maps = self.recorder.stop_recording()
        self.model.zero_grad(set_to_none=True)
        rollout = self._compute_rollout(attention_maps)
        token_scores = self._extract_image_scores(rollout, meta["num_total_image_tokens"])
        heatmap = self._scores_to_heatmap(token_scores, meta)
        overlay = self._render_overlay(image_path, heatmap)
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

    def _render_overlay(self, image_path: Path, heatmap: torch.Tensor) -> np.ndarray:
        heatmap_np = heatmap.detach().cpu().numpy()
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        blended = show_cam_on_image(image, heatmap_np, self.colormap_code, self.alpha)
        return np.uint8(255 * blended)


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
    )
    runner.generate_scene_heatmaps(args.scene_dir, args.output_dir, suffix=args.suffix)


if __name__ == "__main__":
    main()
