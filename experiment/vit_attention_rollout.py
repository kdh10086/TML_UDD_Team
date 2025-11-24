#!/usr/bin/env python3
"""Vision transformer attention-rollout heatmap generator for Sim-Lingo (cached payload only)."""

from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoProcessor
from tqdm import tqdm

from experiment.overlay_utils import overlay_trajectories, resolve_overlay_dirs
from experiment.simlingo_inference_baseline import DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_DIR


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, colormap_code: int, alpha: float) -> np.ndarray:
    """입력 이미지를 [0,1]로 정규화했다고 가정하고 히트맵을 덮어 씌운다."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap_code)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    cam = (1 - alpha) * img + alpha * heatmap
    return np.clip(cam, 0, 1)


class VisionAttentionRollout:
    """Sim-Lingo의 ViT 어텐션을 이용해 CLS→패치 rollout relevance를 계산 (캐시 전용)."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        device: Optional[str] = None,
        residual_alpha: float = 0.5,
        start_layer: int = 0,
        colormap: str = "JET",
        alpha: float = 0.5,
        trajectory_overlay_root: Optional[Path] = None,
        scene_dir: Optional[Path] = None,
        payload_root: Optional[Path] = None,
    ) -> None:
        if payload_root is None:
            if scene_dir is None:
                raise ValueError("Either scene_dir or payload_root must be provided.")
            payload_root = Path(scene_dir) / "pt"
        
        self.payload_root = self._resolve_payload_root(payload_root, trajectory_overlay_root)
        if self.payload_root is None and scene_dir is not None:
             # Fallback: try to find pt in scene_dir directly or subdirs
             candidates = [Path(scene_dir) / "pt", Path(scene_dir)]
             for c in candidates:
                 if c.exists() and any(c.glob("*.pt")):
                     self.payload_root = c
                     break
        
        if self.payload_root is None:
             # Relaxed check: Just warn or allow it. The pipeline might populate it later.
             # raise ValueError(f"Could not find payload directory (pt files) in {payload_root} or {scene_dir}")
             print(f"[Warning] Could not find payload directory (pt files) in {payload_root} or {scene_dir}. Assuming it will be populated later.")
             if payload_root:
                 self.payload_root = Path(payload_root)
             elif scene_dir:
                 self.payload_root = Path(scene_dir) / "pt"
        if not (0.0 <= residual_alpha <= 1.0):
            raise ValueError("residual_alpha must lie in [0, 1].")
        self.config_path = Path(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = OmegaConf.load(self.config_path)
        self.processor = AutoProcessor.from_pretrained(self.cfg.model.vision_model.variant, trust_remote_code=True)
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
        self.trajectory_overlay_root = trajectory_overlay_root
        # self.payload_root is already set above
        self._payload_index = self._index_payloads(self.payload_root)

    def generate_scene_heatmaps(
        self,
        scene_dir: Optional[Path],
        output_dir: Path,
        suffix: str = "vit_rollout",
        raw_output_dir: Optional[Path] = None,
        final_output_dir: Optional[Path] = None,
        target_files: Optional[List[Path]] = None,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use final_output_dir if provided
        if final_output_dir is None:
            final_output_dir = output_dir / "final"
            final_output_dir.mkdir(parents=True, exist_ok=True)
        if raw_output_dir:
            raw_output_dir = Path(raw_output_dir)
            raw_output_dir.mkdir(parents=True, exist_ok=True)
        if not self._payload_index:
            raise RuntimeError("No payloads found under payload_root.")
            
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
            for tag, payload_path in tqdm(items_to_process, desc=f"ViTRollout ({scene_dir.name if scene_dir else '?'})", unit="img"):
                # Log the PT file usage (simple format: image_name - pt_filename)
                log_file.write(f"{tag} - {payload_path.name}\n")
                
                payload = torch.load(payload_path, map_location=self.device)
                image_path = self._resolve_image_path(payload, scene_dir)
                route_dir, speed_dir = resolve_overlay_dirs(image_path.parent, self.trajectory_overlay_root)
                self._process_cached_payload(
                    payload, image_path, final_output_dir, suffix, route_dir, speed_dir, raw_output_dir
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
        meta = payload.get("meta")
        if not attention_maps or meta is None:
            raise RuntimeError("Cached payload is missing attention/meta for rollout rendering.")
        moved = self._move_attention_maps_to_device(attention_maps, self.device)
        rollout = self._compute_rollout(moved)
        token_scores = self._extract_image_scores(rollout, meta["num_total_image_tokens"])
        heatmap = self._scores_to_heatmap(token_scores, meta)
        heatmap = self._scores_to_heatmap(token_scores, meta)
        
        if raw_output_dir:
            heatmap_uint8 = np.uint8(255 * heatmap.float().cpu().numpy())
            raw_path = raw_output_dir / f"{image_path.stem}.png"
            Image.fromarray(heatmap_uint8).save(raw_path)

        overlay = self._render_overlay(image_path, heatmap, image_path.stem, route_overlay_dir, speed_overlay_dir)
        output_path = output_dir / f"{image_path.stem}.png"
        Image.fromarray(overlay).save(output_path)
        return output_path

    def _resolve_image_path(self, payload: Dict[str, Any], scene_dir: Optional[Path]) -> Path:
        raw_path = payload.get("image_path")
        if raw_path:
            p = Path(raw_path)
            if p.exists():
                return p
        if scene_dir is not None:
            candidates = []
            scene_dir = Path(scene_dir)
            candidates.append(scene_dir / "input_images" / f"{payload['tag']}.png")
            candidates.append(scene_dir / "video_garmin" / f"{payload['tag']}.png")
            candidates.append(scene_dir / "images" / f"{payload['tag']}.png")
            candidates.append(scene_dir / f"{payload['tag']}.png")
            for c in candidates:
                if c.exists():
                    return c
        for base in [self.payload_root, self.payload_root.parent if self.payload_root else None]:
            if base is None:
                continue
            input_dir = base / "input_images"
            cand = input_dir / f"{payload['tag']}.png"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Image not found for payload tag {payload.get('tag')}")

    @staticmethod
    def _prepare_output_subdir(output_root: Path, scene_dir: Optional[Path], suffix: str) -> Path:
        scenario_name = scene_dir.name if scene_dir is not None else "payload"
        base = f"{scenario_name}_{suffix}"
        candidate = output_root / base
        counter = 1
        while candidate.exists():
            candidate = output_root / f"{base}_{counter}"
            counter += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _collect_vision_matrices(
        self, attention_maps: Dict[str, Sequence[Dict[str, torch.Tensor]]]
    ) -> List[torch.Tensor]:
        import re
        # Filter for vision block attention keys
        # Supports keys like "vision_block_0" or "vision_attn_vision_model_encoder_layers_0_attn_attn_drop"
        items = []
        for name, entries in attention_maps.items():
            if "vision_block" in name or "vision_attn" in name:
                # Try to extract layer index
                # Matches "block_12" or "layers_12"
                match = re.search(r"(?:block|layers)_(\d+)", name)
                if match:
                    layer_idx = int(match.group(1))
                    items.append((layer_idx, name, entries))
        
        if not items:
            raise RuntimeError("No vision block attention maps were recorded.")
        
        # Sort by layer index
        items.sort(key=lambda x: x[0])
        
        matrices: List[torch.Tensor] = []
        # Group by layer index and pick one (if duplicates)
        from collections import defaultdict
        layer_map = defaultdict(list)
        for idx, name, entries in items:
            layer_map[idx].append((name, entries))
        
        sorted_layers = sorted(layer_map.keys())
        
        for idx in sorted_layers:
            # Pick one entry per layer. Maybe pick the one with longest name? (more specific)
            best_entry = max(layer_map[idx], key=lambda x: len(x[0]))
            entries = best_entry[1]
            tensors = [entry["attn"] for entry in entries if entry.get("attn") is not None]
            if not tensors:
                continue
            merged: List[torch.Tensor] = []
            for t in tensors:
                if t.dim() == 5:  # [L,B,H,S,S]
                    merged.extend([t_layer for t_layer in t])
                else:
                    merged.append(t)
            if not merged:
                continue
            stacked = torch.stack(merged, dim=0).mean(dim=0)
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
            # experiment_alt logic: R = R + torch.bmm(attn_mean, R)
            # We initialize joint (R) as identity matrix for the first layer if it's None.
            
            if joint is None:
                # Initialize R as identity matrix [B, S, S]
                joint = torch.eye(seq_len, device=attn.device, dtype=attn.dtype).unsqueeze(0).expand(bsz, seq_len, seq_len)
            
            # R = torch.bmm(attn_mean, R)
            # Note: experiment_alt adds residual and normalizes attn_mean before this step.
            # But wait, in experiment_alt:
            #   attn_mean = attn.mean(dim=1)
            #   attn_mean = attn_mean + torch.eye(S)
            #   attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
            #   R = torch.bmm(attn_mean, R)
            
            # I need to replicate this preprocessing of attn here.
            eye = torch.eye(seq_len, device=attn.device, dtype=attn.dtype).unsqueeze(0)
            attn = attn + eye
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            joint = torch.bmm(attn, joint)
            
        if joint is None:
            raise RuntimeError("Failed to compute rollout due to missing attention matrices.")
            
        return joint

    def _extract_image_scores(self, rollout: torch.Tensor, num_image_tokens: int) -> torch.Tensor:
        # rollout is [B, S, S]
        if rollout.dim() != 3:
             raise ValueError(f"Expected rollout matrix of shape [B,S,S], got {rollout.shape}")
        
        # Select Global View (last batch item)
        # And sum relevance over all tokens (dim 0) excluding CLS (index 0)
        # R[-1, 1:, 1:].sum(dim=0)
        
        # Check if we have enough tokens
        S = rollout.shape[-1]
        if S <= 1:
             raise ValueError("Not enough tokens for rollout.")
             
        scores = rollout[-1, 1:, 1:].sum(dim=0)
        
        # Truncate to num_image_tokens if needed (though usually S-1 == num_image_tokens)
        if scores.shape[0] > num_image_tokens:
            scores = scores[:num_image_tokens]
            
        scores = scores - scores.min()
        return scores / scores.max().clamp_min(1e-6)

    @staticmethod
    def _resolve_grid_size(n: int, h: int, w: int) -> Tuple[int, int]:
        # Find factors of n that best match aspect ratio h/w
        target_ratio = h / w
        best_h, best_w = int(math.sqrt(n)), int(math.sqrt(n))
        min_error = float("inf")
        
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                # Factor pair (i, n//i)
                h1, w1 = i, n // i
                ratio1 = h1 / w1
                err1 = abs(ratio1 - target_ratio)
                if err1 < min_error:
                    min_error = err1
                    best_h, best_w = h1, w1
                
                # Factor pair (n//i, i)
                h2, w2 = n // i, i
                ratio2 = h2 / w2
                err2 = abs(ratio2 - target_ratio)
                if err2 < min_error:
                    min_error = err2
                    best_h, best_w = h2, w2
        return best_h, best_w

    def _scores_to_heatmap(self, token_scores: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
        num_views = max(1, int(meta["num_patch_views"]))
        total_tokens = int(meta["num_total_image_tokens"])
        H_orig = int(meta["original_height"])
        W_orig = int(meta["original_width"])

        scores_flat = token_scores.flatten()
        expected = total_tokens if total_tokens > 0 else scores_flat.numel()
        if scores_flat.numel() < expected:
            pad = torch.zeros(expected - scores_flat.numel(), device=scores_flat.device, dtype=scores_flat.dtype)
            scores_flat = torch.cat([scores_flat, pad], dim=0)
        elif scores_flat.numel() > expected:
            scores_flat = scores_flat[:expected]

        per_view = max(1, expected // num_views)
        grids: List[torch.Tensor] = []
        target_ratio = H_orig / max(W_orig, 1)

        for v in range(num_views):
            start = v * per_view
            end = min(start + per_view, scores_flat.numel())
            view_scores = scores_flat[start:end]
            if view_scores.numel() == 0:
                continue
            grid_h = max(1, int(round(math.sqrt(view_scores.numel() * target_ratio))))
            grid_w = max(1, int(math.ceil(view_scores.numel() / grid_h)))
            if grid_h * grid_w < view_scores.numel():
                grid_h = int(math.ceil(view_scores.numel() / grid_w))
            if grid_h * grid_w != view_scores.numel():
                pad = torch.zeros(grid_h * grid_w - view_scores.numel(), device=view_scores.device, dtype=view_scores.dtype)
                view_scores = torch.cat([view_scores, pad], dim=0)
            grids.append(view_scores.view(1, 1, grid_h, grid_w))

        if not grids:
            raise RuntimeError("No view scores available to build heatmap.")

        scores = torch.stack(grids, dim=0).mean(dim=0)  # [1,1,H,W]
        heatmap = F.interpolate(scores, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
        heatmap = heatmap.squeeze(0).squeeze(0)
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()).clamp_min(1e-6)

    def _render_overlay(
        self,
        image_path: Path,
        heatmap: torch.Tensor,
        record_tag: str,
        route_overlay_dir: Optional[Path],
        speed_overlay_dir: Optional[Path],
    ) -> np.ndarray:
        heatmap_np = heatmap.detach().float().cpu().numpy()
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
    parser.add_argument("--scene_dir", type=Path, default=None, help="Optional directory with scene images (fallback if input_images is missing).")
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
        help="Path to cached .pt payload directory (optional, defaults to scene_dir/pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = VisionAttentionRollout(
        config_path=args.config,
        device=args.device,
        residual_alpha=args.residual_alpha,
        start_layer=args.start_layer,
        colormap=args.colormap,
        alpha=args.alpha,
        trajectory_overlay_root=args.trajectory_overlay_root,
        scene_dir=args.scene_dir,
        payload_root=args.payload_root,
    )
    runner.generate_scene_heatmaps(args.scene_dir, args.output_dir, suffix=args.suffix)


if __name__ == "__main__":
    main()
