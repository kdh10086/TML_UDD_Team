#!/usr/bin/env python3
"""Vision transformer attention-flow heatmap generator for Sim-Lingo (cached payload only)."""

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
    """Blend normalized 이미지와 히트맵을 결합한다."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap_code)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    cam = (1 - alpha) * img + alpha * heatmap
    return np.clip(cam, 0, 1)


class VisionAttentionFlow:
    """ViT attention-flow 기반 히트맵 생성기 (캐시 전용)."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        device: Optional[str] = None,
        residual_alpha: float = 0.5,
        discard_ratio: float = 0.0,
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
        if not (0.0 <= discard_ratio < 1.0):
            raise ValueError("discard_ratio must lie in [0, 1).")
        self.config_path = Path(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = OmegaConf.load(self.config_path)
        self.processor = AutoProcessor.from_pretrained(self.cfg.model.vision_model.variant, trust_remote_code=True)
        self.residual_alpha = residual_alpha
        self.discard_ratio = discard_ratio
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
        suffix: str = "vit_flow",
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

        if target_files:
            target_stems = {p.stem for p in target_files}
            filtered_index = {k: v for k, v in self._payload_index.items() if k in target_stems}
            items_to_process = sorted(filtered_index.items())
        else:
            items_to_process = sorted(self._payload_index.items())

        with open(pt_log_path, "a") as log_file:
            for tag, payload_path in tqdm(items_to_process, desc=f"ViTFlow ({scene_dir.name if scene_dir else '?'})", unit="img"):
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
            raise RuntimeError("Cached payload is missing attention/meta for attention flow rendering.")
        moved = self._move_attention_maps_to_device(attention_maps, self.device)
        token_scores = self._compute_attention_flow(moved, meta["num_total_image_tokens"])
        heatmap = self._scores_to_heatmap(token_scores, meta)
        heatmap = self._scores_to_heatmap(token_scores, meta)
        
        if raw_output_dir:
            heatmap_uint8 = np.uint8(255 * heatmap.float().cpu().numpy())
            raw_path = raw_output_dir / f"{image_path.stem}.png"
            Image.fromarray(heatmap_uint8).save(raw_path)

        overlay = self._render_overlay(image_path, heatmap, image_path.stem, route_overlay_dir, speed_overlay_dir)
        output_path = output_dir / f"{image_path.stem}_{suffix}.png"
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
        
        # Group by layer index and pick one (if duplicates)
        from collections import defaultdict
        layer_map = defaultdict(list)
        for idx, name, entries in items:
            layer_map[idx].append((name, entries))
        
        sorted_layers = sorted(layer_map.keys())
        
        matrices: List[torch.Tensor] = []
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
        return matrices

    def _compute_attention_flow(
        self,
        attention_maps: Dict[str, Sequence[Dict[str, torch.Tensor]]],
        num_image_tokens: int,
    ) -> torch.Tensor:
        matrices = self._collect_vision_matrices(attention_maps)
        
        # experiment_alt logic for flow:
        # R = torch.eye(S).expand(B, S, S)
        # for layer:
        #   attn_mean = attn.mean(dim=1)
        #   R = torch.bmm(attn_mean, R)
        # return R[-1, 1:, 1:].sum(dim=0)
        
        first_map = matrices[0]
        B, _, S, S = first_map.shape # Note: collect_vision_matrices returns [B,H,S,S] or [B,S,S]? 
        # Wait, collect_vision_matrices returns [B,H,S,S] (averaged over layers if stacked).
        # Actually in my code it returns [B,H,S,S].
        
        # Initialize R
        R = torch.eye(S, device=first_map.device, dtype=first_map.dtype).unsqueeze(0).expand(B, S, S)
        
        for mat in matrices:
            # mat is [B, H, S, S]
            attn = mat.mean(dim=1) # [B, S, S]
            # No residual, no discard (experiment_alt implementation is simple chain mult)
            # But wait, my previous implementation had residual and discard.
            # experiment_alt's _get_flow_relevance does NOT have residual or discard.
            # I will follow experiment_alt for consistency.
            
            R = torch.bmm(attn, R)
            
        # Select Global View (last batch item) and sum relevance
        if S <= 1:
             raise RuntimeError("Not enough tokens for flow.")
             
        scores = R[-1, 1:, 1:].sum(dim=0)
        
        if scores.shape[0] > num_image_tokens:
            scores = scores[:num_image_tokens]

        scores = scores - scores.min()
        scores = scores / scores.max().clamp_min(1e-6)
        return scores

    def _apply_residual(self, attn: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(attn.size(-1), dtype=attn.dtype, device=attn.device)
        eye = eye.unsqueeze(0).expand(attn.size(0), -1, -1)
        return self.residual_alpha * eye + (1 - self.residual_alpha) * attn

    def _apply_discard(self, attn: torch.Tensor) -> torch.Tensor:
        if self.discard_ratio <= 0:
            return attn
        keep = int((1 - self.discard_ratio) * attn.size(-1))
        if keep <= 0:
            return attn
        topk = torch.topk(attn, keep, dim=-1)
        threshold = topk.values[..., -1:].expand_as(attn)
        mask = attn >= threshold
        attn = attn * mask
        return attn

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
        num_views = int(meta["num_patch_views"])
        tokens_per_view = int(meta["num_image_tokens_per_patch"])
        total_tokens = int(meta["num_total_image_tokens"])
        if token_scores.numel() != total_tokens:
            raise RuntimeError(
                f"Token score length ({token_scores.numel()}) does not match meta ({total_tokens})."
            )
        
        H_orig = int(meta["original_height"])
        W_orig = int(meta["original_width"])
        
        grid_h, grid_w = self._resolve_grid_size(tokens_per_view, H_orig, W_orig)
        
        if grid_h * grid_w != tokens_per_view:
             raise RuntimeError(f"Calculated grid {grid_h}x{grid_w} != tokens_per_view {tokens_per_view}")

        scores = token_scores.view(num_views, 1, grid_h, grid_w).mean(dim=0)
        
        scores = scores.unsqueeze(0) # [1, 1, grid_h, grid_w]
        heatmap = F.interpolate(scores, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
        return heatmap.squeeze(0).squeeze(0)

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
    parser = argparse.ArgumentParser(description="Sim-Lingo ViT attention-flow generator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Hydra config path.")
    parser.add_argument("--scene_dir", type=Path, default=None, help="Optional directory with scene images.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save heatmaps.")
    parser.add_argument("--device", type=str, default=None, help="Device identifier, e.g., cuda:0.")
    parser.add_argument(
        "--residual_alpha",
        type=float,
        default=0.5,
        help="Mixing coefficient between identity connection and raw attention.",
    )
    parser.add_argument(
        "--discard_ratio",
        type=float,
        default=0.0,
        help="Fraction of lowest attention weights to discard per query.",
    )
    parser.add_argument("--colormap", type=str, default="JET", help="OpenCV colormap name (JET, VIRIDIS 등).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay blend ratio.")
    parser.add_argument("--suffix", type=str, default="vit_flow", help="Output filename suffix.")
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
    runner = VisionAttentionFlow(
        config_path=args.config,
        device=args.device,
        residual_alpha=args.residual_alpha,
        discard_ratio=args.discard_ratio,
        colormap=args.colormap,
        alpha=args.alpha,
        trajectory_overlay_root=args.trajectory_overlay_root,
        scene_dir=args.scene_dir,
        payload_root=args.payload_root,
    )
    runner.generate_scene_heatmaps(args.scene_dir, args.output_dir, suffix=args.suffix)


if __name__ == "__main__":
    main()
