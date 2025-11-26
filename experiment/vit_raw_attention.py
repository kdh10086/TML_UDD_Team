#!/usr/bin/env python3
"""Sim-Lingo vision encoder raw-attention heatmap generator (cached payload only)."""

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
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from transformers import AutoProcessor
from tqdm import tqdm

from experiment.overlay_utils import overlay_trajectories, resolve_overlay_dirs
from experiment.simlingo_inference_baseline import DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_DIR


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, colormap_code: int, alpha: float) -> np.ndarray:
    """Blend 입력 이미지와 히트맵을 결합한다."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap_code)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    cam = (1 - alpha) * img + alpha * heatmap
    return np.clip(cam, 0, 1)


class VisionRawAttention:
    """ViT 어텐션 가중치 자체를 이용한 히트맵 시각화 (캐시 전용)."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        device: Optional[str] = None,
        layer_index: int = -1,
        head_strategy: str = "mean",
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
        self.config_path = Path(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = OmegaConf.load(self.config_path)
        self.processor = AutoProcessor.from_pretrained(self.cfg.model.vision_model.variant, trust_remote_code=True)
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
        self.trajectory_overlay_root = trajectory_overlay_root
        # self.payload_root is already set above
        self._payload_index = self._index_payloads(self.payload_root)

    def generate_scene_heatmaps(
        self,
        scene_dir: Optional[Path],
        output_dir: Path,
        suffix: str = "vit_raw",
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
            for tag, payload_path in tqdm(items_to_process, desc=f"ViTRaw ({scene_dir.name if scene_dir else '?'})", unit="img"):
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
            raise RuntimeError("Cached payload is missing attention/meta for raw attention rendering.")
        moved = self._move_attention_maps_to_device(attention_maps, self.device)
        attention_tensor = self._select_layer_attention(moved)
        
        # Check for multiple views (crops)
        # attention_tensor shape: [B, S] or [B, H, S] -> we aggregated heads, so [B, S] or [B, S, S]
        # Actually _select_layer_attention returns [B, S, S] (if heads aggregated)
        
        B = attention_tensor.shape[0]
        heatmap = None
        
        if B > 1:
            # High-res mode: B-1 crops + 1 global thumbnail
            crops_attn = attention_tensor[:-1]  # [B-1, S, S]
            global_attn = attention_tensor[-1]  # [S, S]
            
            # 1. Compute global heatmap for fallback/reference
            global_scores = self._extract_image_scores(global_attn, meta["num_total_image_tokens"])
            # Note: num_total_image_tokens is for ALL patches. 
            # For a single view, we should use num_image_tokens_per_patch
            tokens_per_view = int(meta["num_image_tokens_per_patch"])
            global_scores = self._extract_image_scores(global_attn, tokens_per_view)
            global_heatmap = self._scores_to_heatmap(global_scores, meta, force_square=True) 
            # force_square because thumbnail is resized to square-ish usually? 
            # Actually InternVL2 thumbnail is resized to 448x448 (or similar).
            
            # 2. Compute crop heatmaps
            crop_heatmaps = []
            for i in range(B - 1):
                c_attn = crops_attn[i]
                c_scores = self._extract_image_scores(c_attn, tokens_per_view)
                # Each crop is a square tile (e.g. 448x448)
                c_map = self._scores_to_heatmap(c_scores, meta, target_size=(448, 448), force_square=True)
                crop_heatmaps.append(c_map)
                
            # 3. Stitch
            orig_h, orig_w = meta["original_height"], meta["original_width"]
            grid_h, grid_w = self._resolve_grid_layout(B - 1, orig_h, orig_w)
            stitched = self._stitch_heatmaps(crop_heatmaps, grid_h, grid_w)
            
            # 4. Resize stitched to original size
            heatmap = cv2.resize(stitched, (orig_w, orig_h))
            
            # Normalize stitched
            if heatmap.max() - heatmap.min() > 1e-9:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
                
        else:
            # Low-res mode: only global view
            token_scores = self._extract_image_scores(attention_tensor[0], meta["num_total_image_tokens"])
            heatmap = self._scores_to_heatmap(token_scores, meta)

        if raw_output_dir:
            heatmap_uint8 = np.uint8(255 * heatmap)
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
            candidates.append(scene_dir / "images" / f"{payload['tag']}.png")
            candidates.append(scene_dir / f"{payload['tag']}.png")
            for c in candidates:
                if c.exists():
                    return c
        # fallback to sibling input_images under payload_root
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
            if cand.exists() and cand.is_dir() and any(cand.glob("*.pt")):
                return cand
        return None

    @staticmethod
    def _index_payloads(payload_root: Optional[Path]) -> Dict[str, Path]:
        if payload_root is None:
            return {}
        return {p.stem: p for p in Path(payload_root).glob("*.pt")}

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
        # experiment_alt logic:
        # attn = vision_maps[last_name].float()
        # return attn.mean(dim=1)[-1, 1:, 1:].sum(dim=0)
        
        # attn is [B, S, S] (already mean-ed over heads in _select_layer_attention)
        if attn.dim() != 3:
            raise ValueError(f"Expected tensor of shape [B,S,S], got {attn.shape}")
            
        # Select Global View (last batch item)
        # Sum over rows (dim 0 of the S,S matrix) -> relevance of all tokens TO target?
        # Wait, attn[b, i, j] is attention FROM i TO j.
        # experiment_alt does: attn.mean(dim=1)[-1, 1:, 1:].sum(dim=0)
        # This is sum over i (rows) for each j (cols).
        # So it measures "how much attention does this token j receive from all other tokens i".
        # This is "Incoming Attention" or "Centrality".
        
        scores = attn[-1, 1:, 1:].sum(dim=0)
        
        if scores.shape[0] > num_image_tokens:
            scores = scores[:num_image_tokens]
            
        scores = scores - scores.min()
        scores = scores / scores.max().clamp_min(1e-6)
        return scores

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

    def _scores_to_heatmap(
        self, 
        token_scores: torch.Tensor, 
        meta: Dict[str, int], 
        target_size: Optional[Tuple[int, int]] = None,
        force_square: bool = False
    ) -> np.ndarray:
        # If force_square is True, we assume the token grid is square (e.g. for crops)
        # Otherwise we try to match aspect ratio
        
        total_tokens = int(token_scores.numel())
        if force_square:
            side = int(math.sqrt(total_tokens))
            grid_h, grid_w = side, side
        else:
            H_orig = int(meta["original_height"])
            W_orig = int(meta["original_width"])
            target_ratio = H_orig / max(W_orig, 1)
            grid_h = max(1, int(round(math.sqrt(total_tokens * target_ratio))))
            grid_w = max(1, int(math.ceil(total_tokens / grid_h)))
            if grid_h * grid_w < total_tokens:
                grid_h = int(math.ceil(total_tokens / grid_w))

        if grid_h * grid_w != total_tokens:
            pad = torch.zeros(grid_h * grid_w - total_tokens, device=token_scores.device, dtype=token_scores.dtype)
            token_scores = torch.cat([token_scores, pad], dim=0)

        heatmap = token_scores.view(grid_h, grid_w).detach().float().to("cpu").numpy()
        
        # Normalize locally
        if heatmap.max() - heatmap.min() > 1e-9:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            
        if target_size:
            heatmap = cv2.resize(heatmap, target_size)
        elif not force_square:
             heatmap = cv2.resize(heatmap, (meta["original_width"], meta["original_height"]))
             
        return heatmap

    def _resolve_grid_layout(self, n: int, h: int, w: int) -> Tuple[int, int]:
        """Determine (rows, cols) for n tiles to best match image aspect ratio h/w."""
        target_ratio = h / w
        best_r, best_c = 1, n
        min_err = float("inf")
        
        for r in range(1, n + 1):
            if n % r == 0:
                c = n // r
                # Aspect ratio of the grid of square tiles is r/c
                ratio = r / c
                err = abs(ratio - target_ratio)
                if err < min_err:
                    min_err = err
                    best_r, best_c = r, c
        return best_r, best_c

    def _stitch_heatmaps(self, heatmaps: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
        """Stitch list of heatmaps into a single large heatmap."""
        if not heatmaps:
            return np.zeros((100, 100), dtype=np.float32)
        
        # Assume all heatmaps have same size (they should, coming from same model)
        h, w = heatmaps[0].shape
        
        # Create canvas
        canvas = np.zeros((rows * h, cols * w), dtype=np.float32)
        
        for idx, hm in enumerate(heatmaps):
            if idx >= rows * cols:
                break
            r = idx // cols
            c = idx % cols
            canvas[r*h : (r+1)*h, c*w : (c+1)*w] = hm
            
        return canvas

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
    parser = argparse.ArgumentParser(description="Sim-Lingo ViT raw-attention generator (cache only)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Hydra config path.")
    parser.add_argument("--scene_dir", type=Path, default=None, help="Optional directory with scene images (fallback if input_images is missing).")
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
        help="Path to route/speed overlays (optional).",
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
    runner = VisionRawAttention(
        config_path=args.config,
        device=args.device,
        layer_index=args.layer_index,
        head_strategy=args.head_strategy,
        colormap=args.colormap,
        alpha=args.alpha,
        trajectory_overlay_root=args.trajectory_overlay_root,
        scene_dir=args.scene_dir,
        payload_root=args.payload_root,
    )
    runner.generate_scene_heatmaps(args.scene_dir, args.output_dir, suffix=args.suffix)


if __name__ == "__main__":
    main()
