from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def _normalize_root(root: Optional[Path]) -> Optional[Path]:
    if root is None:
        return None
    root_path = Path(root)
    if not root_path.exists():
        return None
    return root_path


def _find_named_dir(root: Optional[Path], name: str) -> Optional[Path]:
    root_path = _normalize_root(root)
    if root_path is None:
        return None
    # allow passing the directory itself or its parent
    if root_path.name == name and root_path.is_dir():
        return root_path
    candidate = root_path / name
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def resolve_overlay_dirs(scene_dir: Path, overlay_root: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """Return directories for route/speed overlays if available."""
    scene_dir = Path(scene_dir)
    search_roots = []
    if overlay_root is not None:
        search_roots.append(Path(overlay_root))
    search_roots.append(scene_dir)
    # if caller passes .../images, also inspect its parent (scenario root)
    if scene_dir.name == "images" and scene_dir.parent:
        search_roots.append(scene_dir.parent)
    route_dir: Optional[Path] = None
    speed_dir: Optional[Path] = None
    for root in search_roots:
        if route_dir is None:
            route_dir = _find_named_dir(root, "route_overlay")
        if speed_dir is None:
            speed_dir = _find_named_dir(root, "speed_overlay")
    return route_dir, speed_dir


def overlay_trajectories(
    base_image: np.ndarray,
    record_tag: str,
    route_dir: Optional[Path],
    speed_dir: Optional[Path],
) -> np.ndarray:
    """Composite transparent PNG overlays onto the given RGB array (0~1)."""
    result = base_image
    for directory in (route_dir, speed_dir):
        if directory is None:
            continue
        overlay_path = directory / f"{record_tag}.png"
        if not overlay_path.exists():
            continue
        overlay = Image.open(overlay_path).convert("RGBA")
        if overlay.size != (result.shape[1], result.shape[0]):
            overlay = overlay.resize((result.shape[1], result.shape[0]), Image.NEAREST)
        overlay_np = np.array(overlay, dtype=np.float32) / 255.0
        alpha = overlay_np[..., 3:4]
        color = overlay_np[..., :3]
        result = color * alpha + result * (1 - alpha)
    return result

