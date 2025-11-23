#!/usr/bin/env python3
"""
Filter a dataset to keep only forward frames, heatmap frames, and speed text directories.

Usage:
  python tools/filter_dataset.py --src data/DREYEVE_DATA_preprocessed --dst data/DREYEVE_DATA_filtered

Default keeps: video_garmin, video_saliency, video_garmin_speed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

from tqdm.auto import tqdm


def copy_subdir(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def filter_dataset(src_root: Path, dst_root: Path, keep_dirs: List[str]) -> None:
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    scenarios = [p for p in sorted(src_root.iterdir()) if p.is_dir()]
    if not scenarios:
        print(f"No scenarios found under {src_root}")
        return

    print(f"Filtering {len(scenarios)} scenarios from {src_root} -> {dst_root}")
    for scen in tqdm(scenarios, desc="scenarios", unit="scenario"):
        rel = scen.relative_to(src_root)
        dst_scen = dst_root / rel
        dst_scen.mkdir(parents=True, exist_ok=True)
        for d in keep_dirs:
            if d:
                src_d = scen / d
                if src_d.exists():
                    dst_d = dst_scen / d
                    copy_subdir(src_d, dst_d)
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter dataset to keep only selected subdirectories per scenario.")
    parser.add_argument("--src", type=Path, required=True, help="Source dataset root.")
    parser.add_argument("--dst", type=Path, required=True, help="Destination dataset root.")
    parser.add_argument(
        "--keep",
        nargs="+",
        default=["video_garmin", "video_saliency", "video_garmin_speed"],
        help="Subdirectories to keep for each scenario (default: video_garmin video_saliency video_garmin_speed).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    mode = input("경로 입력 방식 선택 (1: data/<name>, 2: 전체 경로) [1/2]: ").strip() or "1"
    if mode == "1":
        src_name = input("원본 데이터셋 이름 (data/<name>): ").strip()
        dst_name = input("목적지 데이터셋 이름 (data/<name>): ").strip()
        src = Path("data") / src_name
        dst = Path("data") / dst_name
    else:
        src = Path(input("원본 루트 전체 경로: ").strip())
        dst = Path(input("목적지 루트 전체 경로: ").strip())
    keep_dirs = input("유지할 서브디렉토리(공백 구분, 엔터시 기본: video_garmin video_saliency video_garmin_speed): ").strip()
    keep = keep_dirs.split() if keep_dirs else ["video_garmin", "video_saliency", "video_garmin_speed"]
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Source root not found: {src}")
    filter_dataset(src, dst, keep_dirs=keep)


if __name__ == "__main__":
    main()
