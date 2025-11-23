#!/usr/bin/env python3
"""
Overlay viewer for forward frames, heatmaps, and speed texts.

Usage: run and enter dataset root (e.g., data/DREYEVE_DATA_preprocessed). It will find scenarios
with video_garmin (frames), video_saliency (heatmaps), and video_garmin_speed (speed txt) and
open an interactive window per scenario. Use Left/Right arrows to navigate frames; q/Esc to quit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def collect_scenarios(root: Path) -> List[Path]:
    scenarios: List[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "video_garmin").is_dir() and (p / "video_saliency").is_dir() and (p / "video_garmin_speed").is_dir():
            scenarios.append(p)
    return scenarios


def load_speed_table(speed_dir: Path) -> Dict[str, float]:
    table: Dict[str, float] = {}
    for txt in speed_dir.glob("*.txt"):
        stem = txt.stem
        try:
            val = float(txt.read_text().strip().split()[0])
            table[stem] = val
        except Exception:
            continue
    return table


def prepare_frames(scenario_dir: Path) -> Tuple[List[str], Dict[str, float], Path, Path]:
    fwd_dir = scenario_dir / "video_garmin"
    heat_dir = scenario_dir / "video_saliency"
    spd_dir = scenario_dir / "video_garmin_speed"

    frames = [p for p in fwd_dir.glob("*.png")]
    heats = {p.stem for p in heat_dir.glob("*.png")}
    speeds = load_speed_table(spd_dir)

    common_stems = sorted({p.stem for p in frames} & heats & speeds.keys(), key=lambda x: int(x))
    return common_stems, speeds, fwd_dir, heat_dir


def overlay_frame(base_path: Path, heat_path: Path, speed: float, scenario: str, idx: int, total: int) -> np.ndarray:
    base = cv2.imread(str(base_path))
    heat = cv2.imread(str(heat_path), cv2.IMREAD_UNCHANGED)
    if base is None or heat is None:
        raise RuntimeError(f"이미지 로드 실패: {base_path}, {heat_path}")
    if heat.shape[:2] != base.shape[:2]:
        heat = cv2.resize(heat, (base.shape[1], base.shape[0]))
    if heat.ndim == 2:
        heat_gray = heat
    else:
        heat_gray = cv2.cvtColor(heat, cv2.COLOR_BGR2GRAY)
    # Normalize heat to 0-255 for colormap
    heat_norm = cv2.normalize(heat_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(base, 0.6, heat_color, 0.4, 0)
    text = f"Speed: {speed:.2f} m/s ({speed*3.6:.2f} km/h)"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    title = f"{scenario} - {idx+1}/{total}"
    try:
        cv2.setWindowTitle("Overlay Viewer", title)
    except Exception:
        pass
    return overlay


def run_viewer(root: Path, start_path: Path | None = None) -> None:
    scenarios = collect_scenarios(root)
    if not scenarios:
        print(f"전방/히트맵/속도 디렉토리가 모두 있는 시나리오를 찾지 못했습니다: {root}")
        return

    cv2.namedWindow("Overlay Viewer", cv2.WINDOW_NORMAL)

    scenario_idx = 0
    if start_path is not None:
        try:
            scenario_idx = scenarios.index(start_path)
        except ValueError:
            pass

    def load_scenario(idx: int):
        sdir = scenarios[idx]
        stems, speed_table, fwd_dir, heat_dir = prepare_frames(sdir)
        return sdir.name, stems, speed_table, fwd_dir, heat_dir

    scenario_name, stems, speed_table, fwd_dir, heat_dir = load_scenario(scenario_idx)
    frame_idx = 0

    first_display = True

    while True:
        if not stems:
            print(f"[viewer][{scenario_name}] 공통 프레임이 없습니다. 종료합니다.")
            break
        stem = stems[frame_idx]
        frame_path = fwd_dir / f"{stem}.png"
        heat_path = heat_dir / f"{stem}.png"
        overlay = overlay_frame(frame_path, heat_path, speed_table[stem], scenario_name, frame_idx, len(stems))
        cv2.imshow("Overlay Viewer", overlay)
        if first_display:
            h, w = overlay.shape[:2]
            cv2.resizeWindow("Overlay Viewer", w * 2, h * 2)
            first_display = False
        key = cv2.waitKey(0)
        key_low = key & 0xFF
        if key_low in (ord("q"), 27):  # q or Esc
            break
        elif key_low in (2555904, 83, ord("d")):  # right arrow or 'd'
            frame_idx += 1
            if frame_idx >= len(stems):
                scenario_idx = (scenario_idx + 1) % len(scenarios)
                scenario_name, stems, speed_table, fwd_dir, heat_dir = load_scenario(scenario_idx)
                frame_idx = 0
                first_display = True
        elif key_low in (2424832, 81, ord("a")):  # left arrow or 'a'
            frame_idx -= 1
            if frame_idx < 0:
                scenario_idx = (scenario_idx - 1) % len(scenarios)
                scenario_name, stems, speed_table, fwd_dir, heat_dir = load_scenario(scenario_idx)
                frame_idx = len(stems) - 1 if stems else 0
                first_display = True
        else:
            continue

    cv2.destroyWindow("Overlay Viewer")


def main() -> None:
    mode = input("경로 입력 방식 선택 (1: data/<name>, 2: 전체 경로) [1/2]: ").strip() or "1"
    start_path = None
    if mode == "1":
        dataset_name = input("데이터셋 이름을 입력하세요 (data/<name>, 기본: DREYEVE_DATA_preprocessed): ").strip() or "DREYEVE_DATA_preprocessed"
        candidate = Path("data") / dataset_name
        # 만약 이름에 하위 경로가 포함되어 scenario 디렉토리를 직접 가리키면 start_path로 사용
        if (candidate / "video_garmin").is_dir() and (candidate / "video_saliency").is_dir() and (candidate / "video_garmin_speed").is_dir():
            start_path = candidate
            root = candidate.parent
        else:
            root = candidate
    else:
        raw = input("데이터셋 루트 또는 특정 시나리오 경로를 입력하세요: ").strip()
        candidate = Path(raw)
        if (candidate / "video_garmin").is_dir():
            # scenario path given; set root one level up
            start_path = candidate
            root = candidate.parent
        else:
            root = candidate
    if not root.exists() or not root.is_dir():
        print(f"경로가 존재하지 않습니다: {root}")
        sys.exit(1)
    print("조작법: Left/Right 또는 a/d = 이전/다음 프레임, q/Esc = 종료")
    run_viewer(root, start_path=start_path)


if __name__ == "__main__":
    main()
