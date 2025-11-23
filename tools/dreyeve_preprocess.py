#!/usr/bin/env python3
"""
DR(eye)VE preprocessing utilities.

Provides:
- crop: interactive 2:1 crop selector (single video or batch under dataset root, in-place; paired heatmap cropped 같이 처리, 기본 무손실 FFV1).
- frames: extract video frames to numbered PNG files in a video-named directory.
- sync-speed: align per-frame speed values to extracted frames and save as text files (m/s).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


TARGET_FPS = 4.0


def largest_aspect_crop(width: int, height: int, aspect: float = 2.0) -> Tuple[int, int]:
    """
    Compute the largest rectangle with the given aspect ratio that fits inside the frame.
    Returns (crop_w, crop_h).
    """
    crop_h = min(height, int(width / aspect))
    crop_w = int(aspect * crop_h)
    return crop_w, crop_h


def interactive_crop(
    video_path: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Open a window to preview the video with a 2:1 crop box.
    Space toggles play/pause, up/down arrows move the crop vertically, Enter confirms and saves the cropped video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Video has no frames: {video_path}")

    window_name = "Crop selector (2:1)"
    height, width = frame.shape[:2]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width * 3, height * 3)

    crop_w, crop_h = largest_aspect_crop(width, height, aspect=2.0)
    x0 = (width - crop_w) // 2
    y0 = 0
    size_step = 10
    min_h = 20
    max_h = min(height, width // 2)

    paused = True

    instructions = (
        "Space: play/pause | Up/Down: move Y | Left/Right: move X | A/D: shrink/grow | Enter: confirm | q/Esc: quit"
    )

    while True:
        display = frame.copy()
        cv2.rectangle(display, (x0, y0), (x0 + crop_w - 1, y0 + crop_h - 1), (0, 0, 255), 2)
        cv2.putText(
            display,
            f"Crop {crop_w}x{crop_h}px, y={y0}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            instructions,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30)
        key_low = key & 0xFF

        if key_low in (ord("q"), 27):  # q or Esc
            cap.release()
            cv2.destroyWindow(window_name)
            raise SystemExit("Cancelled by user.")
        elif key_low in (13, 10):  # Enter
            break
        elif key_low in (ord(" "),):
            paused = not paused
        elif key in (2490368, 63232, 82):
            # Up arrow (platform-dependent codes)
            y0 = max(0, y0 - 1)
        elif key in (2621440, 63233, 84):
            # Down arrow (platform-dependent codes)
            y0 = min(height - crop_h, y0 + 1)
        elif key in (2424832, 63234, 81):
            # Left arrow
            x0 = max(0, x0 - 1)
        elif key in (2555904, 63235, 83):
            # Right arrow
            x0 = min(width - crop_w, x0 + 1)
        elif key_low == ord("a"):  # shrink (2:1 fixed)
            new_h = max(min_h, crop_h - size_step)
            new_w = 2 * new_h
            center_x = x0 + crop_w // 2
            x0 = int(center_x - new_w // 2)
            if x0 < 0 or x0 + new_w > width:
                # Keep center; if it would exceed boundary, skip resize
                continue
            y0 = min(y0, height - new_h)
            crop_w, crop_h = new_w, new_h
        elif key_low == ord("d"):  # grow (2:1 fixed)
            new_h = min(max_h, crop_h + size_step)
            new_w = 2 * new_h
            center_x = x0 + crop_w // 2
            x0 = int(center_x - new_w // 2)
            if x0 < 0 or x0 + new_w > width:
                # Cannot grow without changing center; skip.
                continue
            y0 = min(y0, height - new_h)
            crop_w, crop_h = new_w, new_h

        if not paused:
            ret, frame = cap.read()
            if not ret:
                paused = True
                # Rewind one frame to keep displaying the last frame.
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, 0))

    cv2.destroyWindow(window_name)

    # Perform the actual crop and write the output video.
    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_cropped{video_path.suffix}")

    # Close the preview window before heavy encode.
    cv2.destroyAllWindows()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_w, crop_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
        writer.write(crop)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path


def get_video_info(video_path: Path) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frames


def is_cropped_aspect(width: int, height: int, target: float = 2.0, tol: float = 0.01) -> bool:
    return height > 0 and abs((width / height) - target) <= tol


def crop_video_inplace(video_path: Path, rect: Tuple[int, int, int, int]) -> None:
    """
    Crop video to rect (x0, y0, w, h) and overwrite original file.
    """
    x0, y0, cw, ch = rect
    scenario = video_path.parent.name
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    log_interval = max(1, total_frames // 40) if total_frames else 250
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")

    with tempfile.NamedTemporaryFile(delete=False, suffix=video_path.suffix) as tmp:
        tmp_path = Path(tmp.name)
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (cw, ch))

    if total_frames:
        print(f"[crop][{scenario}] {video_path.name}: {total_frames} frames, writing...", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y0 : y0 + ch, x0 : x0 + cw]
        writer.write(crop)
        if total_frames:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos % log_interval == 0 or pos == total_frames:
                pct = (pos / total_frames) * 100
                print(f"[crop][{scenario}] {video_path.name}: {pos}/{total_frames} ({pct:.1f}%)", flush=True)

    cap.release()
    writer.release()
    tmp_path.replace(video_path)
    print(f"[crop][{scenario}] {video_path.name}: done. Output overwritten.", flush=True)


def crop_pair_inplace(
    main_video: Path,
    rect: Tuple[int, int, int, int],
    heatmap_video: Optional[Path] = None,
) -> None:
    crop_video_inplace(main_video, rect)
    if heatmap_video and heatmap_video.exists():
        crop_video_inplace(heatmap_video, rect)


def batch_extract_frames(
    dataset_root: Path,
    forward_name: str = "video_garmin.avi",
    heatmap_name: str = "video_saliency.avi",
    overwrite: bool = False,
) -> None:
    """
    Extract frames for all 2:1 cropped forward and heatmap videos under dataset_root.
    Skips videos that are not already 2:1. Uses existing extract_frames.
    """
    forward_videos = sorted(dataset_root.rglob(forward_name))
    heatmap_videos = sorted(dataset_root.rglob(heatmap_name))

    def _process(vlist: List[Path], label: str) -> None:
        if not vlist:
            print(f"[frames] {label}: 대상 영상이 없습니다.")
            return
        for vid in vlist:
            w, h, _, _ = get_video_info(vid)
            if not is_cropped_aspect(w, h):
                print(f"[frames] {label} skip (not 2:1): {vid}")
                continue
            try:
                out_dir = extract_frames(video_path=vid, overwrite=overwrite, target_fps=TARGET_FPS)
                print(f"[frames] {label} done: {vid} -> {out_dir}")
            except FileExistsError:
                print(f"[frames] {label} skip (output exists): {vid}")
            except Exception as exc:
                print(f"[frames] {label} 실패: {vid} ({exc})", file=sys.stderr)

    print(f"[frames] 전방 영상 처리 시작 (root={dataset_root})")
    _process(forward_videos, "forward")
    print(f"[frames] 히트맵 영상 처리 시작 (root={dataset_root})")
    _process(heatmap_videos, "heatmap")


def integrated_pipeline(
    dataset_root: Path,
    forward_name: str = "video_garmin.avi",
    heatmap_name: str = "video_saliency.avi",
    overwrite_frames: bool = False,
    input_unit: str = "kmh",
) -> None:
    """
    Interactive crop selection first, then automatic crop encode, frame extraction, and speed sync
    for the selected (2:1) scenarios only.
    """
    queued = interactive_crop_batch(
        dataset_root=dataset_root,
        forward_name=forward_name,
        heatmap_name=heatmap_name,
        encode_now=False,
    )
    if not queued:
        return

    # Crops selected; proceed to frames + speed for selected scenarios (no video re-encode).
    scenarios = {}
    for vpath, _ in queued:
        scenario = vpath.parent.name
        scenarios.setdefault(scenario, set()).add(vpath)

    print(f"[integrated] 총 {len(scenarios)}개 시나리오에 대해 프레임/속도 변환을 진행합니다.")
    crop_map = dict(queued)
    for scenario, vids in scenarios.items():
        for vpath in vids:
            try:
                out_dir = extract_frames(video_path=vpath, overwrite=overwrite_frames, target_fps=TARGET_FPS)
                crop_frames_inplace(out_dir, crop_map[vpath], label=scenario)
                print(f"[integrated][{scenario}] frames done: {vpath} -> {out_dir}")
            except FileExistsError:
                print(f"[integrated][{scenario}] frames skip (output exists): {vpath}")
            except Exception as exc:
                print(f"[integrated][{scenario}] frames 실패: {vpath} ({exc})", file=sys.stderr)

            heatmap_path = vpath.parent / heatmap_name
            if heatmap_path.exists():
                try:
                    h_out = extract_frames(video_path=heatmap_path, overwrite=overwrite_frames, target_fps=TARGET_FPS)
                    crop_frames_inplace(h_out, crop_map[vpath], label=f"{scenario}-heatmap")
                    print(f"[integrated][{scenario}] heatmap frames done: {heatmap_path} -> {h_out}")
                except FileExistsError:
                    print(f"[integrated][{scenario}] heatmap frames skip (output exists): {heatmap_path}")
                except Exception as exc:
                    print(f"[integrated][{scenario}] heatmap frames 실패: {heatmap_path} ({exc})", file=sys.stderr)
            else:
                print(f"[integrated][{scenario}] heatmap 영상이 없어 프레임 추출을 건너뜁니다: {heatmap_path}")

        speed_file = (dataset_root / scenario / "speed_course_coord.txt")
        if not speed_file.exists():
            print(f"[integrated][{scenario}] speed 파일이 없어 속도 동기화를 건너뜁니다.")
            continue
        # Use forward frames dir from the first video in this scenario
        for vpath in vids:
            frames_dir = vpath.with_suffix("")
            if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
                print(f"[integrated][{scenario}] 프레임 디렉토리 없음: {frames_dir}")
                continue
            try:
                out_speed = sync_speed_to_frames(
                    frames_dir=frames_dir,
                    speed_file=speed_file,
                    output_dir=None,
                    input_unit=input_unit,
                )
                print(f"[integrated][{scenario}] speed done: {frames_dir} -> {out_speed}")
            except Exception as exc:
                print(f"[integrated][{scenario}] speed 실패: {frames_dir} ({exc})", file=sys.stderr)


def interactive_crop_batch(
    dataset_root: Path,
    forward_name: str = "video_garmin.avi",
    heatmap_name: str = "video_saliency.avi",
    encode_now: bool = True,
) -> List[Tuple[Path, Tuple[int, int, int, int]]]:
    """
    Iterate over all uncropped forward videos under dataset_root, interactively select crop, and apply
    the same crop to paired heatmap videos. Stops when no uncropped videos remain or user presses q/Esc.
    Returns the queued crop info list.
    """
    candidates = sorted(dataset_root.rglob(forward_name))
    pending: List[Path] = []
    for vid in candidates:
        w, h, _, _ = get_video_info(vid)
        if not is_cropped_aspect(w, h):
            pending.append(vid)

    if not pending:
        print("크롭할 전방 주행 영상이 없습니다 (이미 2:1 크롭 완료).")
        return []

    window_name = "Crop selector (batch)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    queued: List[Tuple[Path, Tuple[int, int, int, int]]] = []

    for vid in pending:
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"영상 열기에 실패했습니다: {vid}", file=sys.stderr)
            continue
        ret, frame = cap.read()
        if not ret:
            print(f"영상에 프레임이 없습니다: {vid}", file=sys.stderr)
            cap.release()
            continue

        # Refresh window for the new video to avoid dangling destroyed handles.
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        height, width = frame.shape[:2]
        cv2.resizeWindow(window_name, width * 3, height * 3)
        crop_w, crop_h = largest_aspect_crop(width, height, aspect=2.0)
        x0 = (width - crop_w) // 2
        y0 = 0
        size_step = 10
        min_h = 20
        max_h = min(height, width // 2)
        paused = True

        rel_path = vid.relative_to(dataset_root)
        scenario = vid.parent.name
        instructions = (
            "Space: play/pause | Up/Down: move Y | Left/Right: move X | A/D: shrink/grow | Enter: save crop | "
            + ("q/Esc: encode queued & quit" if encode_now else "q/Esc: quit (later processing)")
        )
        title = f"{rel_path} (scenario {scenario})"
        try:
            cv2.setWindowTitle(window_name, title)
        except Exception:
            # 일부 플랫폼에서는 setWindowTitle 미지원
            pass
        while True:
            display = frame.copy()
            cv2.rectangle(display, (x0, y0), (x0 + crop_w - 1, y0 + crop_h - 1), (0, 0, 255), 2)
            cv2.putText(
                display,
                f"{rel_path} (scenario {scenario})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Crop {crop_w}x{crop_h}px, y={y0}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                instructions,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(30)
            key_low = key & 0xFF

            if key_low in (ord("q"), 27):  # q or Esc
                cap.release()
                cv2.destroyWindow(window_name)
                if encode_now:
                    print("선택을 중단합니다. 지금까지 큐에 있는 영상들을 인코딩합니다.")
                else:
                    print("선택을 중단합니다. 인코딩 없이 선택만 완료합니다.")
                break
            elif key_low == ord("x"):
                cap.release()
                cv2.destroyWindow(window_name)
                queued.clear()
                print("큐를 비우고 인코딩 없이 종료합니다.")
                return []
            elif key_low in (13, 10):  # Enter
                queued.append((vid, (x0, y0, crop_w, crop_h)))
                print(f"크롭 위치 저장: {vid} ({rel_path})")
                break
            elif key_low in (ord(" "),):
                paused = not paused
            elif key in (2490368, 63232, 82):
                y0 = max(0, y0 - 1)
            elif key in (2621440, 63233, 84):
                y0 = min(height - crop_h, y0 + 1)
            elif key in (2424832, 63234, 81):
                x0 = max(0, x0 - 1)
            elif key in (2555904, 63235, 83):
                x0 = min(width - crop_w, x0 + 1)
            elif key_low == ord("a"):
                new_h = max(min_h, crop_h - size_step)
                new_w = 2 * new_h
                center_x = x0 + crop_w // 2
                x0 = int(center_x - new_w // 2)
                if x0 < 0 or x0 + new_w > width:
                    continue
                y0 = min(y0, height - new_h)
                crop_w, crop_h = new_w, new_h
            elif key_low == ord("d"):
                new_h = min(max_h, crop_h + size_step)
                new_w = 2 * new_h
                center_x = x0 + crop_w // 2
                x0 = int(center_x - new_w // 2)
                if x0 < 0 or x0 + new_w > width:
                    continue
                y0 = min(y0, height - new_h)
                crop_w, crop_h = new_w, new_h

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    paused = True
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, 0))

        cap.release()

    cv2.destroyWindow(window_name)
    if not queued:
        print("저장된 크롭 위치가 없습니다. 인코딩 없이 종료합니다.")
        return []

    if encode_now:
        print(f"총 {len(queued)}개 영상에 대해 순차 인코딩을 시작합니다.")
        for vpath, rect in queued:
            print(f"[crop] applying to: {vpath}")
            heatmap_path = vpath.parent / heatmap_name
            crop_pair_inplace(vpath, rect, heatmap_video=heatmap_path if heatmap_path.exists() else None)
        print("모든 크롭되지 않은 전방 주행 영상을 처리했습니다.")
    return queued


def extract_frames(
    video_path: Path,
    output_dir: Path | None = None,
    overwrite: bool = False,
    pad: int | None = None,
    target_fps: float | None = TARGET_FPS,
) -> Path:
    """
    Extract frames into PNG files named by the original frame index (001.png, ...).
    Frames are uniformly subsampled using stride = round(src_fps / target_fps) when target_fps is set.
    """
    output_dir = output_dir or video_path.with_suffix("")
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = 1
    if target_fps and src_fps > 0:
        stride = max(1, round(src_fps / target_fps))
    scenario = video_path.parent.name
    total_saved = (total_frames + stride - 1) // stride if total_frames else 0
    log_interval = max(1, total_saved // 40) if total_saved else 250
    pad_width = pad or max(3, len(str(total_frames)) if total_frames else 6)

    raw_idx = 1
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (raw_idx - 1) % stride == 0:
            filename = f"{raw_idx:0{pad_width}d}.png"
            cv2.imwrite(str(output_dir / filename), frame)
            saved += 1
            if total_saved and (saved % log_interval == 0 or saved == total_saved):
                pct = (saved / total_saved) * 100
                print(f"[frames][{scenario}] {video_path.name}: {saved}/{total_saved} ({pct:.1f}%)", flush=True)
        raw_idx += 1

    cap.release()
    return output_dir


def crop_frames_inplace(frames_dir: Path, rect: Tuple[int, int, int, int], label: str) -> None:
    """
    Crop all PNG frames in frames_dir to rect and overwrite. Keeps filenames.
    """
    x0, y0, w, h = rect
    files = sorted(frames_dir.glob("*.png"))
    total = len(files)
    if total == 0:
        return
    log_interval = max(1, total // 40)
    for idx, f in enumerate(files, start=1):
        img = cv2.imread(str(f))
        if img is None:
            continue
        crop = img[y0 : y0 + h, x0 : x0 + w]
        cv2.imwrite(str(f), crop)
        if idx % log_interval == 0 or idx == total:
            pct = (idx / total) * 100
            print(f"[crop-frames][{label}] {frames_dir.name}: {idx}/{total} ({pct:.1f}%)", flush=True)


def load_speed_table(speed_file: Path) -> Dict[int, float]:
    """
    Load #frame and speed columns from speed_course_coord.txt.
    Returns a mapping {frame_index: speed_value}.
    """
    speeds: Dict[int, float] = {}
    with speed_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                continue
            try:
                frame_idx = int(parts[0])
                speed_val = float(parts[1])
                speeds[frame_idx] = speed_val
            except ValueError:
                continue
    return speeds


def sync_speed_to_frames(
    frames_dir: Path,
    speed_file: Path,
    output_dir: Path | None = None,
    input_unit: str = "kmh",
) -> Path:
    """
    Match per-frame speeds to extracted frame images and save as text files (one per frame) in m/s.
    """
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")

    speeds = load_speed_table(speed_file)
    if not speeds:
        raise RuntimeError(f"No speed entries found in {speed_file}")

    # Use the numeric stem of the first frame to determine padding.
    sample_stem = frame_files[0].stem
    pad_width = len(sample_stem)
    output_dir = output_dir or frames_dir.parent / f"{frames_dir.name}_speed"
    output_dir.mkdir(parents=True, exist_ok=True)

    unit = input_unit.lower()
    if unit not in {"kmh", "kmph", "mps"}:
        raise ValueError(f"Unsupported input unit: {input_unit}")

    conversion = 1.0 if unit == "mps" else 1000.0 / 3600.0

    scenario = frames_dir.parent.name
    eligible_frames = []
    for frame_file in frame_files:
        try:
            frame_idx = int(frame_file.stem)
        except ValueError:
            raise RuntimeError(f"Frame filename is not numeric: {frame_file.name}")
        if frame_idx in speeds:
            eligible_frames.append(frame_file)

    total = len(eligible_frames)
    if total == 0:
        raise RuntimeError("No speed files were written; check indexing alignment.")

    log_interval = max(1, total // 40)

    written = 0
    for frame_file in eligible_frames:
        frame_idx = int(frame_file.stem)

        speed_mps = speeds[frame_idx] * conversion
        out_name = f"{frame_idx:0{pad_width}d}.txt"
        with (output_dir / out_name).open("w", encoding="utf-8") as out_f:
            out_f.write(f"{speed_mps:.6f}\n")
        written += 1
        if written % log_interval == 0 or written == total:
            pct = (written / total) * 100
            print(f"[speed][{scenario}] {frames_dir.name}: {written}/{total} ({pct:.1f}%)", flush=True)

    if written == 0:
        raise RuntimeError("No speed files were written; check indexing alignment.")

    # After writing, align filenames across frames/heatmap/speed.
    heatmap_dir = frames_dir.parent / "video_saliency"
    _rename_aligned_sequences(
        frames_dir=frames_dir,
        speed_dir=output_dir,
        heatmap_dir=heatmap_dir if heatmap_dir.exists() else None,
    )

    return output_dir


def _rename_aligned_sequences(
    frames_dir: Path,
    speed_dir: Path,
    heatmap_dir: Optional[Path] = None,
) -> None:
    """
    Rename forward frames, optional heatmap frames, and speed txt to contiguous indices while keeping alignment.
    Uses the intersection of stems across available sets.
    """
    frame_files = sorted(frames_dir.glob("*.png"))
    speed_files = sorted(speed_dir.glob("*.txt"))
    heat_files = sorted(heatmap_dir.glob("*.png")) if heatmap_dir and heatmap_dir.exists() else []

    def stems(files: List[Path]) -> List[int]:
        vals = []
        for f in files:
            try:
                vals.append(int(f.stem))
            except ValueError:
                continue
        return vals

    frame_stems = set(stems(frame_files))
    speed_stems = set(stems(speed_files))
    common = frame_stems & speed_stems
    if heat_files:
        heat_stems = set(stems(heat_files))
        common &= heat_stems
    if not common:
        print(f"[rename] 공통 프레임이 없어 이름 정렬을 건너뜁니다: {frames_dir}")
        return

    common_sorted = sorted(common)
    pad_width = max(3, len(str(len(common_sorted))))

    def plan(files: List[Path], allowed: set[int]) -> List[Tuple[Path, Path]]:
        mapping: List[Tuple[Path, Path]] = []
        stem_to_file = {int(f.stem): f for f in files if f.stem.isdigit()}
        for new_idx, stem in enumerate(common_sorted, start=1):
            if stem not in allowed or stem not in stem_to_file:
                continue
            dest = stem_to_file[stem].with_name(f"{new_idx:0{pad_width}d}{stem_to_file[stem].suffix}")
            mapping.append((stem_to_file[stem], dest))
        return mapping

    maps = []
    maps.extend(plan(frame_files, frame_stems))
    if heat_files:
        maps.extend(plan(heat_files, set(stems(heat_files))))
    maps.extend(plan(speed_files, speed_stems))

    # Two-phase rename to avoid collisions
    temp_paths: List[Tuple[Path, Path]] = []
    for src, dest in maps:
        tmp = src.with_name(f".renametmp_{uuid.uuid4().hex}{src.suffix}")
        src.rename(tmp)
        temp_paths.append((tmp, dest))

    for tmp, dest in temp_paths:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(dest)

    print(f"[rename] 정렬 완료: {frames_dir} (총 {len(common_sorted)}개)")


def batch_sync_speed(
    dataset_root: Path,
) -> None:
    """
    For each scenario under dataset_root that has speed_course_coord.txt and a 2:1 forward frames directory,
    generate per-frame speed text files.
    """
    speed_files = sorted(dataset_root.rglob("speed_course_coord.txt"))
    if not speed_files:
        print(f"[speed] speed_course_coord.txt를 찾지 못했습니다 (root={dataset_root})")
        return

    for sfile in speed_files:
        scenario_dir = sfile.parent
        scenario = scenario_dir.name
        # candidate forward frame dirs (already extracted)
        candidates = [
            p
            for p in scenario_dir.glob("video_garmin*")
            if p.is_dir() and any(p.glob("*.png"))
        ]
        if not candidates:
            print(f"[speed][{scenario}] 프레임 디렉토리가 없어 건너뜁니다.")
            continue
        for frames_dir in candidates:
            try:
                output = sync_speed_to_frames(
                    frames_dir=frames_dir,
                    speed_file=sfile,
                    output_dir=None,
                    input_unit="kmh",
                )
                print(f"[speed][{scenario}] 완료: {frames_dir} -> {output}")
            except Exception as exc:
                print(f"[speed][{scenario}] 실패: {frames_dir} ({exc})", file=sys.stderr)


def _prompt_path(prompt: str, must_exist: bool = True, default: Path | None = None, is_dir: bool = False) -> Path:
    """
    Prompt the user for a path with optional existence check.
    """
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            path = default
        else:
            if not raw:
                print("경로를 입력하세요.")
                continue
            path = Path(raw).expanduser()
        if must_exist:
            if not path.exists():
                print(f"경로가 존재하지 않습니다: {path}")
                continue
            if is_dir and not path.is_dir():
                print(f"디렉토리가 아닙니다: {path}")
                continue
            if not is_dir and path.is_dir():
                print(f"파일 경로가 필요합니다: {path}")
                continue
        return path


def _prompt_bool(prompt: str, default: bool = False) -> bool:
    raw = input(prompt).strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true", "t"}


def interactive_cli() -> None:
    """
    Interactive prompt mode (no command-line args needed).
    """
    try:
        print("DR(eye)VE 전처리 통합 모드입니다. (frames→crop→speed)")
        while True:
            mode = input("경로 입력 방식 선택 (1: data/<name> 입력, 2: 절대/상대 전체 경로 입력) [1/2]: ").strip() or "1"
            if mode == "1":
                dataset_name_raw = input("데이터셋 이름을 입력하세요 (data/<name>, 기본: DREYEVE_DATA): ").strip()
                dataset_name = dataset_name_raw if dataset_name_raw else "DREYEVE_DATA"
                root_candidate = Path("data") / dataset_name
            elif mode == "2":
                root_candidate = Path(input("데이터셋 루트 전체 경로를 입력하세요: ").strip())
            else:
                print("1 또는 2를 입력하세요.")
                continue
            if not root_candidate.exists() or not root_candidate.is_dir():
                print(f"경로가 존재하지 않습니다: {root_candidate}")
                continue
            confirm = input(f"선택된 경로: {root_candidate}  진행할까요? [y/N]: ").strip().lower()
            if confirm in {"y", "yes"}:
                root = root_candidate
                break
            print("다시 입력하세요.")
        overwrite = _prompt_bool("프레임 디렉토리가 있어도 덮어쓸까요? [y/N]: ", default=False)
        # DREYEVE speed_course_coord.txt는 km/h 단위이므로 입력 단위를 고정합니다.
        integrated_pipeline(dataset_root=root, overwrite_frames=overwrite, input_unit="kmh")
    except Exception as exc:  # pragma: no cover
        print(f"오류가 발생했습니다: {exc}", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DR(eye)VE preprocessing tools for Sim-Lingo pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    crop_p = subparsers.add_parser(
        "crop", help="Interactively crop videos. Supports single video or batch under a dataset root."
    )
    crop_p.add_argument(
        "video",
        type=Path,
        nargs="?",
        help="(옵션) 단일 비디오 경로. 지정하지 않으면 --root 하위 uncropped 전방 영상들을 순차 처리.",
    )
    crop_p.add_argument(
        "--root",
        type=Path,
        help="데이터셋 루트. 지정하면 video_garmin.avi 중 aspect 2:1이 아닌 것만 순차 크롭.",
    )
    crop_p.add_argument(
        "--output",
        type=Path,
        help="Optional output video path. Defaults to <stem>_cropped.<suffix> next to the input.",
    )
    frames_p = subparsers.add_parser(
        "frames", help="Extract frames from a video into numbered PNG files."
    )
    frames_p.add_argument(
        "video",
        type=Path,
        nargs="?",
        help="Input video path (optional if --root is provided).",
    )
    frames_p.add_argument(
        "--root",
        type=Path,
        help="데이터셋 루트. 지정하면 2:1인 전방/히트맵 영상을 모두 프레임으로 추출.",
    )
    frames_p.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for frames. Defaults to a directory named after the video stem.",
    )
    frames_p.add_argument(
        "--pad",
        type=int,
        help="Zero-padding width for filenames (default: inferred from frame count, min=3).",
    )
    frames_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty directory.",
    )

    sync_p = subparsers.add_parser(
        "sync-speed",
        help="Sync per-frame speed values to extracted frames and save as m/s text files.",
    )
    sync_p.add_argument(
        "frames_dir",
        type=Path,
        nargs="?",
        help="Directory containing extracted frame PNGs (numeric filenames). Optional if --root is provided.",
    )
    sync_p.add_argument(
        "speed_file",
        type=Path,
        nargs="?",
        help="Path to speed_course_coord.txt for the same run. Optional if --root is provided.",
    )
    sync_p.add_argument(
        "--root",
        type=Path,
        help="데이터셋 루트. 지정하면 2:1 프레임 디렉토리가 존재하는 시나리오에 대해 speed 텍스트를 일괄 생성.",
    )
    sync_p.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for per-frame speed text files. Defaults to <frames_dir>_speed.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        interactive_cli()
        return

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "crop":
        if args.root:
            interactive_crop_batch(
                dataset_root=args.root,
            )
        elif args.video:
            output = interactive_crop(
                video_path=args.video,
                output_path=args.output,
            )
            print(f"Cropped video saved to: {output}")
        else:
            parser.error("crop 모드에서는 video를 지정하거나 --root를 지정해야 합니다.")
    elif args.command == "frames":
        if args.root:
            batch_extract_frames(dataset_root=args.root, overwrite=args.overwrite)
        elif args.video:
            output = extract_frames(
                video_path=args.video,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                pad=args.pad,
            )
            print(f"Frames saved to: {output}")
        else:
            parser.error("frames 모드에서는 video를 지정하거나 --root를 지정해야 합니다.")
    elif args.command == "sync-speed":
        if args.root:
            batch_sync_speed(dataset_root=args.root)
        elif args.frames_dir and args.speed_file:
            output = sync_speed_to_frames(
                frames_dir=args.frames_dir,
                speed_file=args.speed_file,
                output_dir=args.output_dir,
                input_unit="kmh",
            )
            print(f"Speed text files saved to: {output}")
        else:
            parser.error("sync-speed 모드에서는 frames_dir/speed_file을 지정하거나 --root를 지정해야 합니다.")
    else:
        parser.error("Unknown command.")


if __name__ == "__main__":
    main()
