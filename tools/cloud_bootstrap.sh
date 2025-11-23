#!/usr/bin/env bash
# Quick bootstrap for a fresh cloud GPU Ubuntu node.
# Installs system deps, git-lfs, Python deps, and pulls LFS assets.
# Usage (one-liner after SSH): bash tools/cloud_bootstrap.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[*] Repository root: $REPO_ROOT"

if command -v apt-get >/dev/null 2>&1; then
  echo "[*] Installing system packages (libgl1 ffmpeg git-lfs)..."
  sudo apt-get update -y
  sudo apt-get install -y libgl1 ffmpeg git-lfs
else
  echo "[!] apt-get not found; please install libgl1, ffmpeg, git-lfs manually."
fi

echo "[*] Initializing git-lfs and pulling LFS objects..."
git lfs install
git lfs pull || echo "[!] git lfs pull failed; ensure network/LFS access."

echo "[*] Installing Python dependencies (requirements.txt)..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

cat <<'EOS'
[*] Bootstrap complete.
Remaining steps (manual):
  - Ensure CUDA 12.1-compatible driver is available for torch 2.3.1+cu121.
  - If checkpoints are missing, you can clone the model directly:
      git lfs clone https://huggingface.co/RenzKa/simlingo checkpoints/simlingo/simlingo/checkpoints/
  - Place preprocessed data under data/<dataset>/<scenario>/ with:
      video_garmin/ (frames), video_saliency/ (heatmaps), video_garmin_speed/ (m/s speeds)
  - First inference run will download HuggingFace models (InternVL2-1B) if not cached.
EOS
