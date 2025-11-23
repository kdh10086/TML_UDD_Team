#!/usr/bin/env bash
# Quick bootstrap for a fresh cloud GPU Ubuntu node.
# Installs system deps, git-lfs, Python deps, and pulls LFS assets.
# Usage (one-liner after SSH): bash tools/cloud_bootstrap.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[*] Repository root: $REPO_ROOT"

# Add provided public key to ~/.ssh/authorized_keys if not already present
KEYS=(
  "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBgSJupdBIRqkb5rJmsHRxqzLDstbwRTUvF15soBuzal hyun"
  "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB9TP2KEGvbodSbGxmBNAUFDB4ZWdGNc/Fe7DGHXkxEc Ryu"
)
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
AUTH="$HOME/.ssh/authorized_keys"
touch "$AUTH"
chmod 600 "$AUTH"
for KEY in "${KEYS[@]}"; do
  if ! grep -q "$KEY" "$AUTH"; then
    echo "$KEY" >> "$AUTH"
    echo "[*] Added public key to authorized_keys: $KEY"
  else
    echo "[*] Public key already present: $KEY"
  fi
done

# Configure Git to store credentials to avoid repeated token prompts
if command -v git >/dev/null 2>&1; then
  git config --global credential.helper store
  echo "[*] Git credential helper set to 'store' (tokens will be cached in ~/.git-credentials)."
fi

if command -v apt-get >/dev/null 2>&1; then
  echo "[*] Installing system packages (libgl1 ffmpeg git-lfs)..."
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y libgl1 ffmpeg git-lfs
  else
    apt-get update -y
    apt-get install -y libgl1 ffmpeg git-lfs
  fi
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
