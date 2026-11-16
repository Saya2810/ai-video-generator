#!/bin/bash
# Lip-sync backend for speaking characters: LatentSync 1.6 on Apple MLX, with MediaPipe face landmarks.
#   bash scripts/setup_lipsync.sh
# Adds envs/lipsync (PyTorch + MLX + MediaPipe, ~3 GB) and ~6 GB of model files (models/local/latentsync).
# Licences: LatentSync code Apache-2.0, weights OpenRAIL++ (commercial use allowed with use restrictions);
# latentsync-mlx Apache-2.0; sd-vae-ft-mse MIT; Whisper tiny MIT; MediaPipe Face Landmarker Apache-2.0.
# InsightFace is NOT installed or downloaded (its pretrained models are non-commercial).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
[ -x envs/studio/bin/aivs ] || { echo "Run scripts/setup.sh first."; exit 1; }
LS_COMMIT=a229c3948406bc2cf6eaf4873e662e70c6a04746    # bytedance/LatentSync (Apache-2.0)
MLX_COMMIT=8370f1ff2b7c240c9ea78cbeb134c254336f6806   # sb1992/latentsync-mlx (Apache-2.0)
V=vendor/latentsync; C=models/local/latentsync
mkdir -p "$V" "$C/whisper"

fetch_tar() {  # tarball instead of git clone (clones died on slow links here)
  local repo=$1 commit=$2 dest=$3
  [ -f "$dest/README.md" ] && return 0
  mkdir -p "$dest"
  curl -fL --retry 5 -o "$dest.tar.gz" "https://github.com/$repo/archive/$commit.tar.gz"
  tar -xzf "$dest.tar.gz" -C "$dest" --strip-components 1 && rm "$dest.tar.gz"
}
fetch_tar bytedance/LatentSync "$LS_COMMIT" "$V/LatentSync"
fetch_tar sb1992/latentsync-mlx "$MLX_COMMIT" "$V/latentsync-mlx"

[ -d envs/lipsync ] || uv venv -q --python 3.12 envs/lipsync
uv pip install -q -p envs/lipsync mlx torch torchvision kornia einops opencv-python soundfile tqdm \
  "transformers<5" mediapipe matplotlib imageio ffmpeg-python diffusers safetensors scipy

# weights: only what inference needs (the HF repo also holds ~4 GB of training/eval files)
envs/studio/bin/python - <<'EOF'
from aivs import fetch
fetch.fetch("ByteDance/LatentSync-1.6", include=("latentsync_unet.pt", "whisper/tiny.pt"))
fetch.fetch("stabilityai/sd-vae-ft-mse", include=("config.json", "diffusion_pytorch_model.safetensors"))
EOF
[ -f "$C/face_landmarker.task" ] || curl -fL --retry 5 -o "$C/face_landmarker.task" \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
# OpenCV Zoo SFace (Apache-2.0): identifies the speaker's face when several people are in a shot
[ -f "$C/face_recognition_sface_2021dec.onnx" ] || { curl -fL --retry 5 -o "$C/face_recognition_sface_2021dec.onnx.part" \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
  && mv "$C/face_recognition_sface_2021dec.onnx.part" "$C/face_recognition_sface_2021dec.onnx"; }
cp -n models/local/ByteDance--LatentSync-1.6/whisper/tiny.pt "$C/whisper/tiny.pt" 2>/dev/null || true

if [ ! -f "$C/latentsync_unet_mlx.safetensors" ] || [ ! -f "$C/vae_mlx.safetensors" ]; then
  echo "converting weights to MLX (one time, a few minutes)"
  envs/lipsync/bin/python - <<EOF
import sys
sys.path.insert(0, "$V/latentsync-mlx")
from latentsync_mlx.convert_weights import convert_unet_weights, convert_vae_weights
convert_unet_weights("models/local/ByteDance--LatentSync-1.6/latentsync_unet.pt", "$C/latentsync_unet_mlx.safetensors")
convert_vae_weights("models/local/stabilityai--sd-vae-ft-mse", "$C/vae_mlx.safetensors")
EOF
fi
envs/studio/bin/python -c "from aivs import lipsync; r = lipsync.unavailable_reason(); print('lip-sync ready' if not r else 'NOT READY: ' + r)"
