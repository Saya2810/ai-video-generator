#!/bin/bash
# Talking characters: InfiniteTalk (audio-driven Wan2.1-I2V-14B) on Apple MLX.
#   bash scripts/setup_talk.sh          (needs scripts/setup_wan.sh and scripts/setup_lipsync.sh first)
# Downloads ~30 GB (safetensors only), converts to ~20 GB of 8-bit MLX weights in models/local/infinitetalk-mlx-q8,
# which are streamed block by block at run time (works on 16 GB).
# Licences: InfiniteTalk Apache-2.0; Wan2.1-I2V-14B-480P + lightx2v step distillation Apache-2.0;
# chinese-wav2vec2-base MIT; Wan2.1 VAE / CLIP image encoder Apache-2.0. All commercial use allowed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[ -x envs/wan/bin/python ] || { echo "Run scripts/setup_wan.sh first."; exit 1; }
[ -x envs/lipsync/bin/python ] || { echo "Run scripts/setup_lipsync.sh first."; exit 1; }
CONN=${AIVS_FETCH_CONNECTIONS:-16}
envs/studio/bin/python - <<EOF
from aivs import fetch
fetch.fetch("TencentGameMate/chinese-wav2vec2-base", $CONN, include=("config.json", "preprocessor_config.json", "pytorch_model.bin"))
fetch.fetch("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", $CONN, include=("image_encoder/config.json", "image_encoder/model.safetensors", "image_processor/preprocessor_config.json"))
fetch.fetch("Comfy-Org/Wan_2.1_ComfyUI_repackaged", $CONN, include=("split_files/vae/wan_2.1_vae.safetensors",))
fetch.fetch("lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v", $CONN,
            include=("config.json", "distill_int8/config.json", "distill_int8/block_", "distill_int8/non_block"))
fetch.fetch("MeiGen-AI/InfiniteTalk", $CONN, include=("single/infinitetalk.safetensors",))
EOF
L=models/local
if [ ! -f $L/infinitetalk-mlx-q8/config.json ]; then
  echo "converting to MLX 8-bit (one time, ~10-20 min)"
  envs/wan/bin/python scripts/talk/convert.py \
    --distill $L/lightx2v--Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/distill_int8 \
    --adapter $L/MeiGen-AI--InfiniteTalk/single/infinitetalk.safetensors --out $L/infinitetalk-mlx-q8 --bits 8
fi
envs/studio/bin/python -c "from aivs import talk; r = talk.unavailable_reason(); print('talking characters ready' if not r else 'NOT READY: ' + r)"
