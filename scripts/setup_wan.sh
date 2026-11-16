#!/bin/bash
# OPTIONAL: generative image-to-video backend (Wan 2.2 TI2V-5B, Apache-2.0) for `aivs ... --motion wan`.
# Adds a separate environment (envs/wan) and ~20 GB of model files. Expect ~10 minutes per generated shot on 16 GB.
#   bash scripts/setup_wan.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
[ -x envs/studio/bin/aivs ] || { echo "Run scripts/setup.sh first."; exit 1; }
COMMIT=87db56a51758fefb748a359b90a5283bb8ba4837   # Blaizzy/mlx-video (MIT), the revision tested here
if [ ! -f vendor/mlx-video/pyproject.toml ]; then
  # tarball instead of `git clone`: a full clone failed repeatedly on slow links
  mkdir -p vendor/mlx-video
  curl -fL --retry 5 -o vendor/mlx-video.tar.gz "https://github.com/Blaizzy/mlx-video/archive/$COMMIT.tar.gz"
  tar -xzf vendor/mlx-video.tar.gz -C vendor/mlx-video --strip-components 1 && rm vendor/mlx-video.tar.gz
fi
[ -d envs/wan ] || uv venv -q --python 3.12 envs/wan
uv pip install -q -p envs/wan ./vendor/mlx-video
envs/studio/bin/aivs fetch Anes1032/Wan2.2-TI2V-5B-mlx-q8
echo; echo "Wan backend ready. Example:  ./bin/aivs render <project> --motion wan --wan-shots s02"
