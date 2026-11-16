#!/bin/bash
# AI Video Studio — one-time setup on Apple Silicon. Everything stays inside this folder (envs/, models/).
#   bash scripts/setup.sh            # environment + all models of the standard pipeline (~12 GB)
#   bash scripts/setup.sh --no-models
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
[ "$(uname -sm)" = "Darwin arm64" ] || { echo "This setup targets Apple Silicon Macs (MLX)."; exit 1; }
command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
command -v ffmpeg >/dev/null || brew install ffmpeg
command -v uv >/dev/null || brew install uv
[ -d envs/studio ] || uv venv -q --python 3.12 envs/studio
uv pip install -q -p envs/studio -e .
# English text front-end for Kokoro (spaCy model, MIT). Installed up front so the first render never waits on it.
envs/studio/bin/python -c "import en_core_web_sm" 2>/dev/null || uv pip install -q -p envs/studio --no-deps \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
mkdir -p bin && ln -sf "$ROOT/envs/studio/bin/aivs" bin/aivs
if [ "${1:-}" != "--no-models" ]; then envs/studio/bin/aivs fetch; fi
echo; echo "Setup complete. Try:  ./bin/aivs ui     or     ./bin/aivs make --storyboard examples/night-migration.json"
