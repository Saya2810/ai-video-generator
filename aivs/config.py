"""Central configuration: paths, model ids, render presets. Everything is overridable by env vars."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("AIVS_ROOT", Path(__file__).resolve().parent.parent))
PROJECTS = Path(os.environ.get("AIVS_PROJECTS", ROOT / "projects"))
HF_HOME = Path(os.environ.get("AIVS_HF_HOME", ROOT / "models" / "hf"))

# All defaults are Apache-2.0 / MIT weights (see THIRD_PARTY.md).
MODELS = {
    "planner": os.environ.get("AIVS_MODEL_PLANNER", "mlx-community/Qwen3-4B-Instruct-2507-4bit"),
    "image": os.environ.get("AIVS_MODEL_IMAGE", "Runpod/FLUX.2-klein-4B-mflux-4bit"),
    "tts_kokoro": os.environ.get("AIVS_MODEL_KOKORO", "mlx-community/Kokoro-82M-bf16"),
    "tts_chatterbox": os.environ.get("AIVS_MODEL_CHATTERBOX", "mlx-community/chatterbox-multilingual-v3"),
    "stt": os.environ.get("AIVS_MODEL_STT", "mlx-community/whisper-large-v3-turbo-asr-fp16"),
    "depth": os.environ.get("AIVS_MODEL_DEPTH", "depth-anything/Depth-Anything-V2-Small-hf"),
    "i2v_wan": os.environ.get("AIVS_MODEL_WAN", "Anes1032/Wan2.2-TI2V-5B-mlx-q8"),  # optional, see aivs/i2v.py
    # character voices for non-English dialogue (Apache-2.0); fetched automatically on first use
    "voice_design": os.environ.get("AIVS_MODEL_VOICE_DESIGN", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"),
}

# Kokoro handles English without extra system packages; every other language goes to Chatterbox Multilingual.
KOKORO_LANGS = {"en": "a", "en-us": "a", "en-gb": "b"}
KOKORO_DEFAULT_VOICE = {"a": "af_heart", "b": "bf_emma"}
CHATTERBOX_LANGS = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt",
    "ru", "sv", "sw", "tr", "zh",
}

ASPECTS = {  # name -> (output WxH, keyframe WxH (multiples of 16))
    "16:9": ((1920, 1080), (1920, 1088)),
    "9:16": ((1080, 1920), (1088, 1920)),
    "1:1": ((1080, 1080), (1440, 1440)),
}
DRAFT_SCALE = 2 / 3  # --draft renders 1280x720 from 1280x720 keyframes

FPS = 30
XFADE = 0.5      # seconds of crossfade between shots
PRE_ROLL = 0.30  # silence before narration inside a shot
POST_ROLL = 0.40  # silence after narration inside a shot
INTRO = 0.50     # extra lead-in for the first shot
OUTRO = 1.20     # extra tail for the last shot
DIALOGUE_PRE = 0.12   # dialogue lines follow each other more tightly than narration
DIALOGUE_POST = 0.28
LOUDNESS = -14.0  # LUFS integrated (YouTube/Spotify reference)

DEFAULT_STYLE = (
    "cinematic film still, natural light, shallow depth of field, rich color grading, highly detailed, 35mm photography"
)


def worker_env() -> dict:
    env = dict(os.environ)
    env["HF_HOME"] = str(HF_HOME)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env
