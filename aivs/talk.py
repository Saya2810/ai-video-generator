"""Talking-character shots with InfiniteTalk (MeiGen-AI, Apache-2.0) ported to Apple MLX (scripts/talk/).

For a shot where a character speaks on screen, the whole shot is generated from its keyframe and the spoken line:
head, face, expressions, gestures and mouth all follow the audio (instead of Wan motion + a repainted mouth).
Base model: Wan2.1-I2V-14B-480P with the lightx2v 4-step distillation (Apache-2.0), audio features from
chinese-wav2vec2-base (MIT), image features from Wan's CLIP ViT-H encoder (Apache-2.0).

Two processes per shot: scripts/talk/features.py (PyTorch, envs/lipsync) -> scripts/talk/run.py (MLX, envs/wan,
transformer blocks streamed from disk). Setup: scripts/setup_talk.sh.
"""
from __future__ import annotations

from pathlib import Path

from . import config, fetch
from .util import digest, log, need, run

TALK_PY = config.ROOT / "envs" / "wan" / "bin" / "python"
FEAT_PY = config.ROOT / "envs" / "lipsync" / "bin" / "python"
RUNNER = config.ROOT / "scripts" / "talk" / "run.py"
FEATURES = config.ROOT / "scripts" / "talk" / "features.py"
MODEL_DIR = config.ROOT / "models" / "local" / "infinitetalk-mlx-q8"
REPOS = {"wav2vec": "TencentGameMate/chinese-wav2vec2-base", "clip": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
         "vae": "Comfy-Org/Wan_2.1_ComfyUI_repackaged"}
SIZES = {"16:9": (832, 480), "9:16": (480, 832), "1:1": (640, 640)}   # Wan 2.1 480p sizes (W, H)
FPS = 25
VERSION = 1


def paths() -> dict:
    loc = lambda repo: Path(fetch.resolve_model(repo))
    return {"model": MODEL_DIR, "wav2vec": loc(REPOS["wav2vec"]), "clip": loc(REPOS["clip"]) / "image_encoder",
            "vae": loc(REPOS["vae"]) / "split_files" / "vae" / "wan_2.1_vae.safetensors",
            "t5": Path(fetch.resolve_model(config.MODELS["i2v_wan"])) / "t5_encoder.safetensors"}


def unavailable_reason() -> str | None:
    if not TALK_PY.exists() or not FEAT_PY.exists():
        return "envs/wan or envs/lipsync is missing — run scripts/setup_wan.sh and scripts/setup_lipsync.sh"
    p = paths()
    missing = [k for k, f in p.items() if not Path(f).exists()]
    if not (MODEL_DIR / "config.json").exists():
        missing.append("converted model")
    if missing:
        return f"InfiniteTalk files missing ({', '.join(missing)}) — run: bash scripts/setup_talk.sh"
    return None


def clip_path(shot: dict, settings: dict, cdir: Path, akey: str, offset: float, frames: int, prompt: str) -> Path:
    size = SIZES[settings["aspect"]]
    key = digest([shot["_ikey"], akey, offset, frames, size, prompt, settings["seed"],
                  settings.get("talk_steps", "1000,750,500,250"), settings.get("talk_audio_guide", 2.0), VERSION])
    return cdir / f"{shot['id']}-{key}.talk.mp4"


def generate(shot: dict, audio16: Path, out: Path, *, prompt: str, frames: int, settings: dict, workdir: Path) -> None:
    """Native InfiniteTalk clip (25 fps, Wan 480p size) covering `frames` frames at 25 fps."""
    p = paths()
    size = SIZES[settings["aspect"]]
    feats = workdir / f"talk-{shot['id']}.npz"
    log(f"talking shot {shot['id']}: InfiniteTalk {size[0]}x{size[1]}, {frames / FPS:.1f}s")
    run([FEAT_PY, "-u", FEATURES, "--wav2vec", p["wav2vec"], "--clip", p["clip"], "--audio", audio16,
         "--image", shot["image"], "--size", f"{size[1]}x{size[0]}", "--out", feats],
        log_path=workdir / "talk.log", env=config.worker_env())
    tmp = out.with_suffix(".tmp.mp4")
    run([TALK_PY, "-u", RUNNER, "--model", p["model"], "--t5", p["t5"], "--vae", p["vae"], "--features", feats,
         "--prompt", prompt, "--frames", frames, "--steps", settings.get("talk_steps", "1000,750,500,250"),
         "--audio-guide", settings.get("talk_audio_guide", 2.0), "--seed", settings["seed"], "--output", tmp,
         "--report", workdir / f"talk-{shot['id']}.json"],
        log_path=workdir / "talk.log", env=config.worker_env())
    tmp.replace(out)
    feats.unlink(missing_ok=True)


def conform(raw: Path, out: Path, frames: int, size: tuple[int, int], fps: int, log_path: Path) -> None:
    """25 fps -> `fps` by repeating frames (no interpolation), cover-scale to the output size, exact frame count."""
    vf = (f"fps={fps},scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={size[0]}:{size[1]},unsharp=5:5:0.5:5:5:0.0,tpad=stop_mode=clone:stop_duration=2,"
          f"scale=out_color_matrix=bt709:out_range=tv,format=yuv420p")
    run([need("ffmpeg"), "-y", "-v", "error", "-i", raw, "-vf", vf, "-frames:v", frames, "-r", fps,
         "-c:v", "libx264", "-preset", "fast", "-crf", 14, "-colorspace", "bt709", "-color_primaries", "bt709",
         "-color_trc", "bt709", out], log_path=log_path)
