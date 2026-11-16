"""Lip-sync backend: LatentSync 1.6 (ByteDance; code Apache-2.0, weights OpenRAIL++) through the Apple-MLX port
latentsync-mlx (Apache-2.0), with Google MediaPipe Face Landmarker (Apache-2.0) instead of InsightFace, whose
pretrained models are licensed for non-commercial research only.

It re-renders the mouth region of an existing clip (the shot after motion, e.g. the Wan clip) so that it matches
the shot's spoken line. Runs in its own environment (envs/lipsync, see scripts/setup_lipsync.sh) via
scripts/lipsync_run.py, one process per shot, so the studio never depends on PyTorch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .util import log, need, run

LS_PY = config.ROOT / "envs" / "lipsync" / "bin" / "python"
RUNNER = config.ROOT / "scripts" / "lipsync_run.py"
VENDOR = config.ROOT / "vendor" / "latentsync"
CKPT = config.ROOT / "models" / "local" / "latentsync"
NEEDED = ["latentsync_unet_mlx.safetensors", "vae_mlx.safetensors", "whisper/tiny.pt", "face_landmarker.task",
          "face_recognition_sface_2021dec.onnx"]
SR = 16000
VERSION = 2  # bump when the runner's output changes, to invalidate cached lip-synced clips


def unavailable_reason() -> str | None:
    if not LS_PY.exists():
        return "envs/lipsync is missing — run: bash scripts/setup_lipsync.sh"
    missing = [n for n in NEEDED if not (CKPT / n).exists()]
    if missing or not (VENDOR / "LatentSync").is_dir() or not (VENDOR / "latentsync-mlx").is_dir():
        return f"lip-sync files missing ({', '.join(missing) or 'vendor code'}) — run: bash scripts/setup_lipsync.sh"
    return None


def line_audio(audio_48k: Path, offset: float, seconds: float, out: Path) -> None:
    """The shot's spoken line at its position inside the clip, silence elsewhere, 16 kHz mono, clip-long.
    Silence keeps the mouth closed before and after the line."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    x, sr = sf.read(str(audio_48k), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    g = gcd(sr, SR)
    x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    buf = np.zeros(int(round(seconds * SR)), dtype=np.float32)
    a = max(int(round(offset * SR)), 0)
    b = min(a + x.size, buf.size)
    if b > a:
        buf[a:b] = x[: b - a]
    sf.write(str(out), buf, SR, subtype="PCM_16")


def run_shot(clip: Path, audio16: Path, out: Path, *, frames: int, size: tuple[int, int], fps: int, resolution: int,
             steps: int, seed: int, workdir: Path, shot_id: str, ref_face: Path | None = None,
             other_faces: list[Path] | None = None) -> None:
    """LatentSync on one clip, then conform its 25 fps output back to exactly `frames` frames at `fps`.
    `ref_face`: photo/portrait of the speaker, `other_faces`: those of the other characters; they decide whose
    mouth is animated when several people are in the shot."""
    workdir.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.mp4")
    # measured on an M5 / 16 GB at 512 px, 20 steps: ~3.7 min per 16-frame chunk at 25 fps (UNet ~85 % of it)
    chunks = -(-round(frames / fps * 25) // 16)
    est = chunks * 3.7 * (0.15 + 0.85 * steps / 20) * (1 if resolution >= 512 else 0.3)
    log(f"lip-sync {shot_id}: {frames / fps:.1f}s at {resolution}px, {steps} steps (about {est:.0f} min)")
    run([LS_PY, "-u", RUNNER, "--vendor", VENDOR, "--ckpt", CKPT, "--video", clip, "--audio", audio16,
         "--output", raw, "--resolution", resolution, "--steps", steps, "--seed", seed,
         "--temp", workdir / f"lipsync-{shot_id}", "--report", workdir / f"lipsync-{shot_id}.json",
         *(["--ref-face", ref_face, "--other-faces", *(other_faces or [])] if ref_face else [])],
        log_path=workdir / "lipsync.log", env=config.worker_env())
    if not raw.exists():
        raise RuntimeError(f"lip-sync wrote no file; see {workdir / 'lipsync.log'}")
    tmp = out.with_suffix(".tmp.mp4")
    run([need("ffmpeg"), "-y", "-v", "error", "-i", raw, "-an",
         "-vf", f"fps={fps},scale={size[0]}:{size[1]}:flags=lanczos,tpad=stop_mode=clone:stop_duration=2,"
                f"scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
         "-frames:v", frames, "-r", fps, "-c:v", "libx264", "-preset", "fast", "-crf", 14,
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", tmp],
        log_path=workdir / "edit.log")
    tmp.replace(out)
    raw.unlink(missing_ok=True)
