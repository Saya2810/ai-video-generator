"""Generative image-to-video backend: Wan 2.2 TI2V-5B (Apache-2.0) through mlx-video. Default motion backend.

Measured on a MacBook Pro M5 / 16 GB: one 49-frame 832x480 clip (2.0 s at 24 fps) takes about 9.5 minutes
(text encoder 19 s, 20 denoising steps 290 s, VAE decode 259 s) at a 13.3 GB memory peak.
The clip is then slowed down with motion-compensated interpolation and upscaled to the shot's size.

It lives in its own environment (envs/wan, see scripts/setup_wan.sh) so the main studio never depends on it.
"""
from __future__ import annotations

import math
from pathlib import Path

from . import config, fetch
from .util import digest, log, need, run

WAN_PY = config.ROOT / "envs" / "wan" / "bin" / "python"
RUNNER = config.ROOT / "scripts" / "wan_run.py"
SIZES = {"16:9": (832, 480), "9:16": (480, 832), "1:1": (640, 640)}  # multiples of 32, ~0.4 MP
FRAMES, WAN_FPS = 49, 24   # FRAMES: the one clip of "stretch" mode
# "native" mode: frames per Wan run. 121 (5 s, the model's longest) peaked at 14.4 GB on the 16 GB M5 (denoise
# ~44 s/step, VAE decode 14 min); 81 (3.4 s) costs about the same per second of video with more memory headroom.
MAX_CHUNK = 81
NATIVE_VERSION = 1         # bump when native-mode output changes
CAMERA_HINT = {
    "push_in": "the camera slowly pushes in", "pull_out": "the camera slowly pulls back",
    "pan_left": "the camera slowly pans left", "pan_right": "the camera slowly pans right",
    "pan_up": "the camera slowly tilts up", "pan_down": "the camera slowly tilts down", "static": "static camera",
}


def unavailable_reason() -> str | None:
    if not WAN_PY.exists():
        return "envs/wan is missing — run: bash scripts/setup_wan.sh"
    if not Path(fetch.resolve_model(config.MODELS["i2v_wan"])).is_dir():
        return f"model {config.MODELS['i2v_wan']} is not downloaded — run: bash scripts/setup_wan.sh"
    return None


def build_prompt(shot: dict, style: str) -> str:
    visual = shot.get("_wan_visual") or shot["visual"]  # reference names replaced by descriptions (pipeline)
    parts = [visual.rstrip(". ")]
    if shot.get("_wan_action"):  # e.g. a speaking character: talking, expressions, gestures (pipeline)
        parts.append(shot["_wan_action"].rstrip(". "))
    if CAMERA_HINT.get(shot.get("motion")):
        parts.append(CAMERA_HINT[shot["motion"]])
    parts.append(f"Realistic, lively, natural human motion, {style.rstrip('. ')}")
    return ". ".join(parts) + "."


def _frames_4n1(n: int) -> int:
    return 4 * math.ceil((max(n, 5) - 1) / 4) + 1


def chunk_plan(seconds: float, max_chunk: int = MAX_CHUNK) -> list[int]:
    """Frame counts of the Wan runs that together cover `seconds` at 24 fps without slowing anything down.
    Each run after the first starts from the previous run's last frame (generated twice, kept once), so k runs of
    n frames cover k*(n-1)+1 frames. Runs are balanced (no tiny last run with odd motion)."""
    need_frames = math.ceil(seconds * WAN_FPS) + 1
    k = max(1, math.ceil((need_frames - 1) / (max_chunk - 1)))
    n = min(max_chunk, _frames_4n1(math.ceil((need_frames - 1) / k) + 1))
    return [n] * k


def _native(settings: dict) -> bool:
    return settings.get("wan_length", "native") != "stretch"


def raw_paths(shot: dict, board: dict, settings: dict, cdir: Path, seconds: float) -> list[Path]:
    """Cache paths of the Wan runs for a shot: one 2 s clip ("stretch" mode, independent of the shot's length) or
    the chunks that cover the shot at real speed ("native" mode, default)."""
    size = SIZES[settings["aspect"]]
    prompt = build_prompt(shot, board["style"])
    steps = int(settings.get("wan_steps", 20))
    if not _native(settings):
        key = digest([shot["_ikey"], prompt, size, FRAMES, steps, settings["seed"], config.MODELS["i2v_wan"]])
        return [cdir / f"{shot['id']}-{key}.wan.mp4"]
    paths, prev = [], None
    for i, n in enumerate(chunk_plan(seconds, int(settings.get("wan_chunk_frames", MAX_CHUNK)))):
        key = digest([shot["_ikey"], prompt, size, n, steps, settings["seed"], config.MODELS["i2v_wan"], i,
                      prev.name if prev else None, NATIVE_VERSION])
        prev = cdir / f"{shot['id']}-{key}.wan{i}.mp4"
        paths.append(prev)
    return paths


def _last_frame(clip: Path, out: Path, log_path: Path) -> None:
    run([need("ffmpeg"), "-y", "-v", "error", "-sseof", "-0.5", "-i", clip, "-update", "1", "-q:v", 1, out],
        log_path=log_path)


def raw_clips(shot: dict, board: dict, settings: dict, cdir: Path, workdir: Path, seconds: float) -> list[Path]:
    """Generate (or reuse) the Wan run(s) for a shot; chunk i > 0 starts from the last frame of chunk i - 1."""
    paths = raw_paths(shot, board, settings, cdir, seconds)
    size = SIZES[settings["aspect"]]
    prompt = build_prompt(shot, board["style"])
    steps = int(settings.get("wan_steps", 20))
    image = Path(shot["image"])
    for i, out in enumerate(paths):
        if i > 0:
            image = workdir / f"wan-{shot['id']}-start{i}.png"
            _last_frame(paths[i - 1], image, workdir / "edit.log")
        if out.exists():
            continue
        n = FRAMES if not _native(settings) else chunk_plan(seconds, int(settings.get("wan_chunk_frames", MAX_CHUNK)))[i]
        log(f"wan image-to-video {shot['id']}" + (f" part {i + 1}/{len(paths)}" if len(paths) > 1 else "")
            + f": {size[0]}x{size[1]}, {n} frames ({n / WAN_FPS:.1f}s), {steps} steps")
        tmp = out.with_suffix(".tmp.mp4")
        run([WAN_PY, "-u", RUNNER, "--model-dir", fetch.resolve_model(config.MODELS["i2v_wan"]), "--image", image,
             "--prompt", prompt, "--width", size[0], "--height", size[1], "--num-frames", n, "--steps", steps,
             "--seed", settings["seed"], "--tile-px", 256, "--tile-frames", 24, "--output", tmp,
             "--report", workdir / f"wan-{shot['id']}-{i}.json"],
            log_path=workdir / "wan.log", env=config.worker_env())
        tmp.replace(out)
    return paths


def conform(raws: list[Path], out: Path, frames: int, size: tuple[int, int], fps: int, log_path: Path,
            native: bool = True) -> float:
    """Turn the Wan run(s) into exactly `frames` frames at `fps` and the output size. Returns the slow-down factor.
    native: chunks joined (the repeated first frame of every later chunk dropped), 24 -> fps by repeating frames,
    no motion interpolation (no ghosting). stretch: the one 2 s clip slowed down with motion interpolation."""
    tail = (f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase:flags=lanczos,crop={size[0]}:{size[1]},"
            f"unsharp=5:5:0.6:5:5:0.0,tpad=stop_mode=clone:stop_duration=2,"
            f"scale=out_color_matrix=bt709:out_range=tv,format=yuv420p")
    if native:
        stretch = 1.0
        parts = [f"[{i}:v]" + ("trim=start_frame=1,setpts=PTS-STARTPTS," if i else "") + f"settb=AVTB,fps={WAN_FPS}[c{i}]"
                 for i in range(len(raws))]
        graph = ";".join(parts) + ";" + "".join(f"[c{i}]" for i in range(len(raws))) + \
            f"concat=n={len(raws)}:v=1:a=0,fps={fps},{tail}[v]"
        inputs = [a for r in raws for a in ("-i", r)]
        run([need("ffmpeg"), "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]",
             "-frames:v", frames, "-r", fps, "-c:v", "libx264", "-preset", "fast", "-crf", 14, "-colorspace", "bt709",
             "-color_primaries", "bt709", "-color_trc", "bt709", out], log_path=log_path)
        return stretch
    stretch = (frames / fps) / (FRAMES / WAN_FPS)
    vf = f"setpts={stretch:.5f}*PTS,minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,{tail}"
    run([need("ffmpeg"), "-y", "-v", "error", "-i", raws[0], "-vf", vf, "-frames:v", frames, "-r", fps,
         "-c:v", "libx264", "-preset", "fast", "-crf", 14, "-colorspace", "bt709", "-color_primaries", "bt709",
         "-color_trc", "bt709", out], log_path=log_path)
    return stretch
