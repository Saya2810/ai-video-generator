"""Turn keyframes into moving clips.

Backend "parallax" (default): 2.5D camera. A depth map (Depth Anything V2 Small) displaces near and far pixels by
different amounts while the virtual camera dollies / trucks, so the still gains real depth motion.
Backend "camera": flat sub-pixel zoom / pan (no depth model needed).
A generative image-to-video backend would plug in at pipeline.stage_motion with the same contract:
(keyframe, frame count, size, fps) -> silent clip of exactly that many frames.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from .util import need

# (zoom_start, zoom_end, pan_x_start, pan_x_end, pan_y_start, pan_y_end); pan in [-1, 1] of the available margin
MOVES = {
    "push_in": (1.00, 1.14, 0.0, 0.0, 0.0, 0.0),
    "pull_out": (1.14, 1.00, 0.0, 0.0, 0.0, 0.0),
    "pan_left": (1.10, 1.10, 1.0, -1.0, 0.0, 0.0),
    "pan_right": (1.10, 1.10, -1.0, 1.0, 0.0, 0.0),
    "pan_up": (1.10, 1.12, 0.0, 0.0, 1.0, -1.0),
    "pan_down": (1.10, 1.12, 0.0, 0.0, -1.0, 1.0),
    "static": (1.03, 1.03, 0.0, 0.0, 0.0, 0.0),
}


def encoder_args(crf: int = 14, preset: str = "fast") -> list[str]:
    # explicit BT.709 conversion + tags: swscale would otherwise use BT.601 for RGB input and shift HD colours
    return ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-preset", preset,
            "-crf", str(crf), "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


def _trajectory(move: str, frames: int, fps: int, speed_ref_seconds: float):
    """Return per-frame (zoom, pan_x, pan_y) with duration-normalised speed."""
    z0, z1, px0, px1, py0, py1 = MOVES.get(move, MOVES["push_in"])
    amount = min(max((frames / fps) / speed_ref_seconds, 0.6), 1.5)
    zmid = (z0 + z1) / 2
    z0, z1 = zmid + (z0 - zmid) * amount, zmid + (z1 - zmid) * amount
    zfloor = min(z0, z1)
    if zfloor < 1.0:
        z0, z1 = z0 + (1.0 - zfloor), z1 + (1.0 - zfloor)
    px0, px1, py0, py1 = (min(max(p * amount, -1.0), 1.0) for p in (px0, px1, py0, py1))
    for i in range(frames):
        t = i / max(frames - 1, 1)
        yield t, z0 + (z1 - z0) * t, px0 + (px1 - px0) * t, py0 + (py1 - py0) * t


def _ffmpeg_sink(out: Path, size: tuple[int, int], fps: int, pix: str = "rgb24") -> subprocess.Popen:
    return subprocess.Popen(
        [need("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{size[0]}x{size[1]}",
         "-r", str(fps), "-i", "-", *encoder_args(), "-r", str(fps), str(out)], stdin=subprocess.PIPE)


def parallax_clip(image: Path, depth: Path, out: Path, frames: int, size: tuple[int, int], fps: int, move: str,
                  strength: float = 1.0, speed_ref_seconds: float = 6.0) -> None:
    """2.5D move: flat camera trajectory + depth-proportional displacement (near pixels move more)."""
    import cv2
    import numpy as np

    W, H = size
    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    d = cv2.imread(str(depth), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    sh, sw = img.shape[:2]
    if d.shape[:2] != (sh, sw):
        d = cv2.resize(d, (sw, sh), interpolation=cv2.INTER_CUBIC)
    # soften depth edges so displacement stretches gently instead of tearing at silhouettes
    k = max(int(sw * 0.012) | 1, 3)
    d = cv2.GaussianBlur(cv2.dilate(d, np.ones((k, k), np.uint8)), (0, 0), sw * 0.008)
    focus = float(np.median(d))                      # the plane that stays put
    rel = d - focus
    cover = max(W / sw, H / sh)
    max_shift = 0.022 * sw * strength                # px of parallax between nearest and farthest content
    zoom_gain = 0.06 * strength                      # extra magnification of near content on dolly moves
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    proc = _ffmpeg_sink(out, size, fps, "bgr24")
    try:
        for t, z, px, py in _trajectory(move, frames, fps, speed_ref_seconds):
            vw, vh = W / (cover * z), H / (cover * z)
            mx, my = (sw - vw) / 2, (sh - vh) / 2
            cx, cy = sw / 2 + mx * px, sh / 2 + my * py
            ax = cx - vw / 2 + gx * (vw / W)         # flat-camera source coordinates
            ay = cy - vh / 2 + gy * (vh / H)
            r = cv2.remap(rel, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            u = 2 * t - 1                            # -1 .. 1 across the clip
            dolly = (z - 1.07) * zoom_gain / 0.07    # signed: pushes near content outward as we move in
            sx = ax + r * (px * max_shift * 1.0 + 0.0) - (ax - cx) * r * dolly
            sy = ay + r * (py * max_shift * 1.0 + 0.0) - (ay - cy) * r * dolly
            if move in ("push_in", "pull_out", "static"):
                sx += r * u * max_shift * 0.35       # slight truck so pure dolly shots still show parallax
            frame = cv2.remap(img, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while encoding {out}")


def camera_clip(image: Path, out: Path, frames: int, size: tuple[int, int], fps: int, move: str,
                speed_ref_seconds: float = 6.0) -> None:
    """Render `frames` frames of a smooth camera move over a still image.

    Motion speed is normalised to `speed_ref_seconds` so long shots do not crawl and short shots do not rush:
    the full move is covered in ~6 s and scaled proportionally for other lengths (clamped).
    """
    W, H = size
    img = Image.open(image).convert("RGB")
    sw, sh = img.size
    cover = max(W / sw, H / sh)           # source px -> output px at zoom 1.0
    proc = _ffmpeg_sink(out, size, fps)
    try:
        for _, z, px, py in _trajectory(move, frames, fps, speed_ref_seconds):
            vw, vh = W / (cover * z), H / (cover * z)      # visible window in source px
            mx, my = (sw - vw) / 2, (sh - vh) / 2          # free margin each side
            x0, y0 = sw / 2 + mx * px - vw / 2, sh / 2 + my * py - vh / 2
            frame = img.transform((W, H), Image.AFFINE, (vw / W, 0.0, x0, 0.0, vh / H, y0), resample=Image.BICUBIC)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while encoding {out}")
