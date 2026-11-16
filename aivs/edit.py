"""Automated edit: crossfaded picture, narration placement, ducked music, loudness normalisation, captions, export."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import config
from .util import need, run

SR = 48000


def narration_track(items: list[dict], total_seconds: float, out: Path) -> None:
    """items: [{audio: path, start: seconds}] -> one mono 48 kHz wav with every line at its exact start sample."""
    import soundfile as sf

    buf = np.zeros(int(round(total_seconds * SR)) + SR // 10, dtype=np.float32)
    for it in items:
        x, sr = sf.read(it["audio"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            raise RuntimeError(f"{it['audio']} is {sr} Hz, expected {SR}")
        a = int(round(it["start"] * SR))
        b = min(a + x.size, buf.size)
        buf[a:b] += x[: b - a]
    sf.write(out, buf[: int(round(total_seconds * SR))], SR, subtype="PCM_16")


def _measure(ff: str, path: Path) -> dict:
    probe = subprocess.run([ff, "-hide_banner", "-nostats", "-i", str(path), "-af",
                            "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", probe.stderr, flags=re.S)
    if not m:
        raise RuntimeError("loudness measurement failed:\n" + probe.stderr[-800:])
    d = json.loads(m.group(0))
    return {"lufs": float(d["input_i"]), "true_peak_db": float(d["input_tp"]), "lra": float(d["input_lra"])}


def mix_audio(narration: Path, music: Path | None, total_seconds: float, out: Path, log_path: Path,
              music_gain_db: float = -15.0, lufs: float = config.LOUDNESS) -> dict:
    """Voice (compressed) + side-chain ducked music, then static gain + limiter to the EBU R128 target.

    A static gain keeps the natural dynamics of the read; loudnorm's dynamic mode would ride the level audibly.
    """
    ff = need("ffmpeg")
    raw = out.with_name("mix_raw.wav")
    fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    voice = f"{fmt},highpass=f=70,acompressor=threshold=-22dB:ratio=3:attack=6:release=150:knee=5:makeup=1"
    if music:
        fade_out = max(total_seconds - 2.5, 0)
        graph = (
            f"[0:a]{voice},asplit=2[vox][key];"
            f"[1:a]{fmt},volume={music_gain_db}dB,afade=t=in:d=1.5,afade=t=out:st={fade_out:.3f}:d=2.5[mus];"
            f"[mus][key]sidechaincompress=threshold=0.02:ratio=6:attack=40:release=700:makeup=1[duck];"
            f"[vox][duck]amix=inputs=2:normalize=0:duration=first[a]"
        )
        cmd = [ff, "-y", "-v", "error", "-i", narration, "-stream_loop", "-1", "-i", music, "-filter_complex", graph,
               "-map", "[a]", "-t", f"{total_seconds:.3f}", "-c:a", "pcm_f32le", raw]
    else:
        cmd = [ff, "-y", "-v", "error", "-i", narration, "-af", voice, "-t", f"{total_seconds:.3f}",
               "-c:a", "pcm_f32le", raw]
    run(cmd, log_path=log_path)
    before = _measure(ff, raw)
    gain = lufs - before["lufs"]
    after = before
    for _ in range(3):  # the limiter shaves a little loudness off; compensate until within 0.3 LU
        chain = f"volume={gain:.2f}dB,alimiter=limit=0.8:attack=4:release=60:level=false,atrim=start=0.004,asetpts=PTS-STARTPTS"  # trim = limiter look-ahead
        run([ff, "-y", "-v", "error", "-i", raw, "-af", chain, "-ar", "48000", "-c:a", "pcm_s24le", out], log_path=log_path)
        after = _measure(ff, out)
        if abs(after["lufs"] - lufs) <= 0.3:
            break
        gain += lufs - after["lufs"]
    raw.unlink(missing_ok=True)
    return {"input_lufs": before["lufs"], "output_lufs": after["lufs"], "true_peak_db": after["true_peak_db"],
            "lra": after["lra"], "gain_db": round(gain, 2), "mode": "static gain + limiter"}


def assemble(clips: list[dict], audio: Path, out: Path, *, fps: int, xfade_frames: int, total_frames: int,
             ass: Path | None, log_path: Path, crf: int = 18, preset: str = "slow", transition: str = "fade",
             title: str = "") -> None:
    """clips: [{path, cut_frame}] in order; every clip is (slot + xfade) frames long."""
    ff = need("ffmpeg")
    cmd = [ff, "-y", "-v", "error"]
    for c in clips:
        cmd += ["-i", Path(c["path"]).resolve()]
    cmd += ["-i", audio.resolve()]
    xf = xfade_frames / fps
    parts = [f"[{i}:v]settb=AVTB,fps={fps},format=yuv420p,setpts=PTS-STARTPTS[c{i}]" for i in range(len(clips))]
    last = "c0"
    for i in range(1, len(clips)):
        parts.append(f"[{last}][c{i}]xfade=transition={transition}:duration={xf:.6f}:"
                     f"offset={clips[i]['cut_frame'] / fps:.6f}[x{i}]")
        last = f"x{i}"
    total = total_frames / fps
    tail = f"[{last}]fade=t=in:st=0:d=0.5,fade=t=out:st={total - 0.8:.3f}:d=0.8"
    if ass is not None:
        fonts = ass.parent / "fonts"  # bundled OFL caption font, copied next to the .ass so no path needs escaping
        fonts.mkdir(exist_ok=True)
        for f in (Path(__file__).parent / "assets" / "fonts").glob("*.ttf"):
            if not (fonts / f.name).exists():
                shutil.copy(f, fonts / f.name)
        tail += f",subtitles={ass.name}:fontsdir=fonts"
    parts.append(tail + "[v]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", f"{len(clips)}:a",
            "-frames:v", total_frames, "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-af", f"afade=t=out:st={total - 0.8:.3f}:d=0.8",
            "-t", f"{total:.6f}", "-movflags", "+faststart", "-metadata", f"title={title}", out.resolve()]
    cmd = [str(c) for c in cmd]
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        # cwd = caption folder so the subtitles filter gets a path without characters that need escaping
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ass.parent) if ass else None)
    if proc.returncode != 0:
        raise RuntimeError(f"final assembly failed; see {log_path}")
