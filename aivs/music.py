"""Background music.

"ambient": built-in procedural pad synthesiser (pure numpy/scipy, no model, no licence questions).
A file path: use the user's own track. "none": no music. ACE-Step support lives in music_acestep.py (optional).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 48000

# chord progressions as MIDI note stacks, one chord per bar
MOODS = {
    "calm": [[38, 45, 54, 57, 64], [35, 42, 50, 57, 62], [31, 38, 47, 55, 62], [33, 40, 49, 57, 64]],
    "wonder": [[36, 43, 52, 59, 64], [33, 40, 52, 57, 60], [29, 41, 48, 57, 64], [31, 43, 50, 59, 62]],
    "uplifting": [[36, 43, 52, 55, 64], [31, 43, 50, 59, 62], [33, 45, 52, 60, 64], [29, 41, 53, 57, 60]],
    "dark": [[33, 40, 48, 52, 57], [29, 36, 45, 53, 57], [26, 38, 45, 50, 53], [28, 40, 47, 52, 56]],
}


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def ambient(duration: float, mood: str = "calm", seed: int = 0, bar: float = 8.0) -> np.ndarray:
    """Return a stereo float32 array (n, 2) with an evolving pad of the requested length."""
    from scipy.signal import butter, fftconvolve, sosfilt

    rng = np.random.default_rng(seed)
    chords = MOODS.get(mood, MOODS["calm"])
    final_sr, SR = globals()["SR"], 24000  # synthesise at 24 kHz (content is low-passed anyway), upsample at the end
    n = int((duration + 4.0) * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros((n, 2), dtype=np.float64)
    xfade = bar * 0.45
    n_bars = int(np.ceil((duration + 4.0) / bar))
    for b in range(n_bars):
        chord = chords[b % len(chords)]
        start, end = b * bar - xfade / 2, (b + 1) * bar + xfade / 2
        env = np.clip((t - start) / xfade, 0, 1) * np.clip((end - t) / xfade, 0, 1)
        env = np.sin(env * np.pi / 2) ** 2
        active = env > 0
        if not active.any():
            continue
        ta, ea = t[active], env[active]
        for k, note in enumerate(chord):
            level = (0.9 if k == 0 else 0.55) / (1 + 0.25 * k)
            for det in (-5.0, 0.0, 6.0):  # cents
                f = _hz(note + det / 100)
                phase = rng.uniform(0, 2 * np.pi)
                lfo = 1 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.05, 0.16) * ta + rng.uniform(0, 6.28))
                sig = np.zeros_like(ta)
                for h in range(1, 6):
                    if f * h > 5000:
                        break
                    sig += np.sin(2 * np.pi * f * h * ta + phase * h) / h ** (1.9 if note < 45 else 1.6)
                pan = np.clip(0.5 + rng.uniform(-0.35, 0.35) + 0.12 * det / 6, 0.05, 0.95)
                voice = sig * lfo * ea * level
                out[active, 0] += voice * np.cos(pan * np.pi / 2)
                out[active, 1] += voice * np.sin(pan * np.pi / 2)
        # slow shimmer an octave above the top note
        f = _hz(chord[-1] + 12)
        shim = np.sin(2 * np.pi * f * ta + rng.uniform(0, 6.28)) * ea * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * ta)) * 0.06
        out[active, 0] += shim
        out[active, 1] += shim
    sos = butter(2, 1700, btype="low", fs=SR, output="sos")
    out = sosfilt(sos, out, axis=0)
    # reverb: convolution with exponentially decaying stereo noise
    ir_n = int(3.0 * SR)
    ir = rng.standard_normal((ir_n, 2)) * np.exp(-np.arange(ir_n) / (0.9 * SR))[:, None]
    ir = sosfilt(butter(2, 3500, btype="low", fs=SR, output="sos"), ir, axis=0)
    wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:n] for c in (0, 1)], axis=1)
    wet /= np.max(np.abs(wet)) + 1e-9
    out = 0.55 * out / (np.max(np.abs(out)) + 1e-9) + 0.6 * wet
    out = out[int(1.5 * SR): int(1.5 * SR) + int(duration * SR)]
    out = sosfilt(butter(2, 35, btype="high", fs=SR, output="sos"), out, axis=0)
    from scipy.signal import resample_poly
    out = resample_poly(out, final_sr // SR, 1, axis=0)
    out *= 0.5 / (np.max(np.abs(out)) + 1e-9)
    return out.astype(np.float32)


def make(kind: str, duration: float, out: Path, seed: int = 0) -> Path | None:
    """kind: 'none' | mood name | path to an audio file. Returns a path usable as an ffmpeg input, or None."""
    if not kind or kind == "none":
        return None
    if kind in MOODS or kind == "ambient":
        import soundfile as sf
        sf.write(out, ambient(duration, "calm" if kind == "ambient" else kind, seed), SR, subtype="PCM_16")
        return out
    path = Path(kind).expanduser()
    if not path.exists():
        raise SystemExit(f"music: '{kind}' is neither a mood ({', '.join(MOODS)}) nor an existing audio file")
    return path
