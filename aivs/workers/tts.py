"""Narration worker: one TTS model, many lines. Usage: python -m aivs.workers.tts job.json result.json

job = {"engine": "kokoro"|"chatterbox", "model": str, "lang": "en", "voice": str|None, "speed": 1.0,
       "ref_audio": path|None, "exaggeration": float, "cfg_weight": float, "seed": int,
       "items": [{"id": "s01", "text": "...", "out": "/abs/s01.wav", "take": 0}]}
An item may override voice, ref_audio, cfg_weight, exaggeration and speed (one voice per character).
"""
from __future__ import annotations

import json
import sys

import numpy as np

OUT_SR = 48000


def trim_silence(x: np.ndarray, sr: int, thresh_db: float = -45.0, keep: float = 0.03) -> np.ndarray:
    if x.size == 0:
        return x
    win = max(int(sr * 0.01), 1)
    n = x.size // win
    if n < 3:
        return x
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    loud = np.where(20 * np.log10(rms / (np.max(np.abs(x)) + 1e-9) + 1e-9) > thresh_db)[0]
    if loud.size == 0:
        return x
    a = max(loud[0] * win - int(keep * sr), 0)
    b = min((loud[-1] + 1) * win + int(keep * sr), x.size)
    return x[a:b]


def resample(x: np.ndarray, sr: int, out_sr: int) -> np.ndarray:
    if sr == out_sr:
        return x
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, out_sr)
        return resample_poly(x, out_sr // g, sr // g).astype(np.float32)
    except ImportError:
        t = np.arange(int(x.size * out_sr / sr)) * (sr / out_sr)
        return np.interp(t, np.arange(x.size), x).astype(np.float32)


def finish(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - float(np.mean(x)) if x.size else x
    x = trim_silence(x, sr)
    x = resample(x, sr, OUT_SR)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x * (0.89 / peak)  # -1 dBFS peak; final loudness is set in the mix
    fade = min(int(0.008 * OUT_SR), x.size // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    return x


def main(job_path: str, out_path: str) -> None:
    import mlx.core as mx
    import soundfile as sf
    from mlx_audio.tts.utils import load_model

    job = json.load(open(job_path, encoding="utf-8"))
    model = load_model(job["model"])
    engine = job["engine"]
    results = []
    for item in job["items"]:
        mx.random.seed(int(job.get("seed", 0)) + 1000 * int(item.get("take", 0)) + (hash(item["id"]) % 997))
        opt = lambda k, d=None: item[k] if k in item else job.get(k, d)  # per-line voice settings (characters)
        kwargs = dict(text=item["text"], speed=float(opt("speed", 1.0)), verbose=False)
        if engine == "kokoro":
            kwargs.update(voice=opt("voice"), lang_code=job["lang"])
        else:  # chatterbox multilingual
            kwargs.update(lang_code=job["lang"], exaggeration=float(opt("exaggeration", 0.4)),
                          cfg_weight=float(opt("cfg_weight", 0.5)),
                          temperature=max(0.5, 0.8 - 0.1 * int(item.get("take", 0))))
            if opt("ref_audio"):
                kwargs["ref_audio"] = opt("ref_audio")
        chunks, sr = [], None
        for res in model.generate(**kwargs):
            chunks.append(np.asarray(res.audio, dtype=np.float32).reshape(-1))
            sr = int(res.sample_rate)
        if not chunks:
            raise RuntimeError(f"TTS returned no audio for {item['id']}")
        gap = np.zeros(int(0.12 * sr), dtype=np.float32)
        audio = chunks[0] if len(chunks) == 1 else np.concatenate([c for ch in chunks for c in (ch, gap)][:-1])
        audio = finish(audio, sr)
        sf.write(item["out"], audio, OUT_SR, subtype="PCM_16")
        results.append({"id": item["id"], "out": item["out"], "duration": round(audio.size / OUT_SR, 3)})
        print(f"tts {item['id']}: {audio.size / OUT_SR:.2f}s", flush=True)
    json.dump({"items": results}, open(out_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
