"""Voice design worker: one ~10 s voice sample per character from a text description (Qwen3-TTS VoiceDesign,
Apache-2.0). The samples are then cloned line by line with Chatterbox, so a character keeps one voice.

Usage: python -m aivs.workers.voice_design job.json result.json
job = {"model": str, "seed": int,
       "items": [{"id": "anna", "text": "...", "language": "german", "instruct": "...", "out": "/abs/anna.wav"}]}
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .tts import finish, OUT_SR


def main(job_path: str, out_path: str) -> None:
    import mlx.core as mx
    import soundfile as sf
    from mlx_audio.tts.utils import load_model

    job = json.load(open(job_path, encoding="utf-8"))
    model = load_model(job["model"])
    results = []
    for n, item in enumerate(job["items"]):
        mx.random.seed(int(job.get("seed", 0)) + 7919 * n + sum(map(ord, item["id"])))
        chunks, sr = [], None
        for res in model.generate_voice_design(text=item["text"], instruct=item["instruct"],
                                               language=item.get("language", "auto"), verbose=False):
            chunks.append(np.asarray(res.audio, dtype=np.float32).reshape(-1))
            sr = int(res.sample_rate)
        if not chunks:
            raise RuntimeError(f"voice design returned no audio for {item['id']}")
        audio = finish(np.concatenate(chunks), sr)
        sf.write(item["out"], audio, OUT_SR, subtype="PCM_16")
        results.append({"id": item["id"], "out": item["out"], "duration": round(audio.size / OUT_SR, 2)})
        print(f"voice {item['id']}: {audio.size / OUT_SR:.1f}s", flush=True)
    json.dump({"items": results}, open(out_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
