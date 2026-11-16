"""Speech check + word timing worker (Whisper). Usage: python -m aivs.workers.stt job.json result.json

job = {"model": str, "lang": "en", "items": [{"id": "s01", "audio": "/abs/s01.wav"}]}
result = {"items": [{"id", "text", "words": [{"word", "start", "end"}]}]}
"""
from __future__ import annotations

import json
import sys


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def main(job_path: str, out_path: str) -> None:
    from mlx_audio.stt.utils import load_model

    job = json.load(open(job_path, encoding="utf-8"))
    model = load_model(job["model"])
    lang = job.get("lang", "en").split("-")[0]
    out = []
    for item in job["items"]:
        res = model.generate(item["audio"], language=lang, word_timestamps=True, verbose=False,
                             condition_on_previous_text=False, temperature=0.0)
        words = []
        for seg in _get(res, "segments", []) or []:
            for w in _get(seg, "words", []) or []:
                tok = str(_get(w, "word", "")).strip()
                if tok:
                    words.append({"word": tok, "start": float(_get(w, "start", 0.0)), "end": float(_get(w, "end", 0.0))})
        text = str(_get(res, "text", "")).strip()
        out.append({"id": item["id"], "text": text, "words": words})
        print(f"stt {item['id']}: {text[:90]}", flush=True)
    json.dump({"items": out}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
