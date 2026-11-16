"""Planner LLM worker. Usage: python -m aivs.workers.llm job.json result.json

job = {"model": str, "requests": [{"id": str, "system": str, "user": str, "max_tokens": int, "temperature": float}]}
result = {"responses": {id: text}}
"""
from __future__ import annotations

import json
import re
import sys


def main(job_path: str, out_path: str) -> None:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    job = json.load(open(job_path, encoding="utf-8"))
    model, tok = load(job["model"])
    responses = {}
    for req in job["requests"]:
        messages = [{"role": "system", "content": req["system"]}, {"role": "user", "content": req["user"]}]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text = generate(model, tok, prompt=prompt, max_tokens=int(req.get("max_tokens", 1200)),
                        sampler=make_sampler(temp=float(req.get("temperature", 0.6)), top_p=0.9), verbose=False)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        responses[req["id"]] = text
        print(f"llm {req['id']}: {len(text)} chars", flush=True)
    json.dump({"responses": responses}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
