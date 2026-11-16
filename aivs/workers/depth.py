"""Monocular depth worker (Depth Anything V2 Small, Apache-2.0). Usage: python -m aivs.workers.depth job.json result.json

job = {"model": str, "items": [{"id": "s01", "image": "/abs/s01.png", "out": "/abs/s01.depth.png"}]}
Writes 16-bit PNGs where bright = near.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main(job_path: str, out_path: str) -> None:
    import cv2
    import torch
    from transformers import AutoModelForDepthEstimation

    job = json.load(open(job_path, encoding="utf-8"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForDepthEstimation.from_pretrained(job["model"]).to(device).eval()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    done = []
    for item in job["items"]:
        bgr = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read {item['image']}")
        h, w = bgr.shape[:2]
        scale = 518 / min(h, w)
        nw, nh = max(int(round(w * scale / 14)) * 14, 14), max(int(round(h * scale / 14)) * 14, 14)
        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]
        with torch.no_grad():
            depth = model(pixel_values=torch.from_numpy(x).to(device)).predicted_depth[0].float().cpu().numpy()
        lo, hi = np.percentile(depth, 1), np.percentile(depth, 99)
        depth = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)  # relative inverse depth: 1 = near
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(item["out"], (np.clip(depth, 0, 1) * 65535).astype(np.uint16))
        done.append({"id": item["id"], "out": item["out"]})
        print(f"depth {item['id']}", flush=True)
    json.dump({"items": done}, open(out_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
