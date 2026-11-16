"""Convert InfiniteTalk weights to the MLX port (scripts/talk/model.py), one transformer block at a time.

Inputs (safetensors only):
  * lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v, distill_int8/: Wan2.1-I2V-14B-480P with the
    lightx2v 4-step / CFG distillation merged (Apache-2.0); per block `block_<i>.safetensors` with int8 Linear
    weights + per-output-channel bf16 `weight_scale`, and `non_block.safetensors` (fp32)
  * MeiGen-AI/InfiniteTalk single/infinitetalk.safetensors: the audio adapter (fp32, Apache-2.0)

Output (<out>/): `non_block.safetensors` (bf16, fp32 modulation), `blocks/block_<i>.safetensors` with the block's
Linear layers quantised to MLX affine `bits`-bit (group 64) and everything else in bf16 (modulation fp32), and
`config.json`. The runner streams the block files from disk, so only one block is in memory at a time.

  envs/wan/bin/python scripts/talk/convert.py --distill DIR/distill_int8 --adapter FILE --out DIR [--bits 8]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

GROUP = 64
NUM_BLOCKS = 40
# reference key -> MLX module path (model.py); keys not listed keep their name
RENAME = [(r"^patch_embedding\.", "patch_embedding_proj."), (r"^text_embedding\.0\.", "text_embedding_0."),
          (r"^text_embedding\.2\.", "text_embedding_1."), (r"^time_embedding\.0\.", "time_embedding_0."),
          (r"^time_embedding\.2\.", "time_embedding_1."), (r"^time_projection\.1\.", "time_projection."),
          (r"^img_emb\.proj\.0\.", "img_emb.norm_in."), (r"^img_emb\.proj\.1\.", "img_emb.fc1."),
          (r"^img_emb\.proj\.3\.", "img_emb.fc2."), (r"^img_emb\.proj\.4\.", "img_emb.norm_out."),
          (r"\.ffn\.0\.", ".ffn.fc1."), (r"\.ffn\.2\.", ".ffn.fc2.")]
FP32_KEYS = re.compile(r"(^|\.)modulation$")


def rename(key: str) -> str:
    for pat, rep in RENAME:
        key = re.sub(pat, rep, key)
    return key


def read(path: Path) -> dict:
    out = {}
    with safe_open(str(path), framework="mlx") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def dequant_int8(t: dict) -> dict:
    """lightx2v int8: weight (int8 [out, in]) * weight_scale ([out, 1]) -> bf16; other tensors unchanged."""
    out = {}
    for k, v in t.items():
        if k.endswith(".weight_scale"):
            continue
        sk = k + "_scale"
        if k.endswith(".weight") and sk in t:
            v = (v.astype(mx.float32) * t[sk].astype(mx.float32)).astype(mx.bfloat16)
        out[k] = v
    return out


def finish(t: dict, quantize: bool, bits: int) -> dict:
    out = {}
    for k, v in t.items():
        name = rename(k)
        if FP32_KEYS.search(name):
            out[name] = v.astype(mx.float32)
            continue
        if name == "patch_embedding_proj.weight":
            v = v.reshape(v.shape[0], -1)  # Conv3d [dim, 36, 1, 2, 2] -> Linear [dim, 144] (mlx-video patchify order)
        v = v.astype(mx.bfloat16)
        if quantize and name.endswith(".weight") and v.ndim == 2 and v.shape[1] % GROUP == 0:
            w, s, b = mx.quantize(v, group_size=GROUP, bits=bits)
            base = name[: -len(".weight")]
            out[name], out[base + ".scales"], out[base + ".biases"] = w, s, b
        else:
            out[name] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distill", required=True, help="lightx2v distill_int8 directory")
    ap.add_argument("--adapter", required=True, help="InfiniteTalk single/infinitetalk.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=8, choices=[4, 6, 8])
    a = ap.parse_args()
    src, out = Path(a.distill), Path(a.out)
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with safe_open(a.adapter, framework="mlx") as ad:
        akeys = list(ad.keys())
        non_block = dequant_int8(read(src / "non_block.safetensors"))
        non_block.update({k: ad.get_tensor(k) for k in akeys if not k.startswith("blocks.")})
        nb = finish(non_block, quantize=False, bits=a.bits)
        mx.save_safetensors(str(out / "non_block.safetensors"), nb)
        del nb, non_block
        for i in range(NUM_BLOCKS):
            dst = out / "blocks" / f"block_{i}.safetensors"
            if dst.exists():
                continue
            t = dequant_int8(read(src / f"block_{i}.safetensors"))
            t.update({k: ad.get_tensor(k) for k in akeys if k.startswith(f"blocks.{i}.")})
            blk = finish(t, quantize=True, bits=a.bits)
            prefix = f"blocks.{i}."
            blk = {k[len(prefix):]: v for k, v in blk.items()}   # block-local names
            mx.eval(blk)
            mx.save_safetensors(str(dst.with_suffix(".tmp.safetensors")), blk)
            dst.with_suffix(".tmp.safetensors").replace(dst)
            print(f"block {i}: {len(blk)} tensors ({time.time() - t0:.0f}s)", flush=True)
            del t, blk
            mx.clear_cache()
    (out / "config.json").write_text(json.dumps({"bits": a.bits, "group_size": GROUP, "num_blocks": NUM_BLOCKS,
                                                 "source": [str(src), a.adapter]}, indent=2))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
