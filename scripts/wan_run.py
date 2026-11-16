"""Memory-frugal front-end for mlx-video's Wan 2.2 TI2V-5B (image-to-video) on 16 GB Macs. EXPERIMENTAL.

Run with the separate environment:  envs/wan/bin/python scripts/wan_run.py --help

Why not `python -m mlx_video.models.wan_2.generate` directly?
  * upstream loads the UMT5-XXL text encoder as float32 (~23 GB) — impossible on 16 GB. Here it is quantised to
    8 bit layer by layer while loading (~6 GB) and the resulting prompt embeddings are cached on disk, so the
    encoder is only ever needed once per distinct prompt.
  * `--probe` skips the text encoder entirely (random conditioning) to measure speed / memory of the denoiser + VAE.
  * IPv4 is preferred for the tokenizer download (urllib/httpx wait for timeouts on networks with broken IPv6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

_orig_gai = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: sorted(_orig_gai(*a, **k), key=lambda r: r[0] != socket.AF_INET)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402


def load_t5_quantized(model_path: Path, config, bits: int = 8):
    """Build the T5 encoder with quantised Linear/Embedding layers, materialising one module at a time."""
    from mlx_video.models.wan_2.text_encoder import T5Encoder

    enc = T5Encoder(vocab_size=config.t5_vocab_size, dim=config.t5_dim, dim_attn=config.t5_dim_attn,
                    dim_ffn=config.t5_dim_ffn, num_heads=config.t5_num_heads, num_layers=config.t5_num_layers,
                    num_buckets=config.t5_num_buckets, shared_pos=False)
    enc.load_weights(list(mx.load(str(model_path)).items()))  # lazy: nothing is read yet
    nn.quantize(enc, group_size=64, bits=bits)                # lazy: quantisation graphs only
    for _, module in enc.named_modules():
        if not module.children() or isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            mx.eval(module.parameters())
            mx.clear_cache()
    mx.eval(enc.parameters())
    return enc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--num-frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guide-scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="wan_out.mp4")
    ap.add_argument("--tiling", default="aggressive", help="upstream preset; ignored when --tile-px/--tile-frames are set")
    ap.add_argument("--tile-px", type=int, default=0, help="custom VAE spatial tile size in pixels (multiple of 32)")
    ap.add_argument("--tile-frames", type=int, default=0, help="custom VAE temporal tile size in frames (multiple of 8, >=16)")
    ap.add_argument("--save-latents", default=None, help="save the denoised latents (.npy) before decoding")
    ap.add_argument("--decode-latents", default=None, help="skip generation: decode this .npy with the VAE only")
    ap.add_argument("--t5-bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--cache", default=None, help="directory for cached prompt embeddings (default: <model-dir>/context_cache)")
    ap.add_argument("--probe", action="store_true", help="random conditioning, no text encoder: speed/memory measurement only")
    ap.add_argument("--report", default=None, help="write timings and peak memory as JSON here")
    args = ap.parse_args()

    import mlx_video.models.wan_2.generate as G
    from mlx_video.models.ltx_2.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig

    model_dir = Path(args.model_dir)

    def custom_tiling():
        if not (args.tile_px or args.tile_frames):
            return None
        return TilingConfig(
            spatial_config=SpatialTilingConfig(args.tile_px, 64 if args.tile_px > 128 else 32) if args.tile_px else None,
            temporal_config=TemporalTilingConfig(args.tile_frames, 8) if args.tile_frames else None)

    # the VAE proxy lets us dump latents and substitute a custom tiling without touching upstream code
    orig_load_vae = G.load_vae_decoder

    class VaeProxy:
        def __init__(self, vae):
            self.vae = vae

        def decode_tiled(self, z, tiling_config=None):
            if args.save_latents:
                np.save(args.save_latents, np.array(z.astype(mx.float32)))
            return self.vae.decode_tiled(z, custom_tiling() or tiling_config)

        def __call__(self, z):
            return self.decode_tiled(z, TilingConfig.aggressive())

        def __getattr__(self, name):
            return getattr(self.vae, name)

    G.load_vae_decoder = lambda path, config=None: VaeProxy(orig_load_vae(path, config))

    if args.decode_latents:
        from types import SimpleNamespace
        t0 = time.time()
        mx.reset_peak_memory()
        vae = G.load_vae_decoder(model_dir / "vae.safetensors", SimpleNamespace(vae_z_dim=48))
        presets = {"aggressive": TilingConfig.aggressive, "default": TilingConfig.default}
        video = vae.decode_tiled(mx.array(np.load(args.decode_latents)), presets.get(args.tiling, TilingConfig.aggressive)())
        mx.eval(video)
        frames = np.clip((np.array(video[0]) + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        G.save_video(frames, args.output, fps=24)
        out = dict(decode_s=round(time.time() - t0, 1), peak_gb=round(mx.get_peak_memory() / 1e9, 2), frames=int(frames.shape[0]))
        print("WAN_RUN " + json.dumps(out), flush=True)
        if args.report:
            Path(args.report).write_text(json.dumps(out, indent=2))
        return 0

    cache = Path(args.cache) if args.cache else model_dir / "context_cache"
    cache.mkdir(parents=True, exist_ok=True)
    timings: dict = {}

    def cache_path(prompt: str) -> Path:
        return cache / (hashlib.sha256(f"q{args.t5_bits}|{prompt}".encode()).hexdigest()[:24] + ".npy")

    class LazyT5:
        """Stands in for the encoder; the real (quantised) model is only built if an embedding is not cached."""
        def __init__(self, path, config):
            self.path, self.config, self.model = path, config, None

        def get(self):
            if self.model is None:
                t = time.time()
                self.model = load_t5_quantized(self.path, self.config, args.t5_bits)
                timings["t5_load_s"] = round(time.time() - t, 1)
                timings["t5_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
            return self.model

    orig_encode = G.encode_text

    def encode_text(encoder, tokenizer, prompt, text_len=512):
        if args.probe:
            rng = np.random.default_rng(abs(hash(prompt)) % (2 ** 31))
            return mx.array(rng.standard_normal((48, encoder.config.t5_dim)).astype(np.float32) * 0.05)
        p = cache_path(prompt)
        if p.exists():
            return mx.array(np.load(p))
        if tokenizer.tok is None:
            from transformers import AutoTokenizer
            tokenizer.tok = _orig_tok("google/umt5-xxl")
        emb = orig_encode(encoder.get(), tokenizer.tok, prompt, text_len)
        mx.eval(emb)
        np.save(p, np.array(emb.astype(mx.float32)))
        return emb

    class LazyTok:
        tok = None

    import transformers
    _orig_tok = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: LazyTok())
    G.load_t5_encoder = lambda path, config: LazyT5(path, config)
    G.encode_text = encode_text

    t0 = time.time()
    mx.reset_peak_memory()
    G.generate_video(model_dir=str(model_dir), prompt=args.prompt, image=args.image, width=args.width,
                     height=args.height, num_frames=args.num_frames, steps=args.steps,
                     guide_scale=args.guide_scale, seed=args.seed, output_path=args.output, tiling=args.tiling)
    timings.update(total_s=round(time.time() - t0, 1), peak_gb=round(mx.get_peak_memory() / 1e9, 2),
                   width=args.width, height=args.height, frames=args.num_frames, steps=args.steps, probe=args.probe)
    print("WAN_RUN " + json.dumps(timings), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(timings, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
