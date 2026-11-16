"""Audio-driven talking video with the MLX InfiniteTalk port (scripts/talk/model.py), one speaker, on a 16 GB Mac.

  envs/wan/bin/python scripts/talk/run.py --model DIR --t5 FILE --vae FILE --features feats.npz --prompt "..." \
      --output out.mp4 [--steps 1000,750,500,250 --shift 5 --audio-guide 2 --seed 7]

Generation follows the reference streaming loop (vendor/infinitetalk/wan/multitalk.py generate_infinitetalk):
windows of up to 81 frames at 25 fps; every window is conditioned on the reference image (CLIP tokens + first-frame
latent in `y`); every window after the first keeps the last `motion` decoded frames of the previous window as its
first latents (re-imposed after each step), so motion continues. With the lightx2v distillation: text guidance 1,
audio guidance 2 (reference README) -> 2 passes per step (with and without audio).
Differences from the reference (memory/speed on 16 GB, not output): transformer blocks are streamed from disk one at
a time (8-bit weights); windows are only as long as the remaining audio needs (4n+1 frames) instead of always 81.

Writes the video (25 fps, the bucket size) and a JSON report.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

FPS, WINDOW, VAE_T, VAE_S = 25, 81, 4, 8


def frames_4n1(n: int) -> int:
    return 4 * math.ceil((max(n, 5) - 1) / 4) + 1


def window_plan(total: int, window: int = WINDOW, motion: int = 9) -> list[tuple[int, int, int]]:
    """[(audio_start, frames, keep_from)]: each window starts `motion` frames before the end of the previous one;
    `keep_from` frames at its start repeat the previous window and are dropped from the output."""
    plan, start, done = [], 0, 0
    while done < total:
        keep = 0 if not plan else motion
        n = min(window, frames_4n1(total - done + keep))
        plan.append((start, n, keep))
        done += n - keep
        start += n - motion
    return plan


def timesteps(step_list: list[float], shift: float) -> list[float]:
    """Shifted timesteps + final 0; Euler between them. lightx2v's WanStepDistillScheduler (4-step distillation):
    step list 1000,750,500,250 with shift 5 -> 1000, 937.5, 833.3, 625, 0 (its step_post is exactly this Euler
    update). The undistilled InfiniteTalk reference instead uses linspace(1000, 1, steps) with shift 7."""
    return [shift * (x / 1000) / (1 + (shift - 1) * (x / 1000)) * 1000 for x in step_list] + [0.0]


class Streamer:
    """InfiniteTalkModel whose 40 transformer blocks live on disk; one quantised block module is reused."""

    def __init__(self, model_dir: Path):
        from model import InfiniteTalkModel, TalkBlock, i2v_14b_config
        cfg = json.loads((model_dir / "config.json").read_text())
        self.config = i2v_14b_config()
        self.model = InfiniteTalkModel(self.config)
        self.model.blocks = []
        self.model.load_weights(list(mx.load(str(model_dir / "non_block.safetensors")).items()), strict=False)
        mx.eval(self.model.parameters())
        c = self.config
        self.block = TalkBlock(c.dim, c.ffn_dim, c.num_heads, c.eps, 768)
        nn.quantize(self.block, group_size=cfg["group_size"], bits=cfg["bits"])
        self.shards = [model_dir / "blocks" / f"block_{i}.safetensors" for i in range(cfg["num_blocks"])]
        self.io_s = 0.0

    def forward(self, x, t, y, context, audio_tokens):
        """x [16,T,h,w] latent, y [20,T,h,w], context [1,769,dim] embedded (image+text), audio_tokens [1,N_t,32,768]."""
        m = self.model
        p, gs = m._patchify(mx.concatenate([x, y], axis=0))
        grid = [gs]
        sinusoid = t[..., None].astype(mx.float32) * m._inv_freq
        e = m.time_embedding_1(m.time_embedding_act(m.time_embedding_0(
            mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1))))
        e0 = m.time_projection(m.time_projection_act(e)).reshape(1, 1, 6, m.dim)
        rope = m.prepare_rope(grid)
        h = p
        for shard in self.shards:
            t0 = time.time()
            self.block.load_weights(list(mx.load(str(shard)).items()))
            self.io_s += time.time() - t0
            ckv = self.block.cross_attn.prepare_kv(context)
            akv = self.block.audio_cross_attn.prepare_kv(audio_tokens)
            h = self.block(h, e0, grid, m.freqs, rope, ckv, akv, gs[0])
            mx.eval(h)
        h = m.head(h, e)
        out = m.unpatchify(h, grid)[0].astype(mx.float32)
        mx.eval(out)
        return out


def load_vae(path: Path):
    from mlx_video.models.wan_2.convert import sanitize_wan_vae_weights
    from mlx_video.models.wan_2.vae import WanVAE
    vae = WanVAE(z_dim=16, encoder=True)
    w = sanitize_wan_vae_weights({k: v.astype(mx.float32) for k, v in mx.load(str(path)).items()})
    vae.load_weights(list(w.items()), strict=False)  # file has all 194 weights; mean/std are constants of the module
    mx.eval(vae.parameters())
    return vae


def text_embedding(t5_path: Path, prompt: str, cache_dir: Path, bits: int = 8) -> mx.array:
    """Same T5 (UMT5-XXL) and cache files as scripts/wan_run.py (Wan 2.1 and 2.2 share the text encoder)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / (hashlib.sha256(f"q{bits}|{prompt}".encode()).hexdigest()[:24] + ".npy")
    if p.exists():
        return mx.array(np.load(p))
    import wan_run
    from transformers import AutoTokenizer
    from mlx_video.models.wan_2.config import WanModelConfig
    from mlx_video.models.wan_2.utils import encode_text
    enc = wan_run.load_t5_quantized(t5_path, WanModelConfig(), bits)
    emb = encode_text(enc, AutoTokenizer.from_pretrained("google/umt5-xxl"), prompt, 512)
    mx.eval(emb)
    np.save(p, np.array(emb.astype(mx.float32)))
    del enc
    gc.collect()
    mx.clear_cache()
    return emb


def write_video(frames: np.ndarray, path: str, fps: int = FPS) -> None:
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(frames).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed writing the video")


def to_uint8(video: mx.array) -> np.ndarray:
    """[1,3,T,H,W] in [-1,1] -> [T,H,W,3] uint8"""
    v = np.array(video[0].astype(mx.float32))
    return (np.clip((v + 1) / 2, 0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 3, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="converted model dir (scripts/talk/convert.py)")
    ap.add_argument("--t5", required=True, help="MLX UMT5-XXL weights (t5_encoder.safetensors of the Wan 5B model)")
    ap.add_argument("--t5-cache", default=None)
    ap.add_argument("--vae", required=True, help="Wan 2.1 VAE safetensors (original key names)")
    ap.add_argument("--features", required=True, help="npz from scripts/talk/features.py")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--frames", type=int, default=0, help="output length in frames at 25 fps (default: audio)")
    ap.add_argument("--steps", default="1000,750,500,250", help="timestep list before shifting (lightx2v 4-step)")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--audio-guide", type=float, default=2.0)
    ap.add_argument("--motion", type=int, default=9)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-latents", default=None, help="debug: save each window's final latent here (.npz)")
    a = ap.parse_args()
    t_start = time.time()
    rep: dict = {"steps": a.steps, "shift": a.shift, "audio_guide": a.audio_guide}

    f = np.load(a.features)
    audio = mx.array(f["audio"].astype(np.float32))                 # [F, 12, 768]
    clip = mx.array(f["clip"])                                        # [257, 1280]
    image = mx.array(f["image"])                                      # [3, H, W] in [-1, 1]
    total = a.frames or int(audio.shape[0])
    H, W = image.shape[1:]
    h, w = H // VAE_S, W // VAE_S
    plan = window_plan(total, a.window, a.motion)
    rep.update(size=[int(W), int(H)], frames=total, windows=[list(p) for p in plan])
    print(f"InfiniteTalk: {W}x{H}, {total} frames at 25 fps, windows {plan}", flush=True)

    t0 = time.time()
    text = text_embedding(Path(a.t5), a.prompt, Path(a.t5_cache) if a.t5_cache else Path(a.t5).parent / "context_cache")
    rep["t5_s"] = round(time.time() - t0, 1)

    st = Streamer(Path(a.model))
    m = st.model
    ctx_txt = m.embed_text([text])
    context = mx.concatenate([m.img_emb(clip[None]).astype(ctx_txt.dtype), ctx_txt], axis=1)
    mx.eval(context)

    vae = load_vae(Path(a.vae))
    idx = mx.arange(-2, 3)                                            # 5-frame audio window around each frame
    zero_audio = None
    out_frames, prev_video, y_cache = [], None, {}
    mx.random.seed(a.seed)
    sched = timesteps([float(s) for s in a.steps.split(",")], a.shift)
    rep["timesteps"] = [round(s, 1) for s in sched]
    rep["window_s"] = []
    for wi, (start, n, keep) in enumerate(plan):
        tw = time.time()
        lt = (n - 1) // VAE_T + 1
        # conditioning: reference image as frame 0 of an otherwise empty clip, mask on the first latent frame
        # (identical for every window of the same length -> encoded once per length)
        if n not in y_cache:
            clip_px = mx.concatenate([image[:, None], mx.zeros((3, n - 1, H, W))], axis=1)[None]
            y_lat = vae.encode(clip_px)[0]                            # [16, lt, h, w]
            msk = mx.concatenate([mx.ones((4, 1, h, w)), mx.zeros((4, lt - 1, h, w))], axis=1)
            y_cache[n] = mx.concatenate([msk, y_lat], axis=0).astype(mx.bfloat16)
        y = y_cache[n]
        # motion latents: the reference image (first window) or the last `motion` frames of the previous window
        if prev_video is None:
            motion_lat, n_m = vae.encode(image[:, None][None])[0], 1
        else:
            motion_lat = vae.encode(prev_video[:, :, -a.motion:])[0]
            n_m = motion_lat.shape[1]
        mx.eval(y, motion_lat)
        # audio: a 5-frame window around every frame of this window (indices clamped to the audio)
        centres = mx.arange(start, start + n)[:, None] + idx[None]
        centres = mx.clip(centres, 0, audio.shape[0] - 1)
        aw = audio[centres][None]                                     # [1, n, 5, 12, 768]
        from model import audio_windows
        tokens = m.audio_proj(*audio_windows(aw))
        if a.audio_guide != 1.0:
            zero_audio = m.audio_proj(*audio_windows(mx.zeros_like(aw)))
        mx.eval(tokens, zero_audio) if zero_audio is not None else mx.eval(tokens)
        latent = mx.random.normal((16, lt, h, w)).astype(mx.float32)
        for i in range(len(sched) - 1):
            latent = mx.concatenate([motion_lat, latent[:, n_m:]], axis=1)
            t = mx.array([sched[i]], dtype=mx.float32)
            xin = latent.astype(mx.bfloat16)
            v = st.forward(xin, t, y, context, tokens)
            if a.audio_guide != 1.0:
                v_drop = st.forward(xin, t, y, context, zero_audio)
                v = v_drop + a.audio_guide * (v - v_drop)
            latent = latent - v * ((sched[i] - sched[i + 1]) / 1000.0)
            mx.eval(latent)
            print(f"  window {wi + 1}/{len(plan)} step {i + 1}/{len(sched) - 1} ({time.time() - tw:.0f}s)", flush=True)
        latent = mx.concatenate([motion_lat, latent[:, n_m:]], axis=1)
        if a.save_latents:
            np.savez(f"{a.save_latents}.w{wi}.npz", latent=np.array(latent))
        video = vae.decode(latent[None])                              # [1, 3, n, H, W]
        mx.eval(video)
        prev_video = video
        out_frames.append(to_uint8(video)[keep:])
        rep["window_s"].append(round(time.time() - tw, 1))
        mx.clear_cache()
    frames = np.concatenate(out_frames)[:total]
    write_video(frames, a.output)
    rep.update(total_s=round(time.time() - t_start, 1), block_io_s=round(st.io_s, 1),
               peak_gb=round(mx.get_peak_memory() / 1e9, 2), out_frames=int(frames.shape[0]))
    print("TALK_RUN " + json.dumps(rep), flush=True)
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
