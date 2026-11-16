"""InfiniteTalk (MeiGen-AI, Apache-2.0) on Apple MLX: Wan2.1-I2V-14B + audio cross-attention, single speaker.

Port of `vendor/infinitetalk/wan/modules/multitalk_model.py` (commit in vendor/infinitetalk/COMMIT) on top of the
MLX Wan implementation in vendor/mlx-video (its self-attention, RoPE, norms, head and time/text embeddings are
reused unchanged). Added here:
  * Wan2.1 image conditioning: `img_emb` (CLIP ViT-H features -> dim) and cross-attention with extra
    `k_img`/`v_img` over the 257 image tokens (WanI2VCrossAttention),
  * InfiniteTalk audio: `audio_proj` (wav2vec2 window features -> 32 tokens per latent frame) and per block
    `norm_x` + `audio_cross_attn` (the video tokens of each latent frame attend to that frame's audio tokens).
Only the single-speaker path (human_num == 1) is ported; multi-speaker L-RoPE routing is not.

Block order (reference WanAttentionBlock.forward): self-attn (modulated) -> I2V cross-attn -> audio cross-attn ->
FFN (modulated). Weight names follow the reference checkpoints after mlx-video's key sanitizer.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.wan_2.attention import WanLayerNorm, WanRMSNorm, WanSelfAttention, _linear_dtype
from mlx_video.models.wan_2.config import WanModelConfig
from mlx_video.models.wan_2.transformer import WanFFN
from mlx_video.models.wan_2.wan_2 import WanModel

CLIP_TOKENS = 257


def i2v_14b_config() -> WanModelConfig:
    """Wan2.1-I2V-14B-480P (the base of InfiniteTalk); reference config wan/configs/wan_multitalk_14B.py."""
    return WanModelConfig(model_type="i2v", dim=5120, ffn_dim=13824, freq_dim=256, num_heads=40, num_layers=40,
                          in_dim=36, out_dim=16, text_len=512, qk_norm=True, cross_attn_norm=True, eps=1e-6,
                          dual_model=False, boundary=0.0, vae_z_dim=16, vae_stride=(4, 8, 8),
                          patch_size=(1, 2, 2))


def _attend(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


class I2VCrossAttention(nn.Module):
    """Text cross-attention plus a second attention over the CLIP image tokens; outputs are summed before `o`."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.k_img, self.v_img = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.norm_q, self.norm_k, self.norm_k_img = WanRMSNorm(dim, eps), WanRMSNorm(dim, eps), WanRMSNorm(dim, eps)

    def _heads(self, t):
        b = t.shape[0]
        return t.reshape(b, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def prepare_kv(self, context: mx.array) -> tuple:
        """context: [B, 257 + text_len, dim] (image tokens first). Returns (k, v, k_img, v_img) in head layout."""
        ctx = context.astype(_linear_dtype(self.k))
        img, txt = ctx[:, :CLIP_TOKENS], ctx[:, CLIP_TOKENS:]
        return (self._heads(self.norm_k(self.k(txt))), self._heads(self.v(txt)),
                self._heads(self.norm_k_img(self.k_img(img))), self._heads(self.v_img(img)))

    def __call__(self, x: mx.array, context: mx.array | None = None, kv_cache: tuple | None = None) -> mx.array:
        b = x.shape[0]
        k, v, k_img, v_img = kv_cache if kv_cache is not None else self.prepare_kv(context)
        q = self._heads(self.norm_q(self.q(x.astype(_linear_dtype(self.q)))))
        out = _attend(q, k, v, self.scale) + _attend(q, k_img, v_img, self.scale)
        return self.o(out.transpose(0, 2, 1, 3).reshape(b, -1, self.num_heads * self.head_dim))


class AudioCrossAttention(nn.Module):
    """SingleStreamAttention, single speaker: per latent frame, its video tokens attend to its audio tokens.
    No q/k norm (qk_norm=False in the reference block), qkv bias on."""

    def __init__(self, dim: int, audio_dim: int, num_heads: int):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(audio_dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def prepare_kv(self, audio_tokens: mx.array) -> tuple:
        """audio_tokens: [B, N_t, M, audio_dim] -> k, v [B*N_t, H, M, D]."""
        b, n_t, m, _ = audio_tokens.shape
        kv = self.kv_linear(audio_tokens.astype(_linear_dtype(self.kv_linear)).reshape(b * n_t, m, -1))
        kv = kv.reshape(b * n_t, m, 2, self.num_heads, self.head_dim)
        return kv[:, :, 0].transpose(0, 2, 1, 3), kv[:, :, 1].transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, n_t: int, kv_cache: tuple) -> mx.array:
        b, l, c = x.shape
        s = l // n_t
        q = self.q_linear(x.astype(_linear_dtype(self.q_linear))).reshape(b * n_t, s, self.num_heads, self.head_dim)
        out = _attend(q.transpose(0, 2, 1, 3), *kv_cache, self.scale)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(b, l, c))


class TalkBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float, audio_dim: int):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, (-1, -1), True, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)  # cross_attn_norm=True for I2V-14B
        self.cross_attn = I2VCrossAttention(dim, num_heads, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = WanFFN(dim, ffn_dim)
        self.modulation = mx.zeros((1, 6, dim), dtype=mx.float32)
        self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True)  # norm_input_visual=True
        self.audio_cross_attn = AudioCrossAttention(dim, audio_dim, num_heads)

    def __call__(self, x, e, grid_sizes, freqs, rope_cos_sin, cross_kv, audio_kv, n_t):
        mod = self.modulation + e  # [B, 1, 6, dim] float32 (as in mlx-video's block)
        e0, e1, e2, e3, e4, e5 = (mod[:, :, i, :] for i in range(6))
        y = self.self_attn(self.norm1(x) * (1 + e1) + e0, [x.shape[1]] * x.shape[0], grid_sizes, freqs,
                           rope_cos_sin=rope_cos_sin)
        x = x + y * e2
        x = x + self.cross_attn(self.norm3(x), kv_cache=cross_kv)
        x = x + self.audio_cross_attn(self.norm_x(x), n_t, audio_kv)
        x = x + self.ffn(self.norm2(x) * (1 + e4) + e3) * e5
        return x


class MLPProj(nn.Module):
    """img_emb.proj: LayerNorm(1280) -> Linear -> GELU(erf) -> Linear(1280 -> dim) -> LayerNorm(dim) (affine)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.norm_out = nn.LayerNorm(out_dim)

    def __call__(self, x):
        return self.norm_out(self.fc2(nn.gelu(self.fc1(self.norm_in(x)))))


class AudioProj(nn.Module):
    """AudioProjModel: first latent frame from a 5-frame window, every later latent frame from 8 window entries
    (4 video frames); 12 wav2vec2 layers x 768 -> 512 -> 512 -> 32 tokens x 768, LayerNorm (norm_output_audio)."""

    def __init__(self, window=5, window_vf=8, blocks=12, channels=768, inter=512, out_dim=768, tokens=32):
        super().__init__()
        self.tokens, self.out_dim = tokens, out_dim
        self.proj1 = nn.Linear(window * blocks * channels, inter)
        self.proj1_vf = nn.Linear(window_vf * blocks * channels, inter)
        self.proj2 = nn.Linear(inter, inter)
        self.proj3 = nn.Linear(inter, tokens * out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def __call__(self, first: mx.array, latter: mx.array) -> mx.array:
        """first: [B, 1, 5, 12, 768], latter: [B, N_t-1, 8, 12, 768] -> [B, N_t, 32, 768]"""
        b = first.shape[0]
        dt = _linear_dtype(self.proj1)
        f = nn.relu(self.proj1(first.reshape(b * first.shape[1], -1).astype(dt))).reshape(b, first.shape[1], -1)
        l = nn.relu(self.proj1_vf(latter.reshape(b * latter.shape[1], -1).astype(dt))).reshape(b, latter.shape[1], -1)
        c = mx.concatenate([f, l], axis=1)
        n_t = c.shape[1]
        c = nn.relu(self.proj2(c.reshape(b * n_t, -1)))
        tok = self.proj3(c).reshape(b * n_t, self.tokens, self.out_dim)
        return self.norm(tok.astype(mx.float32)).reshape(b, n_t, self.tokens, self.out_dim)


def audio_windows(audio_emb: mx.array, vae_scale: int = 4, window: int = 5):
    """Reference WanModel.forward audio slicing. audio_emb: [B, F, 5, 12, 768] (one 5-frame window per video frame,
    F = 4 * (N_t - 1) + 1). Returns (first [B,1,5,12,768], latter [B,N_t-1,8,12,768])."""
    b, f = audio_emb.shape[:2]
    first = audio_emb[:, :1]
    lat = audio_emb[:, 1:].reshape(b, (f - 1) // vae_scale, vae_scale, *audio_emb.shape[2:])
    mid = window // 2
    a = lat[:, :, :1, : mid + 1].reshape(b, lat.shape[1], -1, *audio_emb.shape[3:])
    m = lat[:, :, 1:-1, mid:mid + 1].reshape(b, lat.shape[1], -1, *audio_emb.shape[3:])
    z = lat[:, :, -1:, mid:].reshape(b, lat.shape[1], -1, *audio_emb.shape[3:])
    return first, mx.concatenate([a, m, z], axis=2)


class InfiniteTalkModel(WanModel):
    def __init__(self, config: WanModelConfig | None = None, audio_dim: int = 768):
        config = config or i2v_14b_config()
        super().__init__(config)
        self.blocks = [TalkBlock(config.dim, config.ffn_dim, config.num_heads, config.eps, audio_dim)
                       for _ in range(config.num_layers)]
        self.img_emb = MLPProj(1280, config.dim)
        self.audio_proj = AudioProj(out_dim=audio_dim)

    # -- conditioning computed once per window ------------------------------------------------------------
    def prepare_context(self, text_emb: mx.array, clip_fea: mx.array) -> list:
        """text_emb: [L, 4096] T5 output; clip_fea: [257, 1280]. Returns per-block cross-attention K/V."""
        ctx_txt = self.embed_text([text_emb])                      # [1, 512, dim]
        ctx_img = self.img_emb(clip_fea[None].astype(mx.float32)).astype(ctx_txt.dtype)
        context = mx.concatenate([ctx_img, ctx_txt], axis=1)
        return [blk.cross_attn.prepare_kv(context) for blk in self.blocks]

    def prepare_audio(self, audio_emb: mx.array) -> list:
        """audio_emb: [1, F, 5, 12, 768]. Returns per-block audio K/V."""
        tokens = self.audio_proj(*audio_windows(audio_emb))
        return [blk.audio_cross_attn.prepare_kv(tokens) for blk in self.blocks]

    def __call__(self, x: mx.array, t: mx.array, y: mx.array, cross_kv: list, audio_kv: list,
                 rope_cos_sin: tuple | None = None) -> mx.array:
        """x: noisy latent [16, T, H, W]; y: [20, T, H, W] (mask + image latent); t: [1]. Returns velocity."""
        xin = mx.concatenate([x, y], axis=0)
        p, gs = self._patchify(xin)                                # [1, L, dim]
        grid_sizes = [gs]
        sinusoid = t[..., None].astype(mx.float32) * self._inv_freq
        sin_emb = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)
        e = self.time_embedding_1(self.time_embedding_act(self.time_embedding_0(sin_emb)))
        e0 = self.time_projection(self.time_projection_act(e)).reshape(1, 1, 6, self.dim)
        if rope_cos_sin is None:
            rope_cos_sin = self.prepare_rope(grid_sizes)
        h = p
        for blk, ckv, akv in zip(self.blocks, cross_kv, audio_kv):
            h = blk(h, e0, grid_sizes, self.freqs, rope_cos_sin, ckv, akv, gs[0])
        h = self.head(h, e)
        return self.unpatchify(h, grid_sizes)[0].astype(mx.float32)
