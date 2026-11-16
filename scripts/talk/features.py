"""Per-shot conditioning for the MLX InfiniteTalk port, computed with PyTorch in envs/lipsync:

  * wav2vec2 audio features at 25 fps: [frames, 12 layers, 768]
  * CLIP ViT-H/14 image features of the reference image: [257, 1280] (output of the first 31 of 32 blocks)
  * the reference image resized + centre-cropped to the InfiniteTalk 480p bucket, in [-1, 1]: [3, H, W]

Own implementation of the reference preprocessing (read from vendor/infinitetalk: generate_infinitetalk.py
get_embedding, src/audio_analysis/wav2vec2.py, wan/multitalk.py resize_and_centercrop + bucket choice,
wan/modules/clip.py CLIPModel.visual) on top of transformers' Wav2Vec2Model and CLIPVisionModel only; none of the
vendored code is executed. Weights: TencentGameMate/chinese-wav2vec2-base (MIT), the image encoder of
Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (Apache-2.0, safetensors).

  python scripts/talk/features.py --wav2vec DIR --clip DIR --audio line.wav --image key.png --out feats.npz
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

FPS = 25
SR = 16000
# InfiniteTalk 480p buckets (wan/utils/multitalk_utils.py ASPECT_RATIO_627): h/w ratio -> (h, w)
BUCKETS_480 = {0.26: (320, 1216), 0.38: (384, 1024), 0.50: (448, 896), 0.67: (512, 768), 0.82: (576, 704),
               1.00: (640, 640), 1.22: (704, 576), 1.50: (768, 512), 1.86: (832, 448), 2.00: (896, 448),
               2.50: (960, 384), 2.83: (1088, 384), 3.60: (1152, 320), 3.80: (1216, 320), 4.00: (1280, 320)}
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def bucket(h: int, w: int) -> tuple[int, int]:
    r = h / w
    return BUCKETS_480[min(BUCKETS_480, key=lambda k: abs(k - r))]


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def audio_features(wav2vec_dir: str, speech: np.ndarray) -> np.ndarray:
    """[frames, 12, 768], frames = int(seconds * 25) — reference get_embedding()."""
    import torch
    import torch.nn.functional as F
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    frames = int(len(speech) / SR * FPS)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).eval()
    iv = torch.from_numpy(np.squeeze(fe(speech, sampling_rate=SR).input_values)).float()[None]
    with torch.no_grad():
        feats = model.feature_extractor(iv).transpose(1, 2)                       # [1, T, 512]
        feats = F.interpolate(feats.transpose(1, 2), size=frames, align_corners=True,
                              mode="linear").transpose(1, 2)                      # to 25 fps
        hidden, _ = model.feature_projection(feats)
        enc = model.encoder(hidden, output_hidden_states=True)
    emb = torch.stack(enc.hidden_states[1:], dim=1)[0]                            # [12, frames, 768]
    return emb.permute(1, 0, 2).float().numpy()


def reference_image(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """Resize (bilinear, cover) + centre crop to the bucket, [-1, 1], [3, H, W] — reference resize_and_centercrop."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    th, tw = size or bucket(img.height, img.width)
    scale = max(th / img.height, tw / img.width)
    fh, fw = math.ceil(scale * img.height), math.ceil(scale * img.width)
    arr = np.asarray(img.resize((fw, fh), resample=Image.BILINEAR), dtype=np.float32)
    top, left = int(round((fh - th) / 2.0)), int(round((fw - tw) / 2.0))     # torchvision center_crop rounding
    arr = arr[top:top + th, left:left + tw]
    return (arr / 255.0 - 0.5).transpose(2, 0, 1) * 2.0


def clip_features(clip_dir: str, image: np.ndarray) -> np.ndarray:
    """[257, 1280] — reference CLIPModel.visual(): bicubic to 224, [-1,1] -> [0,1], CLIP normalisation,
    the vision transformer up to block 31 of 32 (no post-norm)."""
    import torch
    import torch.nn.functional as F
    from transformers import CLIPVisionModel
    model = CLIPVisionModel.from_pretrained(clip_dir, local_files_only=True).eval()
    x = torch.from_numpy(image)[None].float()
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
    x = x.mul(0.5).add(0.5)
    mean, std = torch.tensor(CLIP_MEAN)[None, :, None, None], torch.tensor(CLIP_STD)[None, :, None, None]
    with torch.no_grad():
        out = model(pixel_values=(x - mean) / std, output_hidden_states=True)
    return out.hidden_states[31][0].float().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav2vec", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--audio", required=True, help="the speech of the shot (any rate; silence where nobody talks)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", default=None, help="HxW instead of the 480p bucket")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x")) if a.size else None
    speech = load_audio(a.audio)
    image = reference_image(a.image, size)
    np.savez(a.out, audio=audio_features(a.wav2vec, speech).astype(np.float16),
             clip=clip_features(a.clip, image).astype(np.float32), image=image.astype(np.float32))
    print(json.dumps({"frames": int(len(speech) / SR * FPS), "image": list(image.shape)}), flush=True)


if __name__ == "__main__":
    main()
