# Third-party components and commercial-use status

The application code in `aivs/` is original work under the **MIT License** (`LICENSE`).
It does not vendor or redistribute any model weights or third-party source code; everything below is downloaded
at setup time from its original publisher. The one bundled asset is the caption font (OFL-1.1).

Licence tags were read from the Hugging Face / GitHub APIs on 2026-11-15. Licence texts collected in session 1
are in `evidence/licenses/`. This file is a technical summary, not legal advice.

## Models used by the default pipeline

| Role | Model (repo actually downloaded) | Weights licence | Output / commercial use |
|---|---|---|---|
| Script + storyboard | Qwen3-4B-Instruct-2507, 4-bit MLX (`mlx-community/Qwen3-4B-Instruct-2507-4bit`) | Apache-2.0 | No restriction on generated text |
| Keyframes | FLUX.2 [klein] 4B, 4-bit mflux (`Runpod/FLUX.2-klein-4B-mflux-4bit`; upstream `black-forest-labs/FLUX.2-klein-4B`) | Apache-2.0 (both repos tagged) | No restriction. Only the **4B** klein is Apache; the 9B klein and FLUX.2-dev are non-commercial — do not swap them in |
| Narration (English) | Kokoro-82M (`mlx-community/Kokoro-82M-bf16`; upstream `hexgrad/Kokoro-82M`) | Apache-2.0 | No restriction; built-in voices ship with the model |
| Narration (23 languages) | Chatterbox Multilingual V3 (`mlx-community/chatterbox-multilingual-v3`; upstream Resemble AI) | MIT | No restriction. Upstream embeds an inaudible Perth watermark in its own pipeline. Voice cloning (`--ref-audio`) requires that you own or have licensed the voice |
| Speech check + word timing | Whisper large-v3-turbo (`mlx-community/whisper-large-v3-turbo`; upstream OpenAI) | MIT (upstream; the MLX repo carries no tag) | Analysis only, nothing of it is in the output |
| Depth for parallax | Depth Anything V2 **Small** (`depth-anything/Depth-Anything-V2-Small-hf`) | Apache-2.0 | Analysis only. Base/Large/Giant variants are CC-BY-NC — do not swap them in |
| Music | built-in procedural synthesiser (`aivs/music.py`) | MIT (our code) | Fully owned output, no model involved |

## Optional (installed only by `scripts/setup_wan.sh`)

| Role | Model / code | Licence | Note |
|---|---|---|---|
| Image-to-video (`--motion wan`) | Wan 2.2 TI2V-5B, 8-bit MLX (`Anes1032/Wan2.2-TI2V-5B-mlx-q8`; upstream `Wan-AI/Wan2.2-TI2V-5B`) | Apache-2.0 (LICENSE + NOTICE in the repo) | Upstream README: "We claim no rights over your generated contents". Includes the UMT5-XXL text encoder (Apache-2.0) |
| Runtime for it | `Blaizzy/mlx-video` @ 87db56a (in `vendor/`, not redistributed) | MIT | Driven through `scripts/wan_run.py` (our code, MIT) |
| Tokenizer | `google/umt5-xxl` tokenizer files, fetched by transformers on first use | Apache-2.0 | |

## Characters and dialogue

| Role | Model / code | Licence | Note |
|---|---|---|---|
| Character voices (non-English) | Qwen3-TTS 12Hz 1.7B **VoiceDesign**, 8-bit MLX (`mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`; upstream `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`) | Apache-2.0 | Creates one ~10 s voice sample per character from a text description; Chatterbox (MIT) then speaks every line in that voice. Fetched automatically on first non-English dialogue |
| Character portraits | FLUX.2 [klein] 4B (as above) | Apache-2.0 | Invented characters get a generated portrait that is used as their reference image |

## Optional lip-sync (installed only by `scripts/setup_lipsync.sh`)

| Role | Model / code | Licence | Note |
|---|---|---|---|
| Lip-sync model | LatentSync 1.6 UNet (`ByteDance/LatentSync-1.6`, files `latentsync_unet.pt`, `whisper/tiny.pt` only) | **OpenRAIL++** (HF tag); code `bytedance/LatentSync` Apache-2.0 | OpenRAIL++ allows commercial use but carries use-based restrictions (no unlawful use, no harming/defaming people, no deceptive impersonation…) that apply to everyone you pass the model on to. The repo's auxiliary training/eval files are not downloaded |
| MLX runtime for it | `sb1992/latentsync-mlx` @ 8370f1f (in `vendor/`) | Apache-2.0 | Small community port; driven by `scripts/lipsync_run.py` (our code, MIT) |
| VAE | `stabilityai/sd-vae-ft-mse` | MIT | |
| Audio features | Whisper tiny (inside the LatentSync repo, upstream OpenAI) | MIT | |
| Face landmarks | MediaPipe Face Landmarker (`face_landmarker.task`, Google) | Apache-2.0 (model card) | **Replaces InsightFace**, which upstream LatentSync uses: InsightFace's pretrained models are "for non-commercial research purposes only" (insightface README). InsightFace is neither installed nor downloaded; `decord` is not needed either |
| Talking characters (audio-driven video) | InfiniteTalk adapter (`MeiGen-AI/InfiniteTalk`, `single/infinitetalk.safetensors`); code read from `MeiGen-AI/InfiniteTalk` @ 50aa0a9 (in `vendor/infinitetalk`, reference only — never executed) and re-implemented for MLX in `scripts/talk/` (our code, MIT) | Apache-2.0 (HF licence tag) | Generates the whole speaking shot from keyframe + line audio (`aivs/talk.py`, `scripts/setup_talk.sh`) |
| Its base video model | Wan2.1-I2V-14B-480P with the lightx2v 4-step / CFG distillation (`lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v`, `distill_int8/`) | Apache-2.0 | int8 weights re-quantised to MLX 8-bit; sampling schedule from lightx2v's WanStepDistillScheduler |
| Its audio features | `TencentGameMate/chinese-wav2vec2-base` | MIT | run with transformers' Wav2Vec2Model |
| Its image features / VAE | CLIP ViT-H image encoder from `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` (`image_encoder/`); Wan2.1 VAE re-saved as safetensors by `Comfy-Org/Wan_2.1_ComfyUI_repackaged` | Apache-2.0 | safetensors only (no pickled `.pth` files are loaded) |
| Speaker identification | OpenCV Zoo SFace (`face_recognition_sface_2021dec.onnx`, `opencv/opencv_zoo`), run with `cv2.FaceRecognizerSF` | Apache-2.0 (opencv_zoo model licence) | Only used to decide which face in a multi-person shot is the speaker, by comparing it with the character's photo/portrait; nothing is stored |

## Libraries and tools

| Component | Licence | How it is used |
|---|---|---|
| MLX, mlx-lm, mlx-audio, mflux | MIT | Python dependencies (inference runtimes) |
| misaki (Kokoro text front-end), spaCy + `en_core_web_sm` | Apache-2.0 / MIT / MIT | Python dependencies |
| PyTorch (BSD-3), transformers (Apache-2.0), OpenCV (Apache-2.0), NumPy/SciPy (BSD-3), Pillow (MIT-CMU), soundfile (BSD-3) | permissive | Python dependencies |
| FFmpeg (Homebrew build, GPL because it includes x264) | GPL-2.0+/LGPL | Invoked as an external program; not linked, not redistributed. Encoded files are not covered by the GPL |
| uv | MIT / Apache-2.0 | Environment manager |
| Source Sans 3 Semibold (`aivs/assets/fonts/`) | SIL OFL 1.1 | Bundled caption font; rendering it into video is permitted; licence text alongside |

## Evaluated and rejected for licence reasons

LTX-2.3 (LTX-2 Community License: free only below USD 10M revenue, Gemma-licensed text encoder — usable for many,
but not open source, so not a default), HunyuanVideo 1.5 / HunyuanVideo-Foley (licence excludes the EU), MiniMax
models (territory exclusions), MMAudio / Foley-Omni / Sony Woosh weights (CC-BY-NC), Stable Audio 3 (revenue cap),
Fish Speech / Higgs Audio (research / NC), Wav2Lip public weights (non-commercial), MuseTalk (MIT itself, but
its face-parsing/pose dependencies have unclear or non-commercial weights), InsightFace models (non-commercial),
Draw Things engine (GPL-3.0, cannot be embedded in an MIT app),
the `filipstrand/Z-Image-Turbo-mflux-4bit` repo (tagged with the Tongyi Qianwen licence although upstream is Apache-2.0
— avoided in favour of FLUX.2-klein-4B, whose quantised repo is cleanly tagged Apache-2.0).

## What you should still do yourself

- Generated images, voices and text can resemble existing works or people by accident; review before publishing.
- If you clone a voice, keep proof that you may use it.
- H.264/AAC are patent-encumbered formats; distribution of *content* is royalty-free for free internet video, but
  shipping an encoder inside a product is a separate question (this project ships none).
