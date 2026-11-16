"""Run LatentSync (MLX port) on one clip. Executed with envs/lipsync/bin/python by aivs/lipsync.py.

Differences from upstream `scripts/inference.py` of latentsync-mlx, all done here without editing vendor code:
- Face detection/landmarks: Google MediaPipe Face Landmarker (Apache-2.0) instead of InsightFace (its pretrained
  models are non-commercial). LatentSync only needs three points per frame (both eyebrow centres and the middle
  of the nose) to align the face; they are taken from MediaPipe's 478-point mesh. MediaPipe's detector is a
  short-range model, so the first frame is also searched in overlapping tiles; later frames track the face in a
  crop around its last position. Frames where no face is found reuse the last detection (slow camera moves), a
  clip without any face fails.
- With several people in the shot, the speaker's face is identified against the speaker's photo/portrait with
  OpenCV SFace (Apache-2.0); see MediaPipeFaceDetector.
- `decord` (no macOS arm64 wheels) is never imported: audio is read with ffmpeg instead.
- Absolute checkpoint paths; temporary files go to --temp instead of the current directory.

  python scripts/lipsync_run.py --vendor vendor/latentsync --ckpt models/local/latentsync \
      --video clip.mp4 --audio line16k.wav --output out.mp4 [--resolution 512 --steps 20]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

# MediaPipe face mesh indices: eyebrows (image-left / image-right decided per frame by x) and the nose ridge
BROW_A = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_B = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
NOSE = [195, 5]
# InsightFace 2d106 indices LatentSync reads (left brow, right brow, nose)
L106, R106, N106 = [43, 48, 49, 51, 50], list(range(101, 106)), [74, 77, 83, 86]
SFACE = "face_recognition_sface_2021dec.onnx"  # OpenCV Zoo face recognition (Apache-2.0), picks the speaker's face
# Cosine similarity a face needs to the speaker's portrait. Measured on generated keyframes vs FLUX portraits
# (kurzer-streit-im-cafe): the right person 0.38-0.70, background extras -0.03-0.15.
MIN_SPEAKER_SIM = 0.25


class FaceIdentity:
    """OpenCV SFace (Apache-2.0, opencv_zoo) face embeddings, aligned with five points from MediaPipe's mesh."""

    def __init__(self, model_path: str):
        import cv2
        self.rec = cv2.FaceRecognizerSF.create(model_path, "")

    def embed(self, rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
        import cv2
        eyes = sorted([pts[468], pts[473]], key=lambda p: p[0])     # iris centres, image-left first
        mouth = sorted([pts[61], pts[291]], key=lambda p: p[0])     # mouth corners
        (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
        row = np.array([[x1, y1, x2 - x1, y2 - y1, *eyes[0], *eyes[1], *pts[1], *mouth[0], *mouth[1], 1.0]],
                       dtype=np.float32)
        crop = self.rec.alignCrop(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR), row)
        return self.rec.feature(crop).copy()

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        import cv2
        return float(self.rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))  # same person typically > 0.36


class MediaPipeFaceDetector:
    """Drop-in for latentsync.utils.face_detector.FaceDetector: __call__(frame) -> (bbox, landmarks_106).

    Which face is the speaker (decided on the first frame, then the same face is followed by position, so the choice
    cannot flip to the other person mid-shot):
    - with the speaker's photo (AIVS_REF_FACE): the face most similar to it (SFace) among the faces that resemble
      the speaker more than any other character in the video (AIVS_OTHER_FACES) and at least MIN_SPEAKER_SIM. If no
      face qualifies, e.g. only the listener or an extra is visible, there is no speaker face: the shot is not
      lip-synced rather than animating the wrong mouth;
    - without a photo: the largest face."""

    def __init__(self, device: str = "cpu", model_path: str | None = None):
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path or os.environ["AIVS_FACE_LANDMARKER"]),
            running_mode=vision.RunningMode.IMAGE, num_faces=4,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4)
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.last = None
        self.center = self.width = None
        self.found = self.reused = self.max_faces = 0
        self.choice = None
        self.ident = self.speaker = None
        self.others: list = []
        ref = os.environ.get("AIVS_REF_FACE")
        sface = os.environ.get("AIVS_SFACE")
        if ref and sface and Path(sface).is_file():
            self.ident = FaceIdentity(sface)
            self.speaker = self._ref_embedding(ref)
            self.others = [e for p in filter(None, os.environ.get("AIVS_OTHER_FACES", "").split(os.pathsep))
                           if (e := self._ref_embedding(p)) is not None]
            if self.speaker is None:
                self.ident = None

    def _ref_embedding(self, path: str):
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self._faces(rgb, min_w=20, min_h=25)
        return self.ident.embed(rgb, max(faces, key=lambda f: f[0])[1]) if faces else None

    def _detect(self, img: np.ndarray, x0: int = 0, y0: int = 0) -> list:
        """Landmarker on one image or crop; faces as (area, 478 points, box) in the coordinates of the full frame."""
        import mediapipe as mp
        h, w = img.shape[:2]
        res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
        out = []
        for face in res.face_landmarks or []:
            pts = np.array([[p.x * w + x0, p.y * h + y0] for p in face], dtype=np.float64)
            (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
            out.append(((x2 - x1) * (y2 - y1), pts, [int(x1), int(y1), int(x2), int(y2)]))
        return out

    def _faces(self, frame: np.ndarray, min_w: int = 40, min_h: int = 50) -> list:
        """All faces in the frame. MediaPipe's detector is a short-range (selfie) model that misses faces filling
        only a small part of the picture (people at a table, medium shots), so the frame is also searched in
        overlapping square tiles, in which those faces are large enough. Duplicates are merged."""
        h, w = frame.shape[:2]
        found = self._detect(frame)
        t = max(int(min(h, w) * 0.6), 64)
        step = t // 2
        xs = sorted({*range(0, max(w - t, 0) + 1, step), max(w - t, 0)})
        ys = sorted({*range(0, max(h - t, 0) + 1, step), max(h - t, 0)})
        for y in ys:
            for x in xs:
                found += self._detect(frame[y:y + t, x:x + t], x, y)
        return self._merge(found, min_w, min_h)

    def _around(self, frame: np.ndarray, box, min_w: int = 40, min_h: int = 50) -> list:
        """Faces in a crop around the previous position (tracking): fast, and the face is large in the crop."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        side = int(max(x2 - x1, y2 - y1) * 3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        X1, Y1 = max(cx - side // 2, 0), max(cy - side // 2, 0)
        X2, Y2 = min(X1 + side, w), min(Y1 + side, h)
        return self._merge(self._detect(frame[Y1:Y2, X1:X2], X1, Y1), min_w, min_h)

    @staticmethod
    def _merge(found: list, min_w: int, min_h: int) -> list:
        out = []
        for f in sorted(found, key=lambda f: -f[0]):
            x1, y1, x2, y2 = f[2]
            if x2 - x1 < min_w or y2 - y1 < min_h:
                continue
            c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            if all(np.linalg.norm(c - np.array([(g[2][0] + g[2][2]) / 2, (g[2][1] + g[2][3]) / 2]))
                   > 0.5 * (g[2][2] - g[2][0]) for g in out):
                out.append(f)
        return out

    def _pick(self, frame: np.ndarray, faces: list):
        centre = lambda f: np.array([(f[2][0] + f[2][2]) / 2, (f[2][1] + f[2][3]) / 2])
        if self.center is not None:  # follow the face chosen before
            best = min(faces, key=lambda f: np.linalg.norm(centre(f) - self.center))
            return best if np.linalg.norm(centre(best) - self.center) <= 0.6 * self.width else None
        if self.ident is not None:
            rows, best, best_sim = [], None, -1.0
            for f in faces:
                emb = self.ident.embed(frame, f[1])
                s = self.ident.similarity(self.speaker, emb)
                o = max((self.ident.similarity(e, emb) for e in self.others), default=-1.0)
                rows.append(f"x={int(centre(f)[0])}: speaker {s:.2f}, others {o:.2f}")
                if s > o and s >= MIN_SPEAKER_SIM and s > best_sim:
                    best, best_sim = f, s
            self.choice = ("speaker's photo match" if best is not None else "speaker not found") + ": " + "; ".join(rows)
            return best
        if len(faces) == 1:
            self.choice = "only face"
            return faces[0]
        self.choice = "largest of several faces (no reference photo)"
        return max(faces, key=lambda f: f[0])

    def __call__(self, frame, threshold: float = 0.5):
        frame = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
        faces = self._around(frame, self.last[0]) if self.last is not None else []
        if not faces:  # first frame, or the tracked face was lost: search the whole frame
            faces = self._faces(frame)
            self.max_faces = max(self.max_faces, len(faces))
        best = self._pick(frame, faces) if faces else None
        if best is None:
            if self.last is not None:
                self.reused += 1
                return self.last
            return None, None
        x1, y1, x2, y2 = best[2]
        self.center, self.width = np.array([(x1 + x2) / 2, (y1 + y2) / 2]), float(x2 - x1)
        pts = best[1]
        a, b = pts[BROW_A].mean(0), pts[BROW_B].mean(0)
        left, right = (a, b) if a[0] <= b[0] else (b, a)
        lmk = np.zeros((106, 2), dtype=np.float64)
        lmk[L106], lmk[R106], lmk[N106] = left, right, pts[NOSE].mean(0)
        self.last = (np.array(best[2]), lmk)
        self.found += 1
        return self.last


def read_audio_ffmpeg(audio_path: str, audio_sample_rate: int = 16000):
    import torch
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(audio_path), "-f", "f32le", "-ac", "1",
                          "-ar", str(audio_sample_rate), "-"], capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())


CACHE_LIMIT_BYTES = 512 * 1024 ** 2   # MLX buffer cache cap, see main()
FUSED_HEAD_DIMS = (64, 80, 128)  # head sizes MLX's fused attention kernel handles without the full score matrix


def _fused_attention(self, x, encoder_hidden_states=None):
    """Replacement for latentsync_mlx.attention.Attention.__call__ (same weights, same result within fp16).

    Upstream materialises softmax(q k^T) explicitly: at 512 px the first UNet level has 4096 tokens per frame,
    32 sequences (16 frames x CFG 2) and 8 heads -> one 17 GB allocation, above Metal's ~9.5 GB buffer limit on a
    16 GB Mac. mx.fast.scaled_dot_product_attention avoids that, but only for some head sizes; this UNet's 40 is
    zero-padded to 64 (zero columns change neither q.k nor the softmax; the extra output columns are dropped).
    Measured for 32x8x4096x40: 0.54 GB instead of 8.8 GB, 0.16 s instead of 1.15 s."""
    import mlx.core as mx
    B, S, _ = x.shape
    ctx = encoder_hidden_states if encoder_hidden_states is not None else x
    Sk, d = ctx.shape[1], self.dim_head
    q = self.to_q(x).reshape(B, S, self.heads, d).transpose(0, 2, 1, 3)
    k = self.to_k(ctx).reshape(B, Sk, self.heads, d).transpose(0, 2, 1, 3)
    v = self.to_v(ctx).reshape(B, Sk, self.heads, d).transpose(0, 2, 1, 3)
    to = next((n for n in FUSED_HEAD_DIMS if n >= d), d)
    if to != d:
        pad = [(0, 0)] * 3 + [(0, to - d)]
        q, k, v = mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)[..., :d]
    return self.to_out(out.transpose(0, 2, 1, 3).reshape(B, S, -1))


def _install_shims() -> None:
    """Stand-ins for modules upstream imports at module level but that this pipeline never uses.

    transformers probes for decord with importlib.util.find_spec() when it is first imported; that raises for a
    module without __spec__, and with one it would believe decord is installed. So transformers is imported first
    (it then records "no decord") and the stand-ins get a spec anyway, for any later probe."""
    import importlib.machinery
    try:
        import transformers  # noqa: F401
    except Exception:  # not installed, or (in tests) another stand-in module without a spec upsets its probes
        pass
    mod = lambda name: (m := types.ModuleType(name), setattr(m, "__spec__", importlib.machinery.ModuleSpec(name, None)), m)[2]
    decord = mod("decord")

    class _Unavailable:
        def __init__(self, *a, **k):
            raise RuntimeError("decord is not used on macOS (aivs lipsync_run)")
    decord.AudioReader = decord.VideoReader = _Unavailable
    sys.modules.setdefault("decord", decord)
    ins = mod("insightface")
    ins_app = mod("insightface.app")
    ins_app.FaceAnalysis = _Unavailable
    ins.app = ins_app
    sys.modules.setdefault("insightface", ins)
    sys.modules.setdefault("insightface.app", ins_app)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", required=True, help="folder with LatentSync/ and latentsync-mlx/")
    ap.add_argument("--ckpt", required=True, help="folder with the converted weights and face_landmarker.task")
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resolution", type=int, default=512, choices=[256, 512])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--temp", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--ref-face", default=None, help="photo of the speaker: the face to animate is the one that "
                    "looks like it (needs face_recognition_sface_2021dec.onnx in --ckpt)")
    ap.add_argument("--other-faces", nargs="*", default=[], help="photos of the other characters: a face that "
                    "looks more like one of them than like the speaker is never animated")
    args = ap.parse_args()
    t0 = time.time()
    vendor, ckpt = Path(args.vendor).resolve(), Path(args.ckpt).resolve()
    video, audio, output = (str(Path(p).resolve()) for p in (args.video, args.audio, args.output))
    sys.path[:0] = [str(vendor / "latentsync-mlx"), str(vendor / "LatentSync")]
    os.environ["AIVS_FACE_LANDMARKER"] = str(ckpt / "face_landmarker.task")
    os.environ["AIVS_SFACE"] = str(ckpt / SFACE)
    if args.ref_face:
        os.environ["AIVS_REF_FACE"] = str(Path(args.ref_face).resolve())
        os.environ["AIVS_OTHER_FACES"] = os.pathsep.join(str(Path(p).resolve()) for p in args.other_faces)
    _install_shims()

    import mlx.core as mx
    # MLX keeps freed buffers in a cache for reuse; at 512 px it grew to 8.2 GB on top of 4.5 GB in use, and the
    # 16 GB Mac swapped (UNet step 7-24 s instead of 3.0 s, measured). A small cache costs nothing measurable here.
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    import torch
    import latentsync.utils.image_processor as ip
    import latentsync.utils.util as lu
    lu.read_audio = read_audio_ffmpeg
    ip.FaceDetector = MediaPipeFaceDetector
    import latentsync_mlx.attention as la
    la.Attention.__call__ = _fused_attention
    from latentsync.whisper.audio2feature import Audio2Feature
    from latentsync_mlx.pipeline import LipsyncPipelineMLX
    from latentsync_mlx.sampler import DDIMSampler
    from latentsync_mlx.unet import UNet3DConditionModel
    from latentsync_mlx.vae import Autoencoder

    dtype = mx.float16
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    def load(model, path):
        """fp32 file -> fp16 model one tensor at a time. Casting all at once held the 5 GB fp32 UNet and its 2.5 GB
        fp16 copy together (MLX peak 10.3 GB measured), pushing the rest of a 16 GB Mac into swap."""
        w, params = mx.load(str(path)), []
        for k in list(w):
            v = w.pop(k).astype(dtype)
            mx.eval(v)
            params.append((k, v))
        model.load_weights(params)
        mx.eval(model.parameters())
        mx.clear_cache()
    unet = UNet3DConditionModel()
    load(unet, ckpt / "latentsync_unet_mlx.safetensors")
    vae = Autoencoder()
    load(vae, ckpt / "vae_mlx.safetensors")
    audio_encoder = Audio2Feature(model_path=str(ckpt / "whisper" / "tiny.pt"), device=torch.device(device),
                                  num_frames=16, audio_feat_length=[2, 2])
    mask = ip.load_fixed_mask(args.resolution, mask_image_path=str(vendor / "LatentSync" / "latentsync" / "utils" / "mask.png"))
    processor = ip.ImageProcessor(resolution=args.resolution, device=device, mask_image=mask)
    if getattr(processor, "face_detector", None) is None:  # upstream skips detection on "cpu"
        processor.face_detector = MediaPipeFaceDetector()
    detector = processor.face_detector
    pipe = LipsyncPipelineMLX(unet=unet, vae=vae, sampler=DDIMSampler(), audio_encoder=audio_encoder,
                              image_processor=processor, resolution=args.resolution, dtype=dtype)
    temp = Path(args.temp or (Path(output).parent / "lipsync-tmp")).resolve()
    temp.mkdir(parents=True, exist_ok=True)
    os.chdir(temp)  # upstream writes ./temp/video.mp4 when resampling to 25 fps
    pipe(video_path=video, audio_path=audio, video_out_path=output, num_inference_steps=args.steps,
         guidance_scale=args.guidance, seed=args.seed, temp_dir=str(temp / "out"))
    if not Path(output).exists():
        raise SystemExit("LatentSync produced no output")
    rep = {"seconds": round(time.time() - t0, 1), "resolution": args.resolution, "steps": args.steps,
           "faces_found": detector.found, "frames_reusing_last_face": detector.reused,
           "max_faces_in_frame": detector.max_faces, "speaker_face": detector.choice,
           "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2) if hasattr(mx, "get_peak_memory") else None}
    print(json.dumps(rep), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
