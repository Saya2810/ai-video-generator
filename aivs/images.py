"""Keyframe generation with FLUX.2-klein-4B (Apache-2.0) through the mflux CLI. One process per image.

Reference images (see aivs/references.py) are passed to FLUX.2's multi-image editing mode; a location photo that is
the only reference of a shot is used as the keyframe directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import config, fetch, references
from .util import log, run

NO_TEXT = "No text, no captions, no watermark, no logo."
MAX_IMAGES = 4  # input images per FLUX.2 edit call (user references + optional look reference)


def _bin(name: str) -> str:
    return str(Path(sys.executable).parent / name)


def build_prompt(visual: str, style: str) -> str:
    return f"{visual.rstrip('. ')}. {style.rstrip('. ')}. {NO_TEXT}"


def generate(prompt: str, out: Path, size: tuple[int, int], seed: int, steps: int, log_path: Path,
             reference: Path | list[Path] | None = None) -> None:
    refs = [reference] if isinstance(reference, (str, Path)) else list(reference or [])
    model = fetch.resolve_model(config.MODELS["image"])
    cmd = [_bin("mflux-generate-flux2-edit" if refs else "mflux-generate-flux2"),
           "--model", model, "--base-model", "flux2-klein-4b",
           "--prompt", prompt, "--width", size[0], "--height", size[1], "--steps", steps, "--seed", seed,
           "--mlx-cache-limit-gb", "4", "--output", out]
    if refs:
        cmd += ["--image-paths", *refs]
    run(cmd, log_path=log_path, env=config.worker_env())
    if not out.exists():
        raise RuntimeError(f"image generator wrote no file; see {log_path}")


def render_keyframes(shots: list[dict], style: str, outdir: Path, size: tuple[int, int], seed: int, steps: int,
                     consistency: bool, recurring: str = "") -> None:
    """shots: [{id, visual, image(path str), _refs(list, optional)}].
    With consistency=True, shot 1 is the look reference for the rest."""
    outdir.mkdir(parents=True, exist_ok=True)
    look = None
    for i, shot in enumerate(shots):
        out = Path(shot["image"])
        srefs = shot.get("_refs") or []
        if not out.exists():
            log(f"keyframe {shot['id']} ({i + 1}/{len(shots)})"
                + (f" with {', '.join(r['name'] for r in srefs)}" if srefs else ""))
            tmp = out.with_name(out.stem + ".tmp.png")
            if references.use_as_is(srefs):
                references.fit_image(Path(srefs[0]["path"]), tmp, size)
            else:
                visual = shot["visual"]
                if consistency and recurring and i == 0:
                    visual = f"{visual.rstrip('. ')}. Main subject: {recurring}"
                scene = build_prompt(visual, style)
                inputs = [Path(r["path"]) for r in srefs]
                use_look = look is not None and len(inputs) < MAX_IMAGES
                if srefs:
                    prompt = references.flux_prompt(scene, srefs, look_reference=use_look)
                elif use_look:
                    prompt = ("Keep recurring subjects looking exactly as in the reference image; same visual style, "
                              "color palette and lighting as the reference image. New scene: " + scene)
                else:
                    prompt = scene
                if use_look:
                    inputs.append(look)
                generate(prompt, tmp, size, seed + i, steps, outdir / "images.log", inputs or None)
            tmp.replace(out)
        if consistency and look is None:
            look = out
