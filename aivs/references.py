"""User-supplied reference images: people, objects and locations that must appear in the video.

A project keeps its references in `storyboard["references"]`:

    {"id": "anna", "kind": "person", "name": "Anna", "description": "woman in her thirties, red coat",
     "image": "refs/anna.png", "sha": "<sha256 of the stored PNG>", "as_is": false}

and every shot lists the references visible in it: `shot["refs"] = ["anna", "kitchen"]`.

How a shot is drawn:
- only a location (kind "background") with as_is=True  -> the uploaded photo itself becomes the keyframe
  (cover-cropped to the frame); motion is then added on top of it.
- anything else -> FLUX.2-klein reference editing with all of the shot's images (people first, then objects,
  then the location); the prompt names each image so the model knows what to take from it.

Uploaded files are re-encoded to PNG (EXIF, GPS and other metadata are dropped) and scaled to at most
MAX_SIDE pixels on the longest side, which keeps reference editing within the 16 GB memory budget.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .util import log, slugify

KINDS = ("person", "object", "background")
KIND_ORDER = {"person": 0, "object": 1, "background": 2}
MAX_SIDE = 1536
MAX_PER_SHOT = 3          # user references per keyframe (+1 optional look reference) — memory and quality limit
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
GENERIC = {"person": "the person", "object": "the object", "background": "the place"}


# ---------------------------------------------------------------- files
def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_image(src: Path):
    """Open any image Pillow can read; fall back to macOS `sips` for HEIC/HEIF photos from an iPhone."""
    from PIL import Image, ImageOps
    try:
        img = Image.open(src)
        img.load()
    except Exception:
        sips = shutil.which("sips")
        if not sips:
            raise ValueError(f"cannot read image {src.name} (unsupported format)")
        tmp = Path(tempfile.mkdtemp(prefix="aivs-ref-")) / "converted.png"
        proc = subprocess.run([sips, "-s", "format", "png", str(src), "--out", str(tmp)], capture_output=True)
        if proc.returncode != 0 or not tmp.exists():
            raise ValueError(f"cannot read image {src.name} (unsupported format)")
        img = Image.open(tmp)
        img.load()
    img = ImageOps.exif_transpose(img)  # respect the camera's rotation flag before the metadata is dropped
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
        return flat
    return img.convert("RGB")


def normalize_image(src: Path, dst: Path, max_side: int = MAX_SIDE) -> str:
    """Re-encode `src` as a metadata-free RGB PNG no larger than max_side. Returns the sha256 of the result."""
    from PIL import Image
    img = _open_image(Path(src))
    w, h = img.size
    if min(w, h) < 64:
        raise ValueError(f"image {Path(src).name} is too small ({w}x{h})")
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.png")
    img.save(tmp, format="PNG", optimize=True)
    tmp.replace(dst)
    return _sha(dst)


def fit_image(src: Path, dst: Path, size: tuple[int, int]) -> None:
    """Cover-crop `src` to exactly `size` (centred) — used when a location photo is the keyframe itself."""
    from PIL import Image, ImageOps
    img = Image.open(src).convert("RGB")
    out = ImageOps.fit(img, (int(size[0]), int(size[1])), method=Image.LANCZOS, centering=(0.5, 0.5))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".fit.tmp.png")
    out.save(tmp, format="PNG")
    tmp.replace(dst)


# ---------------------------------------------------------------- specs
def parse_cli_spec(kind: str, value: str) -> dict:
    """'Anna=photos/anna.jpg' or 'photos/anna.jpg' (name from the file name) -> spec dict."""
    if kind not in KINDS:
        raise SystemExit(f"unknown reference kind '{kind}'")
    name, sep, path = value.partition("=")
    if not sep:
        path, name = value, Path(value).stem.replace("_", " ").replace("-", " ")
    path = Path(path.strip()).expanduser()
    if not path.is_file():
        raise SystemExit(f"reference image not found: {path}")
    return {"kind": kind, "name": name.strip() or path.stem, "path": str(path.resolve())}


def clean_spec(spec: dict) -> dict:
    kind = str(spec.get("kind") or "person").lower()
    if kind in ("location", "place", "scene", "hintergrund", "ort"):
        kind = "background"
    if kind not in KINDS:
        raise ValueError(f"unknown reference kind '{kind}' (use person, object or background)")
    name = " ".join(str(spec.get("name") or "").split())[:60]
    out = {"kind": kind, "name": name, "description": " ".join(str(spec.get("description") or "").split())[:240],
           "path": str(spec["path"])}
    if "as_is" in spec:
        out["as_is"] = bool(spec["as_is"])
    if spec.get("id"):
        out["id"] = str(spec["id"])
    return out


def import_refs(pdir: Path, specs: list[dict], existing: list[dict] | None = None) -> list[dict]:
    """Copy (normalised) reference images into <project>/refs and return the updated reference list.
    A reference with the same id is replaced, so re-running `make` with the same images is idempotent."""
    refs = [dict(r) for r in (existing or [])]
    taken = {r["id"] for r in refs}
    old = {r["id"]: r for r in refs}
    added: set = set()
    for raw in specs:
        spec = clean_spec(raw)
        base = slugify(spec.get("id") or spec["name"] or spec["kind"], maxlen=24) or spec["kind"]
        rid = base
        # an explicit id, or the same name as a reference from an earlier upload, replaces that image;
        # two images with the same name in one batch get distinct ids
        replace = (spec.get("id") in old or (base in old and old[base].get("name") == spec["name"])) and base not in added
        if not replace:
            n = 2
            while rid in taken:
                rid, n = f"{base}-{n}", n + 1
        dst = pdir / "refs" / f"{rid}.png"
        sha = normalize_image(Path(spec["path"]), dst)
        ref = {"id": rid, "kind": spec["kind"], "name": spec["name"] or rid, "description": spec["description"],
               "image": f"refs/{rid}.png", "sha": sha,
               "as_is": spec.get("as_is", spec["kind"] == "background")}
        refs = [r for r in refs if r["id"] != rid] + [ref]
        taken.add(rid)
        added.add(rid)
        log(f"reference '{ref['name']}' ({ref['kind']}) -> {ref['image']}")
    return refs


def fingerprint(specs: list[dict]) -> list:
    """Content identity of a list of specs (for the project's request record)."""
    return sorted([clean_spec(s)["kind"], clean_spec(s)["name"], clean_spec(s)["description"], _sha(Path(s["path"]))]
                  for s in specs)


# ---------------------------------------------------------------- shots
def _name_pattern(name: str) -> re.Pattern | None:
    name = name.strip()
    if len(name) < 2:
        return None
    return re.compile(r"(?<![\w])" + re.escape(name) + r"(?![\w])", re.IGNORECASE)


def assign_by_name(shots: list[dict], refs: list[dict], only: set | None = None) -> None:
    """Add to every shot the references whose name occurs in its narration or visual (existing assignments are
    kept). `only` restricts this to some reference ids, e.g. the ones just added to an existing project."""
    pats = [(r["id"], _name_pattern(r["name"])) for r in refs if only is None or r["id"] in only]
    for shot in shots:
        cur = shot["refs"] if isinstance(shot.get("refs"), list) else []
        text = f"{shot.get('narration', '')} {shot.get('visual', '')}"
        shot["refs"] = cur + [rid for rid, pat in pats if pat and pat.search(text) and rid not in cur]


def sanitize_shot_refs(shots: list[dict], refs: list[dict]) -> None:
    valid = {r["id"] for r in refs}
    for shot in shots:
        if isinstance(shot.get("refs"), list):
            seen = []
            for rid in shot["refs"]:
                if isinstance(rid, str) and rid in valid and rid not in seen:
                    seen.append(rid)
            shot["refs"] = seen


def for_shot(shot: dict, refs_by_id: dict, pdir: Path) -> list[dict]:
    """The shot's references, people first, at most MAX_PER_SHOT, with absolute image paths."""
    chosen = [refs_by_id[r] for r in shot.get("refs") or [] if r in refs_by_id]
    chosen.sort(key=lambda r: KIND_ORDER.get(r["kind"], 9))
    if len(chosen) > MAX_PER_SHOT:
        log(f"  {shot['id']}: {len(chosen)} reference images, using the first {MAX_PER_SHOT} "
            f"({', '.join(r['name'] for r in chosen[:MAX_PER_SHOT])})")
        chosen = chosen[:MAX_PER_SHOT]
    return [dict(r, path=str(pdir / r["image"])) for r in chosen]


def use_as_is(srefs: list[dict]) -> bool:
    return len(srefs) == 1 and srefs[0]["kind"] == "background" and srefs[0].get("as_is", True)


def key_part(srefs: list[dict]) -> list:
    return [[r["id"], r["kind"], r["sha"], r["name"], r["description"], bool(r.get("as_is"))] for r in srefs]


def flux_prompt(scene: str, srefs: list[dict], look_reference: bool = False) -> str:
    """Prompt prefix that tells FLUX.2 what to take from each numbered input image."""
    parts = []
    for i, r in enumerate(srefs, 1):
        who = r["name"] or GENERIC[r["kind"]]
        extra = f" ({r['description']})" if r["description"] else ""
        if r["kind"] == "person":
            parts.append(f"Image {i} shows {who}{extra}. Show {who} with exactly the same face, facial features, "
                         f"hairstyle, skin tone and body as in image {i}; keep the clothing unless the scene "
                         f"describes different clothing.")
        elif r["kind"] == "object":
            parts.append(f"Image {i} shows {who}{extra}. Reproduce it exactly as in image {i}: same shape, "
                         f"colors, materials and details.")
        else:
            parts.append(f"Image {i} shows the location {who}{extra}. Set the scene in exactly this place: same "
                         f"architecture, furniture, landscape, colors and light as in image {i}.")
    if look_reference:
        n = len(srefs) + 1
        parts.append(f"Image {n} only defines the overall visual style, color palette and lighting; do not copy "
                     f"its content.")
    parts.append("Do not show the reference images themselves, no collage, no split screen. Scene: " + scene)
    return " ".join(parts)


def plain_text(text: str, srefs: list[dict]) -> str:
    """Replace reference names in a text by their description (for models that never saw the photos, e.g. Wan's
    text encoder — the photo is already baked into the keyframe)."""
    for r in srefs:
        pat = _name_pattern(r["name"] or "")
        if pat:
            text = pat.sub(r["description"] or GENERIC[r["kind"]], text)
    return text


def planner_block(refs: list[dict]) -> str:
    lines = []
    for r in refs:
        desc = f": {r['description']}" if r["description"] else ""
        lines.append(f'- id "{r["id"]}" ({r["kind"]}) called "{r["name"]}"{desc}')
    return ("The user supplied photos that must appear in the video:\n" + "\n".join(lines) + "\n"
            "Use them wherever they fit the narration. In a shot that shows one of them, call it by its name "
            "(for example \"" + refs[0]["name"] + "\") instead of describing its looks — the look comes from the "
            "photo. Photos of kind background are locations: set those shots in that place. ")
