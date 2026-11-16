"""Command line interface: aivs make | render | verify | fetch | ui"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, pipeline
from .util import log, need, read_json


def _settings_from(args) -> dict:
    s = {}
    for key in ("aspect", "voice", "speed", "ref_audio", "music", "captions", "motion", "seed", "transition"):
        val = getattr(args, key, None)
        if val is not None:
            s[key] = val
    if getattr(args, "wan_shots", None):
        s["wan_shots"] = [x.strip() for x in args.wan_shots.split(",") if x.strip()]
    if getattr(args, "tts", None):
        s["tts_engine"] = args.tts
    if getattr(args, "steps", None):
        s["image_steps"] = args.steps
    if getattr(args, "draft", False):
        s["draft"] = True
    if getattr(args, "consistency", None):
        s["consistency"] = args.consistency
    if getattr(args, "lipsync", None):
        s["lipsync"] = args.lipsync
    if getattr(args, "wan_length", None):
        s["wan_length"] = args.wan_length
    if getattr(args, "talk", None):
        s["talk"] = args.talk
    if getattr(args, "lipsync_steps", None):
        s["lipsync_steps"] = args.lipsync_steps
    return s


def _cast_specs(args) -> tuple[list[dict], list[dict]]:
    """--cast JSON -> (character specs, extra person-photo reference specs)."""
    if not getattr(args, "cast", None):
        return [], []
    path = Path(args.cast)
    data = json.loads(path.read_text(encoding="utf-8"))
    chars, photos = [], []
    for c in (data.get("characters", data) if isinstance(data, dict) else data):
        c = dict(c)
        for key in ("photo", "voice_sample"):
            if c.get(key):
                p = Path(c[key]).expanduser()
                p = p if p.is_absolute() else (path.parent / p).resolve()
                if not p.is_file():
                    raise SystemExit(f"{key} not found: {p}")
                c[key] = str(p)
        if c.get("photo"):
            photos.append({"kind": "person", "name": c["name"], "description": c.get("look", ""), "path": c.pop("photo")})
        chars.append(c)
    return chars, photos


def _ref_specs(args) -> list[dict]:
    from . import references
    specs = [references.parse_cli_spec("person", v) for v in getattr(args, "person", None) or []]
    specs += [references.parse_cli_spec("object", v) for v in getattr(args, "object_", None) or []]
    specs += [references.parse_cli_spec("background", v) for v in getattr(args, "background", None) or []]
    if getattr(args, "references", None):
        path = Path(args.references)
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("references", data) if isinstance(data, dict) else data:
            item = dict(item)
            p = Path(item.get("path") or item.get("image") or "").expanduser()
            item["path"] = str(p if p.is_absolute() else (path.parent / p).resolve())
            if not Path(item["path"]).is_file():
                raise SystemExit(f"reference image not found: {item['path']}")
            specs.append(item)
    return specs


def _board_ref_specs(board: dict, base: Path) -> list[dict]:
    """References embedded in a storyboard JSON ({"references": [{..., "image": "path"}]}), paths relative to it."""
    out = []
    for r in board.get("references") or []:
        p = Path(r.get("path") or r.get("image") or "").expanduser()
        p = p if p.is_absolute() else (base / p).resolve()
        if not p.is_file():
            raise SystemExit(f"storyboard reference image not found: {p}")
        out.append(dict(r, path=str(p)))
    return out


def _add_render_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--aspect", choices=list(config.ASPECTS), help="output format (default 16:9 = 1920x1080)")
    p.add_argument("--draft", action="store_true", help="2/3 resolution for fast previews")
    p.add_argument("--voice", help="Kokoro voice name, e.g. af_heart, am_michael, bf_emma, bm_george")
    p.add_argument("--tts", choices=["auto", "kokoro", "chatterbox"], help="narration engine (default auto by language)")
    p.add_argument("--ref-audio", dest="ref_audio", help="5-15 s voice sample to clone (Chatterbox; you must own the rights)")
    p.add_argument("--speed", type=float, help="narration speed factor (default 1.0)")
    p.add_argument("--music", help="none | calm | wonder | uplifting | dark | /path/to/track.wav (default calm)")
    p.add_argument("--captions", choices=["burn", "srt", "off"], help="burn = in picture + .srt (default), srt = sidecar only")
    p.add_argument("--motion", choices=["wan", "parallax", "camera"], help="wan = generative image-to-video for every "
                   "shot (default; ~10 min per shot on M5/16 GB, needs scripts/setup_wan.sh, falls back to parallax "
                   "per shot if it fails); parallax = depth-aware 2.5D camera over stills (fast); camera = flat zoom/pan")
    p.add_argument("--wan-shots", dest="wan_shots", help="with --motion wan: comma-separated shot ids to generate "
                   "(e.g. s02,s05); the other shots use parallax. Default: all shots")
    p.add_argument("--wan-length", dest="wan_length", choices=["native", "stretch"],
                   help="native (default): every scene animated at real speed, in chunks of up to 5 s; stretch: one "
                        "2 s clip slowed down to the scene length (faster, but slow motion and ghosting)")
    p.add_argument("--talk", choices=["auto", "off"],
                   help="characters speaking on screen: auto (default) = whole scene generated from keyframe + line with "
                        "InfiniteTalk when scripts/setup_talk.sh was run (else Wan + lip-sync); off = Wan + lip-sync")
    p.add_argument("--lipsync-steps", dest="lipsync_steps", type=int,
                   help="lip-sync diffusion steps (default 20; 10 measured as good at half the time)")
    g = p.add_argument_group("reference images (people, objects, places that must appear in the video)")
    g.add_argument("--person", action="append", default=[], metavar="NAME=IMAGE",
                   help="photo of a person, e.g. --person \"Anna=photos/anna.jpg\"; use the name in your script or "
                        "prompt. Only use photos of people who agreed to this. Repeatable")
    g.add_argument("--object", dest="object_", action="append", default=[], metavar="NAME=IMAGE",
                   help="photo of an object/product that must look exactly like this. Repeatable")
    g.add_argument("--background", action="append", default=[], metavar="NAME=IMAGE",
                   help="photo of a place; shots set there use it (alone in a shot it becomes the keyframe as is). "
                        "Repeatable")
    d = p.add_argument_group("dialogue (several speakers)")
    d.add_argument("--lipsync", choices=["auto", "off"], help="match mouth movement of characters speaking on screen "
                   "(auto = when scripts/setup_lipsync.sh was run; default auto)")
    g.add_argument("--references", metavar="JSON", help='file with [{"kind": "person|object|background", "name": ..., '
                   '"description": ..., "path": ..., "as_is": true|false}] (paths relative to the file)')
    p.add_argument("--transition", help="ffmpeg xfade transition name (default fade)")
    p.add_argument("--steps", type=int, help="image diffusion steps (default 4)")
    p.add_argument("--consistency", choices=["auto", "on", "off"], help="use shot 1 as the look reference for all other "
                   "keyframes (auto = when the storyboard has a recurring subject)")
    p.add_argument("--seed", type=int)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aivs", description="AI Video Studio — script or prompt to finished video, fully local.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make", help="create a project and render it")
    src = mk.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="topic/idea; the script is written for you")
    src.add_argument("--script", help="path to a text file with the narration (used word for word)")
    src.add_argument("--storyboard", help="path to a storyboard JSON (see examples/)")
    mk.add_argument("--lang", default="en", help="narration language code (en, de, fr, es, ...)")
    mk.add_argument("--seconds", type=int, default=40, help="target length when using --prompt")
    mk.add_argument("--style", help="visual style line applied to every shot")
    mk.add_argument("--name", help="project folder name")
    mk.add_argument("--replan", action="store_true", help="write a new storyboard even if this project already has one")
    mk.add_argument("--plan-only", action="store_true", help="stop after writing project.json so you can edit the storyboard")
    mk.add_argument("--characters", type=int, default=0, help="number of speaking characters: the video becomes a "
                    "dialogue between them (characters not defined with --cast/--person are invented, look and voice "
                    "included). A --script with 'Name: line' lines is a dialogue automatically")
    mk.add_argument("--narrator", action="store_true", help="with --characters: an off-screen narrator may speak too")
    mk.add_argument("--cast", metavar="JSON", help='characters: [{"name": "Anna", "gender": "female", "age": "30s", '
                    '"look": "...", "voice": "warm low voice ...", "photo": "anna.jpg", "voice_sample": "anna.wav"}] '
                    "(all fields optional except name; paths relative to the file)")
    _add_render_options(mk)

    rd = sub.add_parser("render", help="(re)render an existing project folder; only changed parts are recomputed")
    rd.add_argument("project")
    _add_render_options(rd)

    vf = sub.add_parser("verify", help="independently check audio/caption sync of a rendered project")
    vf.add_argument("project")

    ft = sub.add_parser("fetch", help="download models with the resumable multi-connection fetcher")
    ft.add_argument("repos", nargs="*", help="HF repo ids (default: all models of the standard pipeline)")

    ui = sub.add_parser("ui", help="start the local web studio")
    ui.add_argument("--port", type=int, default=8760)
    ui.add_argument("--host", default="127.0.0.1")

    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        from . import fetch
        fetch.main(args.repos)
        return 0
    if args.cmd == "ui":
        from .web import server
        server.serve(args.host, args.port)
        return 0
    need("ffmpeg"), need("ffprobe")

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if (path / "project.json").exists() else config.PROJECTS / p

    if args.cmd == "make":
        script = Path(args.script).read_text(encoding="utf-8") if args.script else None
        board = json.loads(Path(args.storyboard).read_text(encoding="utf-8")) if args.storyboard else None
        lang = (board or {}).get("language", args.lang)
        refs = (_board_ref_specs(board, Path(args.storyboard).resolve().parent) if board else []) + _ref_specs(args)
        chars, photos = _cast_specs(args)
        refs += photos
        if args.characters < 0:
            raise SystemExit("--characters must be 0 or more")
        pdir = pipeline.create_project(topic=args.prompt, script=script, storyboard=board, lang=lang,
                                       seconds=args.seconds, style=args.style, name=args.name,
                                       settings=_settings_from(args), replan=args.replan, refs=refs or None,
                                       cast=chars or None, characters=args.characters, narrator=args.narrator)
        log(f"project folder: {pdir}")
        if args.plan_only:
            print(json.dumps(read_json(pdir / "project.json")["storyboard"], indent=2, ensure_ascii=False))
            return 0
        pipeline.render(pdir)
        return 0
    if args.cmd == "render":
        pdir = resolve(args.project)
        proj = read_json(pdir / "project.json")
        if not proj:
            raise SystemExit(f"no project at {pdir}")
        proj["settings"] = proj["settings"] | _settings_from(args)
        from .util import write_json
        write_json(pdir / "project.json", proj)
        specs = _ref_specs(args)
        if specs:
            pipeline.add_references(pdir, specs)
        pipeline.render(pdir)
        return 0
    if args.cmd == "verify":
        print(json.dumps(pipeline.verify(resolve(args.project)), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
