"""End-to-end orchestration. Every stage is cached by a content hash, so an interrupted run resumes where it stopped."""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

from . import captions, cast as C, config, edit, fetch, images, motion, music, planner, references, safety, text as T
from .util import digest, log, media_duration, probe, read_json, run_worker, slugify, write_json

MAX_WER = 0.25  # a narration take is rejected and re-synthesised above this word error rate
RETRIES = 2


# ---------------------------------------------------------------- settings
def default_settings() -> dict:
    return {
        "aspect": "16:9", "draft": False, "fps": config.FPS, "seed": 7,
        "voice": None, "speed": 1.0, "ref_audio": None, "tts_engine": "auto",
        "image_steps": 4,
        "consistency": "auto",  # auto = on when the storyboard names a recurring subject | on | off
        # wan (default) = generative image-to-video for every shot, ~10 min per shot on M5/16 GB; a shot whose Wan
        # run fails, or every shot when Wan is not installed, falls back to parallax. parallax | camera = fast.
        # A shot can override this with shot["animate"] = "wan" | "parallax" | "camera".
        "motion": "wan",
        "wan_shots": None,     # with motion=wan: list of shot ids to generate; None = all. Others use parallax.
        "wan_steps": 20,
        # native = the shot at real speed in chunks of up to wan_chunk_frames (24 fps, each chunk continues from the
        # previous one's last frame); stretch = one 2 s clip slowed down to the shot length (faster, ghosting)
        "wan_length": "native", "wan_chunk_frames": 81,
        # characters speaking on screen: auto = generated with InfiniteTalk from keyframe + line when installed
        # (scripts/setup_talk.sh), else Wan + lip-sync | off
        "talk": "auto", "talk_steps": "1000,750,500,250", "talk_audio_guide": 2.0,
        # lip-sync for characters who speak on screen: auto = when envs/lipsync is installed | off
        "lipsync": "auto", "lipsync_resolution": 512, "lipsync_steps": 20,
        "music": "calm", "music_gain_db": -15.0,
        "captions": "burn",  # burn | srt | off
        "transition": "fade",
    }


def sizes(settings: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    out, key = config.ASPECTS[settings["aspect"]]
    if settings.get("draft"):
        out = tuple(int(round(v * config.DRAFT_SCALE / 2) * 2) for v in out)
        key = tuple(int(round(v * config.DRAFT_SCALE / 16) * 16) for v in key)
    return out, key


def tts_setup(lang: str, settings: dict) -> dict:
    lang = lang.lower()
    engine = settings.get("tts_engine", "auto")
    if engine == "auto":
        engine = "kokoro" if lang in config.KOKORO_LANGS and not settings.get("ref_audio") else "chatterbox"
    if engine == "kokoro":
        code = config.KOKORO_LANGS.get(lang)
        if not code:
            raise SystemExit(f"Kokoro is configured for English only; use --tts chatterbox for '{lang}'")
        return {"engine": "kokoro", "model": fetch.resolve_model(config.MODELS["tts_kokoro"]), "lang": code,
                "voice": settings.get("voice") or config.KOKORO_DEFAULT_VOICE[code], "speed": settings["speed"]}
    base = lang.split("-")[0]
    if base not in config.CHATTERBOX_LANGS:
        raise SystemExit(f"language '{lang}' is not supported by Chatterbox Multilingual "
                         f"({', '.join(sorted(config.CHATTERBOX_LANGS))})")
    # Upstream guidance: when the reference voice speaks another language than the target (the built-in voice is
    # English), cfg_weight 0 prevents the accent from carrying over.
    cross_lingual = base != "en" and not settings.get("ref_audio")
    return {"engine": "chatterbox", "model": fetch.resolve_model(config.MODELS["tts_chatterbox"]), "lang": base,
            "speed": settings["speed"], "ref_audio": settings.get("ref_audio"),
            "exaggeration": 0.4, "cfg_weight": 0.0 if cross_lingual else 0.5}


# ---------------------------------------------------------------- stages
def _save_board(pdir: Path, board: dict) -> None:
    """Write cast/reference changes made during a render back to project.json (without runtime-only keys)."""
    proj = read_json(pdir / "project.json")
    clean = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    sb = clean(board)
    sb["shots"] = [clean(s) for s in board["shots"]]
    if "cast" in sb:
        sb["cast"] = [clean(c) for c in board["cast"]]
    if "references" in sb:
        sb["references"] = [clean(r) for r in board["references"]]
    proj["storyboard"] = sb
    write_json(pdir / "project.json", proj)


def _ensure_model(repo: str) -> str:
    path = fetch.resolve_model(repo)
    if path == repo:  # not fetched yet: use the resumable downloader rather than the HF client
        fetch.fetch(repo)
        path = fetch.resolve_model(repo)
    return path


def stage_cast(pdir: Path, board: dict, settings: dict) -> None:
    """Characters: link uploaded photos by name, generate portraits for the others (used as reference images for
    every keyframe the character appears in), and prepare one voice per character."""
    cast = board.get("cast") or []
    if not cast:
        return
    refs = board.setdefault("references", [])
    changed = False
    for c in cast:  # an uploaded person photo with the character's name is that character
        if not c.get("ref"):
            hit = next((r for r in refs if r["kind"] == "person" and r["name"].lower() == c["name"].lower()), None)
            if hit:
                c["ref"], changed = hit["id"], True
    by_id = {r["id"]: r for r in refs}
    for i, c in enumerate(cast):
        prompt = C.portrait_prompt(c, board["style"])
        pkey = digest([prompt, settings["seed"] + 101 * i, config.MODELS["image"], 1024])
        ref = by_id.get(c.get("ref"))
        if ref and not ref.get("generated"):
            continue  # the user's photo
        if ref and ref.get("portrait_key") == pkey and (pdir / ref["image"]).exists():
            continue
        rid = c.get("ref") if ref else c["id"]
        while not ref and rid in by_id:
            rid += "-p"
        out = pdir / "refs" / f"{rid}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        log(f"portrait for {c['name']}")
        tmp = out.with_name(out.stem + ".tmp.png")
        (pdir / "work").mkdir(exist_ok=True)
        images.generate(prompt, tmp, (1024, 1024), settings["seed"] + 101 * i, settings["image_steps"],
                        pdir / "work" / "images.log")
        tmp.replace(out)
        rec = {"id": rid, "kind": "person", "name": c["name"], "description": c.get("look") or "",
               "image": f"refs/{rid}.png", "sha": references._sha(out), "as_is": False, "generated": True,
               "portrait_key": pkey}
        refs[:] = [r for r in refs if r["id"] != rid] + [rec]
        by_id[rid] = rec
        c["ref"], changed = rid, True
    lang = board["language"].lower()
    if lang in config.KOKORO_LANGS:
        before = [c.get("kokoro_voice") for c in cast]
        C.assign_kokoro(cast)
        changed |= before != [c.get("kokoro_voice") for c in cast]
    else:
        base = lang.split("-")[0]
        design_lang = C.VOICE_DESIGN_LANGS.get(base, "english")
        text = C.SAMPLE_TEXT.get(base if base in C.VOICE_DESIGN_LANGS else "en", C.SAMPLE_TEXT["en"])
        todo = []
        for i, c in enumerate(cast):
            if c.get("voice_sample"):
                continue
            key = digest([C.voice_instruction(c), design_lang, text, config.MODELS["voice_design"], settings["seed"] + i])
            rel = f"refs/voices/{c['id']}-{key}.wav"
            if c.get("voice_design_sample") != rel:
                c["voice_design_sample"], changed = rel, True
            if not (pdir / rel).exists():
                todo.append({"id": c["id"], "text": text, "language": design_lang,
                             "instruct": C.voice_instruction(c), "out": str(pdir / rel)})
        if todo:
            (pdir / "refs" / "voices").mkdir(parents=True, exist_ok=True)
            log(f"designing {len(todo)} character voice(s) with {config.MODELS['voice_design']}")
            run_worker("voice_design", {"model": _ensure_model(config.MODELS["voice_design"]),
                                        "seed": settings["seed"], "items": todo}, pdir / "work")
    if changed:
        _save_board(pdir, board)


def speaker_setup(pdir: Path, board: dict, shot: dict, settings: dict) -> dict:
    """TTS settings for the shot's speaker; narrator lines (and every line of a project without characters)
    use the project voice exactly as before."""
    base = tts_setup(board["language"], settings)
    c = next((c for c in board.get("cast") or [] if c["id"] == shot.get("speaker")), None)
    if not c:
        return base
    lang = board["language"].lower()
    sample = c.get("voice_sample") or c.get("voice_design_sample")
    if sample:
        code = lang.split("-")[0]
        if code not in config.CHATTERBOX_LANGS:
            raise SystemExit(f"language '{lang}' is not supported by Chatterbox Multilingual")
        # the designed sample speaks the target language (or English when VoiceDesign lacks it)
        same = bool(c.get("voice_sample")) or code in C.VOICE_DESIGN_LANGS
        return {"engine": "chatterbox", "model": fetch.resolve_model(config.MODELS["tts_chatterbox"]), "lang": code,
                "speed": settings["speed"], "ref_audio": str(pdir / sample),
                "exaggeration": 0.5, "cfg_weight": 0.5 if same else 0.0}
    if c.get("kokoro_voice") and lang in config.KOKORO_LANGS:
        return dict(base, engine="kokoro", model=fetch.resolve_model(config.MODELS["tts_kokoro"]),
                    lang=config.KOKORO_LANGS[lang], voice=c["kokoro_voice"])
    return base


def stage_narration(pdir: Path, board: dict, settings: dict) -> None:
    """TTS every shot (each in its speaker's voice), verify each take with Whisper (WER), retry bad takes,
    store word timings."""
    adir = pdir / "audio"
    adir.mkdir(exist_ok=True)
    pending = []
    for shot in board["shots"]:
        setup = speaker_setup(pdir, board, shot, settings)
        key = digest([shot["narration"], dict(setup) | {"seed": settings["seed"]}])
        shot["_akey"], shot["_setup"] = key, setup
        meta = read_json(adir / f"{shot['id']}-{key}.json")
        if meta and Path(meta["audio"]).exists():
            shot["_audio"] = meta
        else:
            pending.append(shot)
    take = 0
    per_item = ("voice", "ref_audio", "cfg_weight", "exaggeration", "speed")
    while pending and take <= RETRIES:
        groups: dict = {}
        for s in pending:  # one worker run per TTS model, so only one model is in memory at a time
            st = s["_setup"]
            groups.setdefault((st["engine"], st["model"], st["lang"]), []).append(s)
        durs = {}
        for (engine, _, _), shots in groups.items():
            log(f"narration: {len(shots)} line(s) with {engine} (take {take + 1})")
            items = [{"id": s["id"], "text": s["narration"], "take": take,
                      "out": str(adir / f"{s['id']}-{s['_akey']}-t{take}.wav"),
                      **{k: s["_setup"][k] for k in per_item if k in s["_setup"]}} for s in shots]
            job = {k: v for k, v in shots[0]["_setup"].items()} | {"seed": settings["seed"], "items": items}
            tts = run_worker("tts", job, pdir / "work")
            durs.update({r["id"]: r for r in tts["items"]})
        stt = run_worker("stt", {"model": fetch.resolve_model(config.MODELS["stt"]), "lang": board["language"],
                                 "items": [{"id": i, "audio": r["out"]} for i, r in durs.items()]}, pdir / "work")
        heard = {r["id"]: r for r in stt["items"]}
        still = []
        for s in pending:
            d, h = durs[s["id"]], heard[s["id"]]
            wer = T.word_error_rate(s["narration"], h["text"])
            meta = {"audio": d["out"], "duration": d["duration"], "wer": round(wer, 3), "heard": h["text"],
                    "take": take, "words": T.align_words(s["narration"], h["words"], d["duration"])}
            best = s.get("_best")
            if best is None or wer < best["wer"]:
                s["_best"] = meta
            if wer > MAX_WER and take < RETRIES:
                log(f"  {s['id']}: WER {wer:.2f} > {MAX_WER} — retrying. heard: {h['text'][:80]!r}")
                still.append(s)
            else:
                s["_audio"] = s.pop("_best")
                write_json(adir / f"{s['id']}-{s['_akey']}.json", s["_audio"])
        pending, take = still, take + 1


def build_timeline(board: dict, settings: dict) -> dict:
    fps = settings["fps"]
    xf = int(round(config.XFADE * fps))
    cum, n = 0, len(board["shots"])
    pre_roll, post_roll = ((config.DIALOGUE_PRE, config.DIALOGUE_POST) if board.get("cast")
                           else (config.PRE_ROLL, config.POST_ROLL))
    for i, shot in enumerate(board["shots"]):
        pre = pre_roll + (config.INTRO if i == 0 else 0.0)
        post = post_roll + (config.OUTRO if i == n - 1 else 0.0)
        slot = math.ceil((pre + shot["_audio"]["duration"] + post) * fps)
        shot["_cut_frame"] = cum
        shot["_frames"] = slot + xf
        shot["_narr_start"] = round((cum + xf / 2) / fps + pre, 4)
        cum += slot
    return {"fps": fps, "xfade_frames": xf, "total_frames": cum + xf, "duration": (cum + xf) / fps}


def use_reference(board: dict, settings: dict) -> bool:
    mode = settings.get("consistency", "auto")
    if isinstance(mode, bool):
        return mode
    # With user-supplied reference photos the photos define the look of people and places; the automatic
    # shot-1 look reference would compete with them, so "auto" turns it off.
    return mode == "on" or (mode == "auto" and bool(board.get("recurring_subject")) and not board.get("references"))


def _with_cast_refs(shot: dict, board: dict) -> dict:
    """The shot with its characters' photos/portraits added: the speaker (when on screen) first, then every
    character named in the visual description."""
    cast = board.get("cast") or []
    if not cast:
        return shot
    extra = []
    for c in cast:
        if not c.get("ref"):
            continue
        speaking = shot.get("speaker") == c["id"] and shot.get("on_screen", True)
        named = references._name_pattern(c["name"])
        if speaking or (named and named.search(shot.get("visual", ""))):
            extra.append(c["ref"])
    own = shot.get("refs") or []
    return dict(shot, refs=extra + [r for r in own if r not in extra])


def stage_keyframes(pdir: Path, board: dict, settings: dict) -> None:
    fdir = pdir / "frames"
    fdir.mkdir(exist_ok=True)
    _, key_size = sizes(settings)
    ref_key, consistent = None, use_reference(board, settings)
    if consistent:
        log("keyframes: shot 1 is the look reference for all other shots")
    refs_by_id = {r["id"]: r for r in board.get("references") or []}
    for i, shot in enumerate(board["shots"]):
        srefs = references.for_shot(_with_cast_refs(shot, board), refs_by_id, pdir) if refs_by_id else []
        parts = [shot["visual"], board["style"], key_size, settings["seed"] + i, settings["image_steps"],
                 config.MODELS["image"], ref_key, board.get("recurring_subject") if consistent else None]
        if srefs:  # appended only when present, so keyframes of projects without references keep their cache keys
            parts.append(["refs", references.key_part(srefs)])
        key = digest(parts)
        if consistent and i == 0:
            ref_key = key
        shot["_ikey"] = key
        shot["_refs"] = srefs
        shot["image"] = str(fdir / f"{shot['id']}-{key}.png")
    images.render_keyframes(board["shots"], board["style"], fdir, key_size, settings["seed"],
                            settings["image_steps"], consistent, board.get("recurring_subject") or "")


def stage_motion(pdir: Path, board: dict, settings: dict) -> None:
    cdir = pdir / "clips"
    cdir.mkdir(exist_ok=True)
    out_size, _ = sizes(settings)
    backend = settings["motion"]
    kinds = ("parallax", "camera", "wan")
    if backend not in kinds:
        raise SystemExit(f"unknown motion backend '{backend}' (available: {', '.join(kinds)})")
    ids = {s["id"] for s in board["shots"]}
    wanted = settings.get("wan_shots")
    unknown = set(wanted or []) - ids
    if unknown:
        raise SystemExit(f"--wan-shots: no such shot(s): {', '.join(sorted(unknown))}")
    fallback = "camera" if backend == "camera" else "parallax"
    for shot in board["shots"]:
        own = shot.get("animate")
        if own in kinds:                       # per-shot choice from the storyboard / web editor wins
            shot["_backend"] = own
        elif backend == "wan":
            shot["_backend"] = "wan" if (not wanted or shot["id"] in wanted) else "parallax"
        else:
            shot["_backend"] = fallback
    shot_notes: dict = {}
    cast_by_id = {c["id"]: c for c in board.get("cast") or []}
    for shot in board["shots"]:  # prompts for the generative backends (names -> descriptions, performance)
        srefs = shot.get("_refs") or []
        shot["_wan_visual"] = references.plain_text(shot["visual"], srefs) if srefs else shot["visual"]
        spk = cast_by_id.get(shot.get("speaker"))
        if spk and shot.get("on_screen", True):  # direct the performance: a speaking character really talks
            # calm wording on purpose: "lively expressions + gestures" made Wan overact (grimaces, wide-open mouth)
            action = (f"{spk['name']} is speaking calmly and naturally, lips moving with the words, relaxed face, "
                      f"subtle head movements")
            shot["_wan_action"] = references.plain_text(action, srefs) if srefs else action
    # characters speaking on screen: the talking model makes the whole shot from keyframe + line (Wan if unavailable)
    talk_shots = [s for s in board["shots"] if s["_backend"] == "wan" and s.get("_wan_action")
                  and settings.get("talk", "auto") != "off"]
    if talk_shots:
        from . import i2v, lipsync, talk
        reason = talk.unavailable_reason()
        if reason:
            log(f"note: the talking-character model is not installed ({reason}); speaking shots use Wan + lip-sync")
        else:
            log(f"talking shots: InfiniteTalk for {len(talk_shots)} shot(s) — slow (measure: see report.json)")
        for n, shot in enumerate(talk_shots if not reason else [], 1):
            frames25 = math.ceil(shot["_frames"] / settings["fps"] * talk.FPS) + 1
            offset = round(shot["_narr_start"] - shot["_cut_frame"] / settings["fps"], 4)
            prompt = i2v.build_prompt(shot, board["style"])
            out = talk.clip_path(shot, settings, cdir, shot["_akey"], offset, frames25, prompt)
            try:
                if not out.exists():
                    audio16 = pdir / "work" / f"talk-{shot['id']}.wav"
                    lipsync.line_audio(Path(shot["_audio"]["audio"]), offset, frames25 / talk.FPS, audio16)
                    t = time.time()
                    talk.generate(shot, audio16, out, prompt=prompt, frames=frames25, settings=settings,
                                  workdir=pdir / "work")
                    log(f"  talking shot {shot['id']} done ({n}/{len(talk_shots)}) in {(time.time() - t) / 60:.1f} min")
                shot["_backend"], shot["_talk_raw"] = "talk", str(out)
            except Exception as exc:  # fall back to Wan (+ lip-sync) for this shot
                log(f"WARNING: talking model failed for {shot['id']} ({exc}); using Wan for this shot. "
                    f"Details: {pdir / 'work' / 'talk.log'}")
                shot_notes[shot["id"]] = f"talking model failed: {exc}"
    wan_shots = [s for s in board["shots"] if s["_backend"] == "wan"]
    if wan_shots:
        from . import i2v
        reason = i2v.unavailable_reason()
        if reason:
            log(f"WARNING: generative motion (Wan) is not available — {reason}. Using the 2.5D camera instead.")
            for shot in wan_shots:
                shot["_backend"], shot_notes[shot["id"]] = "parallax", f"wan unavailable: {reason}"
            wan_shots = []
    if wan_shots:
        seconds = lambda s: s["_frames"] / settings["fps"]
        runs = {s["id"]: i2v.raw_paths(s, board, settings, cdir, seconds(s)) for s in wan_shots}
        todo = [p for ps in runs.values() for p in ps if not p.exists()]
        log(f"motion: Wan image-to-video for {len(wan_shots)} shot(s), {sum(map(len, runs.values()))} run(s)"
            + (f", {len(todo)} to generate" if todo else ", all cached")
            + (" at real speed (no slow motion)" if settings.get("wan_length", "native") != "stretch" else ""))
        for n, shot in enumerate(wan_shots, 1):
            try:
                t = time.time()
                shot["_wan_raw"] = [str(p) for p in i2v.raw_clips(shot, board, settings, cdir, pdir / "work",
                                                                  seconds(shot))]
                if time.time() - t > 5:
                    log(f"  wan {shot['id']} done ({n}/{len(wan_shots)}) in {(time.time() - t) / 60:.1f} min")
            except Exception as exc:  # one failed shot must not cost the whole render
                log(f"WARNING: Wan failed for {shot['id']} ({exc}); using the 2.5D camera for this shot. "
                    f"Details: {pdir / 'work' / 'wan.log'}")
                shot["_backend"], shot_notes[shot["id"]] = "parallax", f"wan failed: {exc}"
    board["_motion_notes"] = shot_notes
    need_depth = []
    for shot in board["shots"]:
        if shot["_backend"] == "parallax":
            shot["_depth"] = str(pdir / "frames" / f"{shot['id']}-{shot['_ikey']}.depth.png")
            if not Path(shot["_depth"]).exists():
                need_depth.append({"id": shot["id"], "image": shot["image"], "out": shot["_depth"]})
    if need_depth:
        log(f"depth maps for {len(need_depth)} keyframe(s)")
        run_worker("depth", {"model": fetch.resolve_model(config.MODELS["depth"]), "items": need_depth}, pdir / "work")
    for shot in board["shots"]:
        kind = shot["_backend"]
        tmp_key = [shot["_ikey"], shot["_frames"], shot["motion"], out_size, settings["fps"], kind]
        if kind == "wan":
            raws = [Path(p) for p in shot["_wan_raw"]]
            tmp_key.append(raws[0].name if len(raws) == 1 else [r.name for r in raws])  # 1 run: key as before
        if kind == "talk":
            tmp_key.append(Path(shot["_talk_raw"]).name)
        key = digest(tmp_key)
        shot["_clip"] = str(cdir / f"{shot['id']}-{key}.mp4")
        if Path(shot["_clip"]).exists():
            continue
        tmp = Path(shot["_clip"]).with_suffix(".tmp.mp4")
        if kind == "talk":
            from . import talk
            talk.conform(Path(shot["_talk_raw"]), tmp, shot["_frames"], out_size, settings["fps"],
                         pdir / "work" / "edit.log")
            log(f"talking clip {shot['id']}: {shot['_frames'] / settings['fps']:.1f}s")
        elif kind == "wan":
            native = settings.get("wan_length", "native") != "stretch"
            stretch = i2v.conform(raws, tmp, shot["_frames"], out_size, settings["fps"], pdir / "work" / "edit.log",
                                  native=native)
            log(f"wan clip {shot['id']}: " + (f"{len(raws)} part(s) at real speed" if native else f"slowed {stretch:.1f}x")
                + f", {shot['_frames'] / settings['fps']:.1f}s")
        else:
            log(f"{kind} move {shot['id']}: {shot['motion']} ({shot['_frames'] / settings['fps']:.1f}s)")
            if kind == "parallax":
                motion.parallax_clip(Path(shot["image"]), Path(shot["_depth"]), tmp, shot["_frames"], out_size,
                                     settings["fps"], shot["motion"])
            else:
                motion.camera_clip(Path(shot["image"]), tmp, shot["_frames"], out_size, settings["fps"], shot["motion"])
        tmp.replace(shot["_clip"])


def stage_lipsync(pdir: Path, board: dict, settings: dict, tl: dict) -> None:
    """Re-render the mouth of characters who speak on screen so it matches their line (LatentSync)."""
    ids = {c["id"] for c in board.get("cast") or []}
    todo = [s for s in board["shots"] if s.get("speaker") in ids and s.get("on_screen", True)
            and s.get("_backend") != "talk"]  # the talking model already made mouth and line match
    notes = board.setdefault("_lipsync_notes", {})
    if not todo or settings.get("lipsync", "auto") == "off":
        return
    from . import lipsync
    reason = lipsync.unavailable_reason()
    if reason:
        log(f"note: lip-sync is not installed ({reason}); {len(todo)} speaking shot(s) keep their mouth motion")
        for s in todo:
            notes[s["id"]] = f"lip-sync unavailable: {reason}"
        return
    out_size, _ = sizes(settings)
    fps, res, steps = settings["fps"], int(settings["lipsync_resolution"]), int(settings["lipsync_steps"])
    cdir, work = pdir / "clips", pdir / "work"
    cast_by_id = {c["id"]: c for c in board.get("cast") or []}
    refs_by_id = {r["id"]: r for r in board.get("references") or []}
    for n, s in enumerate(todo, 1):
        offset = round(s["_narr_start"] - s["_cut_frame"] / fps, 4)
        # the speaker's photo/portrait (and those of the other characters) tell the runner which face to animate
        # when several people are in the shot
        ref = refs_by_id.get(cast_by_id[s["speaker"]].get("ref") or "")
        ref_face = pdir / ref["image"] if ref and (pdir / ref["image"]).is_file() else None
        others = [refs_by_id[c["ref"]] for c in cast_by_id.values()
                  if c["id"] != s["speaker"] and c.get("ref") in refs_by_id] if ref_face else []
        others = [r for r in others if (pdir / r["image"]).is_file()]
        key = digest([Path(s["_clip"]).name, s["_akey"], offset, s["_frames"], res, steps, settings["seed"],
                      ref.get("sha") if ref_face else None, [r.get("sha") for r in others], lipsync.VERSION])
        out = cdir / f"{s['id']}-{key}.lipsync.mp4"
        if not out.exists():
            audio16 = work / f"lipsync-{s['id']}.wav"
            try:
                lipsync.line_audio(Path(s["_audio"]["audio"]), offset, s["_frames"] / fps, audio16)
                t = time.time()
                lipsync.run_shot(Path(s["_clip"]), audio16, out, frames=s["_frames"], size=out_size, fps=fps,
                                 resolution=res, steps=steps, seed=settings["seed"], workdir=work, shot_id=s["id"],
                                 ref_face=ref_face, other_faces=[pdir / r["image"] for r in others])
                log(f"  lip-sync {s['id']} done ({n}/{len(todo)}) in {(time.time() - t) / 60:.1f} min")
            except Exception as exc:  # e.g. no face visible: keep the clip without lip-sync
                log(f"WARNING: lip-sync failed for {s['id']} ({exc}); keeping the clip without it. "
                    f"Details: {work / 'lipsync.log'}")
                notes[s["id"]] = f"lip-sync failed: {exc}"
                continue
        s["_clip"], s["_lipsync"] = str(out), True


def _caption_chunks(board: dict, max_chars: int) -> list[dict]:
    """Narrated videos: one stream of captions as before. Dialogue: per line, the speaker's name in front of
    the first caption when the speaker changes, and one colour per character."""
    cast = board.get("cast") or []
    if not cast:
        words = [{"word": w["word"], "start": w["start"] + s["_narr_start"], "end": w["end"] + s["_narr_start"]}
                 for s in board["shots"] for w in s["_audio"]["words"]]
        return captions.tidy(T.caption_chunks(words, max_chars=max_chars))
    names = {c["id"]: c["name"] for c in cast}
    chunks, prev = [], None
    for s in board["shots"]:
        words = [{"word": w["word"], "start": w["start"] + s["_narr_start"], "end": w["end"] + s["_narr_start"]}
                 for w in s["_audio"]["words"]]
        cc = T.caption_chunks(words, max_chars=max_chars)
        spk = s.get("speaker", C.NARRATOR)
        if cc and spk in names and spk != prev:
            cc[0]["text"] = f"{names[spk]}: {cc[0]['text']}"
        colour = C.colour_for(spk, cast)
        for c in cc:
            if colour:
                c["colour"] = colour
        chunks += cc
        prev = spk
    return captions.tidy(chunks)


def stage_edit(pdir: Path, board: dict, settings: dict, tl: dict) -> dict:
    out_size, _ = sizes(settings)
    work = pdir / "work"
    work.mkdir(exist_ok=True)
    narr = work / "narration.wav"
    edit.narration_track([{"audio": s["_audio"]["audio"], "start": s["_narr_start"]} for s in board["shots"]],
                         tl["duration"], narr)
    mus = music.make(settings["music"], tl["duration"] + 1.0, work / "music.wav", settings["seed"])
    loud = edit.mix_audio(narr, mus, tl["duration"], work / "mix.wav", work / "edit.log", settings["music_gain_db"])
    portrait = out_size[1] > out_size[0]
    chunks = _caption_chunks(board, 24 if portrait else 42)
    captions.write_srt(chunks, pdir / "captions.srt")
    ass = None
    if settings["captions"] == "burn":
        ass = work / "captions.ass"
        captions.write_ass(chunks, ass, out_size)
    final = pdir / "final.mp4"
    log("final edit: crossfades, mix, captions, encode")
    edit.assemble([{"path": s["_clip"], "cut_frame": s["_cut_frame"]} for s in board["shots"]], work / "mix.wav",
                  final, fps=tl["fps"], xfade_frames=tl["xfade_frames"], total_frames=tl["total_frames"], ass=ass,
                  log_path=work / "edit.log", crf=20 if settings.get("draft") else 18,
                  preset="medium" if settings.get("draft") else "slow", transition=settings["transition"],
                  title=board["title"])
    return {"loudness": loud, "captions": len(chunks)}


# ---------------------------------------------------------------- entry points
def add_references(pdir: Path, specs: list[dict]) -> list[dict]:
    """Add (or replace) reference images of an existing project. Shots that never had a `refs` list get the new
    references whose name appears in their narration or visual; everything else is left to the user."""
    proj = read_json(pdir / "project.json")
    if not proj:
        raise SystemExit(f"no project.json in {pdir}")
    board = proj["storyboard"]
    before = {r["id"] for r in board.get("references") or []}
    board["references"] = references.import_refs(pdir, specs, board.get("references"))
    new_ids = {r["id"] for r in board["references"]} - before
    references.assign_by_name(board["shots"], board["references"], only=new_ids)
    references.sanitize_shot_refs(board["shots"], board["references"])
    write_json(pdir / "project.json", proj)
    return board["references"]


def remove_reference(pdir: Path, rid: str) -> list[dict]:
    """Detach a reference from the project and all shots (the image file stays in refs/)."""
    proj = read_json(pdir / "project.json")
    if not proj:
        raise SystemExit(f"no project.json in {pdir}")
    board = proj["storyboard"]
    board["references"] = [r for r in board.get("references") or [] if r["id"] != rid]
    for shot in board["shots"]:
        if isinstance(shot.get("refs"), list):
            shot["refs"] = [x for x in shot["refs"] if x != rid]
    write_json(pdir / "project.json", proj)
    return board["references"]


def create_project(*, topic=None, script=None, storyboard=None, lang="en", seconds=40, style=None, name=None,
                   settings: dict | None = None, replan: bool = False, refs: list[dict] | None = None,
                   cast: list[dict] | None = None, characters: int = 0, narrator: bool = False) -> Path:
    """refs: reference image specs [{kind: person|object|background, name, description, path, as_is?}].
    cast: character specs [{name, gender?, age?, look?, voice?, voice_sample?(path)}]; characters: how many speaking
    characters a prompt-made video gets (the ones not given are invented); narrator: add an off-screen narrator.
    In a dialogue project every uploaded person photo becomes a character of that name."""
    cfg = default_settings() | (settings or {})
    if storyboard:
        board = dict(storyboard)
        board.pop("references", None)  # passed in as `refs` (with resolved paths) by the CLI
        board.setdefault("language", lang)
        board.setdefault("style", style or config.DEFAULT_STYLE)
        board.setdefault("title", name or "Untitled")
        for i, s in enumerate(board["shots"]):
            s.setdefault("id", f"s{i + 1:02d}")
            s.setdefault("motion", planner.MOTIONS[i % len(planner.MOTIONS)])
        slug = slugify(name or board["title"])
    else:
        slug = slugify(name or topic or script[:40])
    pdir = config.PROJECTS / slug
    pdir.mkdir(parents=True, exist_ok=True)
    existing = read_json(pdir / "project.json")
    request = {"topic": topic, "script": script, "lang": lang, "seconds": seconds, "style": style,
               "storyboard": digest(storyboard) if storyboard else None}
    if refs:  # key only present with references, so existing projects keep matching their request record
        request["references"] = digest(references.fingerprint(refs))
    if cast or characters or narrator:
        request["cast"] = digest([cast or [], characters, narrator])
    if existing and existing.get("request") == request and not replan:
        existing["settings"] = existing["settings"] | {k: v for k, v in (settings or {}).items()}
        write_json(pdir / "project.json", existing)
        return pdir
    if any(r.get("kind") == "person" for r in refs or []) or any(c.get("photo") for c in cast or []):
        safety.check({"shots": (storyboard or {}).get("shots") or []},
                     [topic or "", script or "", style or ""] + [str(c.get("look", "")) for c in cast or []],
                     has_photos=True)
    ref_list = references.import_refs(pdir, refs) if refs else []
    wants_dialogue = bool(cast or characters) or bool(script and C.parse_dialogue(script)) or \
        bool(storyboard and storyboard.get("cast"))
    cast_specs = [dict(c) for c in cast or []]
    if wants_dialogue:  # uploaded person photos are characters
        known = {str(c.get("name", "")).lower() for c in cast_specs}
        for r in ref_list:
            if r["kind"] == "person" and r["name"].lower() not in known:
                cast_specs.append({"name": r["name"], "look": r["description"]})
                known.add(r["name"].lower())
    for c in cast_specs:  # voice samples the user brought are copied into the project
        if c.get("voice_sample") and Path(c["voice_sample"]).is_file():
            dst = pdir / "refs" / "voices" / f"user-{slugify(c['name'], 20)}{Path(c['voice_sample']).suffix.lower()}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(c["voice_sample"], dst)
            c["voice_sample"] = str(dst.relative_to(pdir))
        elif c.get("voice_sample"):
            raise SystemExit(f"voice sample not found: {c['voice_sample']}")
    if not storyboard:
        board = planner.plan(topic=topic, script=script, lang=lang, seconds=seconds, style=style,
                             workdir=pdir / "work", refs=ref_list, cast=cast_specs,
                             characters=characters if wants_dialogue else 0, narrator=narrator)
    elif board.get("cast"):  # storyboard JSON with a cast: ids + speaker fields
        taken: set = set()
        board["cast"] = [C.clean_entry(c) for c in board["cast"]]
        for c in board["cast"]:
            c["id"] = C.make_id(c.get("id") or c["name"], taken)
        for s in board["shots"]:
            s["speaker"] = C.speaker_id(s.get("speaker") or "", board["cast"]) if s.get("speaker") not in taken else s["speaker"]
            s.setdefault("on_screen", s["speaker"] != C.NARRATOR)
    if ref_list:
        board["references"] = ref_list
        references.sanitize_shot_refs(board["shots"], ref_list)
        references.assign_by_name(board["shots"], ref_list)
        used = {r for s in board["shots"] for r in s.get("refs") or []}
        for r in ref_list:
            if r["id"] not in used:
                log(f"note: reference '{r['name']}' is not used in any shot yet — assign it in the web editor "
                    f"or add \"{r['id']}\" to a shot's \"refs\" in project.json")
    safety.check(board, [topic or "", script or ""])  # also what the planner wrote
    write_json(pdir / "project.json", {"request": request, "settings": cfg, "storyboard": board})
    return pdir


def render(pdir: Path) -> dict:
    t0 = time.time()
    proj = read_json(pdir / "project.json")
    if not proj:
        raise SystemExit(f"no project.json in {pdir}")
    board, settings = proj["storyboard"], default_settings() | proj["settings"]
    safety.check(board)  # edits in the web editor or project.json are checked too
    timings = {}

    def timed(name, fn, *a):
        t = time.time()
        r = fn(*a)
        timings[name] = round(time.time() - t, 1)
        return r

    log(f"project '{board['title']}' — {len(board['shots'])} shots, {settings['aspect']}, lang {board['language']}")
    if board.get("cast"):
        timed("cast", stage_cast, pdir, board, settings)
    timed("narration", stage_narration, pdir, board, settings)
    tl = build_timeline(board, settings)
    log(f"timeline: {tl['duration']:.1f}s, {tl['total_frames']} frames")
    timed("keyframes", stage_keyframes, pdir, board, settings)
    timed("motion", stage_motion, pdir, board, settings)
    if board.get("cast"):
        timed("lipsync", stage_lipsync, pdir, board, settings, tl)
    info = timed("edit", stage_edit, pdir, board, settings, tl)

    final = pdir / "final.mp4"
    meta = probe(final)
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    a = next(s for s in meta["streams"] if s["codec_type"] == "audio")
    report = {
        "title": board["title"], "output": str(final), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(float(meta["format"]["duration"]), 3), "planned_duration_s": round(tl["duration"], 3),
        "video": {"codec": v["codec_name"], "width": v["width"], "height": v["height"], "fps": v["r_frame_rate"]},
        "audio": {"codec": a["codec_name"], "sample_rate": a["sample_rate"], "channels": a["channels"]},
        "size_mb": round(final.stat().st_size / 1e6, 2), **info,
        "stage_seconds": timings, "total_seconds": round(time.time() - t0, 1),
        "models": {k: config.MODELS[k] for k in ("planner", "image", "stt")} | {
            "tts": tts_setup(board["language"], settings)["engine"], "motion": settings["motion"],
            "reference_consistency": use_reference(board, settings)},
        "references": [{k: r[k] for k in ("id", "kind", "name", "image")} for r in board.get("references") or []],
        "motion_fallbacks": board.get("_motion_notes") or {},
        "cast": [{k: c.get(k) for k in ("id", "name", "ref", "kokoro_voice", "voice_design_sample", "voice_sample")}
                 for c in board.get("cast") or []],
        "lipsync_notes": board.get("_lipsync_notes") or {},
        "shots": [{"id": s["id"], "narration": s["narration"], "visual": s["visual"], "motion": s["motion"],
                   "narration_start_s": s["_narr_start"], "narration_duration_s": s["_audio"]["duration"],
                   "wer": s["_audio"]["wer"], "tts_take": s["_audio"]["take"],
                   "motion_backend": s.get("_backend"), "refs": [r["id"] for r in s.get("_refs") or []],
                   "speaker": s.get("speaker"), "lipsync": bool(s.get("_lipsync")),
                   "clip_start_s": round(s["_cut_frame"] / tl["fps"], 3),
                   "clip_duration_s": round(s["_frames"] / tl["fps"], 3)} for s in board["shots"]],
    }
    write_json(pdir / "report.json", report)
    log(f"done: {final}  ({report['duration_s']}s, {report['size_mb']} MB, rendered in {report['total_seconds']}s)")
    return report


def verify(pdir: Path) -> dict:
    """Independent check of the *finished file*.

    Sync: every narration take is located in the decoded final audio by cross-correlation (sample accurate, works
    through music, compression and AAC) and compared with the planned start that also drives the captions.
    Intelligibility: Whisper transcribes the final mix and the result is compared with the script.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import correlate
    from .util import need, run

    report = read_json(pdir / "report.json")
    if not report:
        raise SystemExit("render the project first")
    proj = read_json(pdir / "project.json")
    board, settings = proj["storyboard"], default_settings() | proj["settings"]
    stage_narration(pdir, board, settings)  # cache hit: just reloads the take metadata
    wav = pdir / "work" / "verify.wav"
    run([need("ffmpeg"), "-y", "-v", "error", "-i", pdir / "final.mp4", "-vn", "-ac", "1", "-ar", "48000", wav])
    final, sr = sf.read(wav, dtype="float32")
    rows = []
    for s, rep in zip(board["shots"], report["shots"]):
        src, _ = sf.read(s["_audio"]["audio"], dtype="float32")
        planned = rep["narration_start_s"]
        lo = max(int((planned - 1.5) * sr), 0)
        seg = final[lo: int((planned + 1.5) * sr) + src.size]
        corr = correlate(seg, src, mode="valid", method="fft")
        k = int(np.argmax(corr))
        window = seg[k: k + src.size]
        conf = float(corr[k] / (np.linalg.norm(src) * np.linalg.norm(window) + 1e-9))
        detected = (lo + k) / sr
        rows.append({"id": s["id"], "planned_start_s": planned, "detected_start_s": round(detected, 4),
                     "offset_ms": round((detected - planned) * 1000, 1), "match_confidence": round(conf, 3)})
    stt = run_worker("stt", {"model": fetch.resolve_model(config.MODELS["stt"]), "lang": board["language"],
                             "items": [{"id": "final", "audio": str(wav)}]}, pdir / "work")["items"][0]
    wav.unlink(missing_ok=True)
    script = " ".join(s["narration"] for s in board["shots"])
    offsets = [abs(r["offset_ms"]) for r in rows]
    result = {
        "max_abs_sync_offset_ms": max(offsets),
        "mean_abs_sync_offset_ms": round(sum(offsets) / len(offsets), 2),
        "min_match_confidence": min(r["match_confidence"] for r in rows),
        "duration_error_ms": round((report["duration_s"] - report["planned_duration_s"]) * 1000),
        "wer_final_mix": round(T.word_error_rate(script, stt["text"]), 3),
        "pass": max(offsets) <= 15 and min(r["match_confidence"] for r in rows) >= 0.5,
        "lines": rows,
    }
    report["verification"] = result
    write_json(pdir / "report.json", report)
    return result
