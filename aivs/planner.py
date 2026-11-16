"""Script / prompt -> storyboard. The LLM never rewrites narration it was given; it only adds visuals."""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import cast as C, config, fetch, references as R, text as T
from .util import log, run_worker

LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
              "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
              "no": "Norwegian", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "hi": "Hindi",
              "ar": "Arabic", "el": "Greek", "he": "Hebrew", "ms": "Malay", "sw": "Swahili"}
MOTIONS = ["push_in", "pull_out", "pan_left", "pan_right", "pan_up", "pan_down"]

SCRIPT_SYSTEM = (
    "You are an award-winning documentary scriptwriter. You write vivid, concrete voice-over narration that sounds "
    "natural when read aloud: short clear sentences, no lists, no headings, no stage directions, no emojis, "
    "no quotation marks, numbers written as words."
)
DIALOGUE_SYSTEM = (
    "You are an award-winning screenwriter. You write natural, vivid spoken dialogue for short films: every line "
    "sounds like real people talking, short sentences, distinct personalities, no stage directions, no actions in "
    "brackets, no emojis, no quotation marks, numbers written as words. Answer with JSON only."
)
VISUAL_SYSTEM = (
    "You are a film director and cinematographer creating a storyboard. For every narration beat you describe ONE "
    "photographable image for an AI image generator. Rules: describe only what is visible (subject, action, setting, "
    "lighting, camera framing); be concrete and specific; 25 to 45 words each; always English; never include text, "
    "captions, logos, watermarks or split screens; scenes must be physically plausible and realistic unless the "
    "narration itself is fantastical; keep recurring subjects visually identical by repeating their "
    "full description every time they appear; vary shot sizes (wide, medium, close-up, macro, aerial). "
    "Write literally, because every word is drawn exactly as written: no figurative language (\"glowing eyes\" "
    "becomes eyes that emit light, \"fire in her eyes\" becomes flames); describe the visible equivalent instead "
    "(\"bright, attentive eyes\"). People look natural and human: no glowing, supernatural or horror features unless "
    "the narration explicitly asks for them. Say what the people are doing (their movement or action), because the "
    "image will be animated. Answer with JSON only."
)
# Figurative phrases the image/video models take literally (seen: "glowing eyes" -> eyes emitting blue light).
FIGURATIVE = [(re.compile(r"\bglowing (eyes|skin|face|gaze)\b", re.I), r"bright \1"),
              (re.compile(r"\b(eyes|skin|face|gaze) (?:are |is )?glow(?:ing|s)?\b", re.I), r"\1 bright")]


def literal(text: str) -> str:
    for pat, rep in FIGURATIVE:
        text = pat.sub(rep, text)
    return text


def _extract_json(raw: str):
    raw = re.sub(r"```(?:json)?", "", raw)
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object in LLM answer")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            if ch != "\\":
                esc = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])
    raise ValueError("unterminated JSON in LLM answer")


def _ask(requests: list[dict], workdir: Path) -> dict:
    job = {"model": fetch.resolve_model(config.MODELS["planner"]), "requests": requests}
    return run_worker("llm", job, workdir)["responses"]


def write_script(topic: str, lang: str, seconds: int, workdir: Path) -> str:
    words = int(seconds * 2.4)
    user = (
        f"Write the voice-over narration for a {seconds}-second short documentary video about:\n{topic}\n\n"
        f"Language: {LANG_NAMES.get(lang, lang)}. Length: about {words} words (never more than {int(words * 1.15)}). "
        "Open with a hook, build one clear idea per sentence, end with a memorable closing line. "
        "Output only the narration text as plain prose."
    )
    raw = _ask([{"id": "script", "system": SCRIPT_SYSTEM, "user": user, "max_tokens": int(words * 2.2) + 120,
                 "temperature": 0.7}], workdir)["script"]
    raw = re.sub(r"^\s*(narration|voice-?over|script)\s*:\s*", "", raw.strip(), flags=re.I)
    raw = re.sub(r"[*_#`]+", "", raw).strip().strip('"“”')
    return re.sub(r"[ \t]+", " ", raw)


def write_dialogue(topic: str, lang: str, seconds: int, cast: list[dict], n_chars: int, narrator: bool,
                   workdir: Path) -> tuple[list[dict], list[dict]]:
    """Write a dialogue between n_chars characters (the given ones plus invented ones).
    Returns (characters, lines) with lines = [{"speaker": name|"narrator", "text": ...}]."""
    words = int(seconds * 2.3)
    lang_name = LANG_NAMES.get(lang, lang)
    known = "\n".join(f"- {c['name']}" + (f": {c['look']}" if c.get("look") else "") +
                      (f" (voice: {c['voice']})" if c.get("voice") else "") for c in cast)
    extra = max(n_chars - len(cast), 0)
    user = (
        f"Write the spoken lines of a {seconds}-second short film scene about:\n{topic}\n\n"
        f"Language of all spoken lines: {lang_name}. About {words} words in total (never more than "
        f"{int(words * 1.15)}). There are exactly {max(n_chars, len(cast))} characters.\n"
        + (f"These characters are given (keep their names):\n{known}\n" if cast else "")
        + (f"Invent {extra} more character(s) that fit the story, with names that fit the language and setting.\n"
           if extra else "")
        + ("A narrator may speak short off-screen lines (speaker \"Narrator\") to set the scene.\n" if narrator else
           "There is no narrator: everything is said by the characters.\n")
        + "Every character speaks at least once. Each line is one to three sentences. Alternate speakers naturally.\n"
        'Return JSON with exactly this shape: {"characters": [{"name": "...", "gender": "female or male", '
        '"age": "e.g. late twenties", "look": "visible appearance in English, 12 to 25 words: face, hair, build, '
        'clothing", "voice": "voice description in English, 8 to 20 words: pitch, timbre, pace, mood"}], '
        '"lines": [{"speaker": "<character name' + (' or Narrator' if narrator else '') + '>", "text": "..."}]}'
    )
    last_err = None
    for attempt in range(3):
        raw = _ask([{"id": "dialogue", "system": DIALOGUE_SYSTEM, "user": user,
                     "max_tokens": int(words * 2.6) + 160 * max(n_chars, 1) + 200,
                     "temperature": 0.75 - 0.15 * attempt}], workdir)["dialogue"]
        try:
            data = _extract_json(raw)
            chars = [C.clean_entry(c) for c in data.get("characters") or [] if isinstance(c, dict) and c.get("name")]
            names = {c["name"].lower() for c in chars} | {c["name"].lower() for c in cast}
            lines = []
            for ln in data.get("lines") or []:
                spk, text = str(ln.get("speaker") or "").strip(), " ".join(str(ln.get("text") or "").split())
                if not text:
                    continue
                if spk.lower() in C._NARRATOR_WORDS:
                    spk = C.NARRATOR
                elif spk.lower() not in names:
                    raise ValueError(f"unknown speaker {spk!r}")
                lines.append({"speaker": spk, "text": re.sub(r"[*_#`]+", "", text).strip('"“”„')})
            if len(lines) < 2:
                raise ValueError("fewer than two lines")
            # given characters first (their fields win), then invented ones up to the requested count
            merged = []
            for c in cast:
                got = next((x for x in chars if x["name"].lower() == c["name"].lower()), {})
                merged.append({**got, **{k: v for k, v in c.items() if v}, "generated": False})
            for c in chars:
                if len(merged) >= max(n_chars, len(cast)):
                    break
                if c["name"].lower() not in {m["name"].lower() for m in merged}:
                    merged.append({**c, "generated": True})
            keep = {m["name"].lower() for m in merged}
            lines = [ln for ln in lines if ln["speaker"] == C.NARRATOR or ln["speaker"].lower() in keep]
            if len(lines) < 2:
                raise ValueError("lines use characters that were dropped")
            return merged, lines
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            last_err = exc
            log(f"dialogue answer rejected ({exc}); retry {attempt + 1}/3")
    raise RuntimeError(f"planner failed to write the dialogue: {last_err}")


def storyboard(beats: list[str], topic: str, style: str | None, workdir: Path, refs: list[dict] | None = None,
               speakers: list[str] | None = None, cast: list[dict] | None = None) -> dict:
    """speakers: per beat the speaking character's name or "narrator" (dialogue projects)."""
    if speakers:
        tag = lambda i: "[Narrator, off screen]" if speakers[i] == C.NARRATOR else f"[{speakers[i]} speaks]"
        listing = "\n".join(f"{i + 1}. {tag(i)} {b}" for i, b in enumerate(beats))
    else:
        listing = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(beats))
    shot_shape = '{"n": 1, "visual": "...", "motion": "push_in"' + (', "refs": ["<id>"]' if refs else "") + "}"
    missing = [c for c in cast or [] if not c.get("look") or not c.get("voice")]
    cast_block = ""
    if cast:
        cast_block = ("Characters:\n" + "\n".join(f"- {c['name']}" + (f": {c['look']}" if c.get("look") else "")
                                                  for c in cast) + "\n"
                      "A line marked [Name speaks] is said on screen by that character: show that character "
                      "speaking, face clearly visible and turned towards the camera (frontal or three-quarter view), "
                      "medium close-up, mouth visible; name the character instead of describing their looks; other "
                      "characters may be visible behind or beside them. Lines marked [Narrator, off screen] are "
                      "illustrated freely. Keep the location consistent while a conversation continues.\n\n")
    characters_shape = (', "characters": [{"name": "...", "gender": "female or male", "age": "...", "look": '
                        '"visible appearance in English, 12 to 25 words", "voice": "voice description in English, '
                        '8 to 20 words"}]' if missing else "")
    user = (
        f"Video topic: {topic}\n\n"
        + cast_block
        + f"{'Lines' if speakers else 'Narration beats'}:\n{listing}\n\n"
        + (R.planner_block(refs) + 'For every shot, "refs" lists the ids of the photos visible in it '
           "(an empty list when none is).\n\n" if refs else "")
        + (f"Also describe these characters (look and voice): {', '.join(c['name'] for c in missing)}.\n" if missing else "")
        + f"Return JSON with exactly this shape:\n"
        '{"title": "short video title", "recurring_subject": "if ONE specific person, animal, object or place appears '
        'in most shots: its exact look in 10-18 words; otherwise an empty string", "style": "one line describing the look shared by ALL shots: medium (e.g. cinematic '
        'photography, 3D render, watercolor), colour palette, lighting quality, film look. No subjects, no shot sizes.", "shots": [' + shot_shape + ']'
        + characters_shape + '}\n'
        f"There must be exactly {len(beats)} shots, numbered 1 to {len(beats)}, one per {'line' if speakers else 'beat'}, in order. "
        + ("Consecutive lines of the same conversation may share the location, but vary the framing (close-up, "
           "over-the-shoulder, two-shot). " if speakers else
           "Every shot must be a clearly different image: change camera distance, angle, location, action or time of "
           "day. ")
        + "When the recurring subject is visible, name it with its look, then describe what is new in this shot. "
        f"motion is one of: {', '.join(MOTIONS)}. "
        + (f"The style must be: {style}. " if style else "")
    )
    last_err = None
    for attempt in range(3):
        raw = _ask([{"id": "board", "system": VISUAL_SYSTEM, "user": user,
                     "max_tokens": 220 * len(beats) + 300 + 120 * len(missing),
                     "temperature": 0.6 - 0.2 * attempt}], workdir)["board"]
        try:
            data = _extract_json(raw)
            shots = data["shots"]
            if len(shots) != len(beats) or not all(isinstance(s.get("visual"), str) and len(s["visual"]) > 15 for s in shots):
                raise ValueError(f"expected {len(beats)} shots with visuals, got {len(shots)}")
            import difflib
            limit = 0.92 if speakers else 0.8  # dialogue shots may legitimately look alike (same room, same people)
            for a in range(len(shots)):
                for b in range(a + 1, len(shots)):
                    if difflib.SequenceMatcher(None, shots[a]["visual"], shots[b]["visual"]).ratio() > limit:
                        raise ValueError(f"shots {a + 1} and {b + 1} describe the same image")
            return data
        except (ValueError, KeyError, TypeError) as exc:
            last_err = exc
            log(f"planner answer rejected ({exc}); retry {attempt + 1}/3")
    raise RuntimeError(f"planner failed to produce a valid storyboard: {last_err}")


def _motion_for(i: int, s: dict, shots: list[dict]) -> str:
    motion = s.get("motion") if s.get("motion") in MOTIONS else MOTIONS[i % len(MOTIONS)]
    if shots and shots[-1]["motion"] == motion:  # never repeat the same camera move twice in a row
        motion = next(m for m in MOTIONS[i % len(MOTIONS):] + MOTIONS if m != shots[-1]["motion"])
    return motion


def plan(*, topic: str | None, script: str | None, lang: str, seconds: int, style: str | None, workdir: Path,
         refs: list[dict] | None = None, cast: list[dict] | None = None, characters: int = 0,
         narrator: bool = False) -> dict:
    """Return a storyboard dict: {title, language, style, shots:[{id, narration, visual, motion, refs?}]}.

    Dialogue projects (a script with "Name: line" lines, or characters > 0, or a given cast) also get
    {cast: [...]} and per shot {speaker, on_screen}."""
    cast = [C.clean_entry(c) for c in cast or []]
    rows = C.parse_dialogue(script) if script else None
    dialogue = bool(rows) or (not script and (characters > 0 or cast))
    if dialogue and not rows:
        n = max(characters, len(cast), 1)
        log(f"writing a {seconds}s dialogue for {n} character(s) with {config.MODELS['planner']}")
        cast, rows = write_dialogue(topic, lang, seconds, cast, n, narrator, workdir)
    if dialogue:
        names = {c["name"].lower() for c in cast}
        for r in rows:  # script names that were not defined become characters (described by the storyboard step)
            if r["speaker"] != C.NARRATOR and r["speaker"].lower() not in names:
                cast.append(C.clean_entry({"name": r["speaker"], "generated": True}))
                names.add(r["speaker"].lower())
        taken: set = set()
        for c in cast:
            c["id"] = C.make_id(c.get("id") or c["name"], taken)
        beats, speakers = [], []
        for r in rows:
            for piece in T.segment_script(r["text"]) or [r["text"]]:
                beats.append(piece)
                speakers.append(r["speaker"])
    else:
        if not script:
            log(f"writing {seconds}s narration script with {config.MODELS['planner']}")
            script = write_script(topic, lang, seconds, workdir)
        beats, speakers = T.segment_script(script), None
    if not beats:
        raise SystemExit("script is empty")
    log(f"storyboarding {len(beats)} shots" + (f" with {len(cast)} character(s)" if dialogue else ""))
    board = storyboard(beats, topic or (script or "")[:200], style, workdir, refs or None, speakers,
                       cast if dialogue else None)
    if dialogue:  # fill in looks/voices the storyboard step described
        described = {str(c.get("name", "")).lower(): C.clean_entry(c) for c in board.get("characters") or []
                     if isinstance(c, dict) and c.get("name")}
        for c in cast:
            d = described.get(c["name"].lower(), {})
            for k in ("gender", "age", "look", "voice"):
                if not c.get(k) and d.get(k):
                    c[k] = d[k]
    shots = []
    for i, (beat, s) in enumerate(zip(beats, board["shots"])):
        shot = {"id": f"s{i + 1:02d}", "narration": beat, "visual": literal(" ".join(s["visual"].split())),
                "motion": _motion_for(i, s, shots)}
        if refs:
            shot["refs"] = [r for r in s.get("refs") or [] if isinstance(r, str)] if isinstance(s.get("refs"), list) else []
        if dialogue:
            shot["speaker"] = C.speaker_id(speakers[i], cast)
            shot["on_screen"] = shot["speaker"] != C.NARRATOR
        shots.append(shot)
    recurring = board.get("recurring_subject")
    if refs or dialogue:  # photos / character portraits define the recurring subjects
        recurring = ""
    out = {
        "title": str(board.get("title") or (topic or "Untitled"))[:80],
        "language": lang,
        "recurring_subject": recurring.strip() if isinstance(recurring, str) and len(recurring.split()) >= 4 else "",
        "style": style or str(board.get("style") or config.DEFAULT_STYLE),
        "shots": shots,
    }
    if dialogue:
        out["cast"] = cast
    return out
