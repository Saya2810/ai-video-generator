"""Characters (cast), dialogue scripts and per-speaker voices.

A project with characters keeps them in `storyboard["cast"]`:

    {"id": "anna", "name": "Anna", "gender": "female", "age": "thirties",
     "look": "woman with short dark hair, round glasses, red coat",          # appearance (English)
     "voice": "warm, calm female voice, medium pitch, slightly husky",       # voice description (English)
     "ref": "anna",              # id of the reference photo (uploaded or generated portrait)
     "voice_sample": null,       # optional: path (relative to the project) of a voice sample to clone
     "kokoro_voice": null,       # optional: English preset voice (assigned automatically for English)
     "generated": true}          # true when the planner invented this character

and every shot says who speaks: `shot["speaker"] = "anna"` (or "narrator") and whether that person is visible
and speaking on screen (`shot["on_screen"]`, used for lip-sync).

Voices:
- English: each character gets its own Kokoro preset voice (by gender), unless a voice sample is given.
- Other languages: a ~10 s voice sample is designed once per character from the voice description with
  Qwen3-TTS VoiceDesign (Apache-2.0); Chatterbox Multilingual then speaks every line of that character in this
  voice. Languages VoiceDesign does not cover are designed from English text (cross-lingual cloning).
"""
from __future__ import annotations

import re

from .util import slugify

NARRATOR = "narrator"

# Kokoro presets used for English characters, in order of preference (narrator default is af_heart).
KOKORO_FEMALE = ["af_bella", "bf_emma", "af_nicole", "af_sarah", "bf_isabella", "af_sky", "bf_alice", "af_heart"]
KOKORO_MALE = ["am_michael", "bm_george", "am_fenrir", "am_adam", "bm_lewis", "am_puck", "bm_daniel", "am_eric"]

# Languages Qwen3-TTS VoiceDesign supports (its codec language names)
VOICE_DESIGN_LANGS = {"zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean", "de": "german",
                      "fr": "french", "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian"}

# Neutral passages (~10 s) spoken in the designed voice; they become the voice sample that Chatterbox clones.
SAMPLE_TEXT = {
    "en": "Good morning. I hope you slept well, because today is going to be a long day. Let me tell you what "
          "happened yesterday, and then we can decide together what to do next.",
    "de": "Guten Morgen. Ich hoffe, du hast gut geschlafen, denn heute wird ein langer Tag. Lass mich erzählen, "
          "was gestern passiert ist, und dann entscheiden wir gemeinsam, wie es weitergeht.",
    "fr": "Bonjour. J'espère que tu as bien dormi, parce que la journée va être longue. Laisse-moi te raconter ce "
          "qui s'est passé hier, et ensuite nous déciderons ensemble de la suite.",
    "es": "Buenos días. Espero que hayas dormido bien, porque hoy va a ser un día largo. Déjame contarte lo que "
          "pasó ayer, y luego decidiremos juntos qué hacer.",
    "it": "Buongiorno. Spero che tu abbia dormito bene, perché oggi sarà una lunga giornata. Lascia che ti racconti "
          "cosa è successo ieri, e poi decideremo insieme cosa fare.",
    "pt": "Bom dia. Espero que você tenha dormido bem, porque hoje vai ser um dia longo. Deixa eu te contar o que "
          "aconteceu ontem, e depois decidimos juntos o que fazer.",
    "ru": "Доброе утро. Надеюсь, ты хорошо выспался, потому что сегодня будет длинный день. Давай я расскажу, что "
          "случилось вчера, а потом мы вместе решим, что делать дальше.",
    "ja": "おはようございます。よく眠れましたか。今日は長い一日になりそうです。昨日あったことを話してから、"
          "これからどうするか一緒に決めましょう。",
    "ko": "좋은 아침이에요. 잘 주무셨길 바라요. 오늘은 긴 하루가 될 거예요. 어제 무슨 일이 있었는지 말씀드리고, "
          "그다음에 함께 무엇을 할지 정해요.",
    "zh": "早上好。希望你昨晚睡得很好，因为今天会是漫长的一天。让我先告诉你昨天发生了什么，然后我们一起决定接下来怎么做。",
}

# Caption colours per speaker (ASS &HBBGGRR&), light tones that read on white-outlined text; narrator stays white.
CAPTION_COLOURS = ["&H9BE3FF&", "&HFFD49B&", "&HB0FFB0&", "&HD9B3FF&", "&H9BC8FF&", "&HFFB3E6&", "&HE6FF9B&"]

_LINE = re.compile(r"^\s*(?:[-–—•*]\s*)?([^\W\d_][\w .'’\-]{0,30}?)\s*(?:\([^)]{0,40}\))?\s*:\s+(\S.*)$", re.UNICODE)
_NARRATOR_WORDS = {"narrator", "erzähler", "erzählerin", "sprecher", "sprecherin", "off", "voice-over", "voiceover",
                   "narrateur", "narrador", "narratore"}


# ---------------------------------------------------------------- script parsing
def parse_dialogue(script: str) -> list[dict] | None:
    """'Anna: Hallo!' lines -> [{"speaker": "Anna", "text": "Hallo!"}]. Lines without a name prefix belong to the
    narrator. Returns None when the text is not a dialogue (fewer than two prefixed lines)."""
    rows, named = [], 0
    for raw in script.splitlines():
        if not raw.strip():
            continue
        m = _LINE.match(raw)
        if m and len(m.group(1).split()) <= 3:
            name = " ".join(m.group(1).split())
            speaker = NARRATOR if name.lower() in _NARRATOR_WORDS else name
            named += speaker != NARRATOR
            rows.append({"speaker": speaker, "text": m.group(2).strip()})
        elif rows and rows[-1]["speaker"] == NARRATOR:
            rows[-1]["text"] += " " + raw.strip()
        else:
            rows.append({"speaker": NARRATOR, "text": raw.strip()})
    return rows if named >= 2 else None


# ---------------------------------------------------------------- cast records
def make_id(name: str, taken: set) -> str:
    base = slugify(name, maxlen=20) or "character"
    if base == NARRATOR:
        base = "narrator-character"
    rid, n = base, 2
    while rid in taken:
        rid, n = f"{base}-{n}", n + 1
    taken.add(rid)
    return rid


def clean_entry(c: dict) -> dict:
    out = {"name": " ".join(str(c.get("name") or "").split())[:40]}
    for k, n in (("gender", 20), ("age", 30), ("look", 300), ("voice", 300)):
        out[k] = " ".join(str(c.get(k) or "").split())[:n]
    g = out["gender"].lower()
    out["gender"] = ("female" if g.startswith(("f", "w", "girl", "woman")) else
                     "male" if g.startswith(("m", "boy", "man")) else "")
    for k in ("ref", "voice_sample", "kokoro_voice", "id"):
        if c.get(k):
            out[k] = str(c[k])
    out["generated"] = bool(c.get("generated", False))
    return out


def by_name(cast: list[dict]) -> dict:
    return {c["name"].lower(): c for c in cast}


def speaker_id(name: str, cast: list[dict]) -> str:
    if not name or name.lower() in _NARRATOR_WORDS or name == NARRATOR:
        return NARRATOR
    c = by_name(cast).get(name.lower())
    return c["id"] if c else NARRATOR


def assign_kokoro(cast: list[dict]) -> None:
    """Distinct English preset voices per character (stable: characters keep a voice once assigned)."""
    used = {c["kokoro_voice"] for c in cast if c.get("kokoro_voice")}
    for c in cast:
        if c.get("kokoro_voice") or c.get("voice_sample"):
            continue
        pool = KOKORO_MALE if c.get("gender") == "male" else KOKORO_FEMALE if c.get("gender") == "female" else \
            [v for pair in zip(KOKORO_FEMALE, KOKORO_MALE) for v in pair]
        free = [v for v in pool if v not in used] or pool
        c["kokoro_voice"] = free[0]
        used.add(free[0])


def colour_for(speaker: str, cast: list[dict]) -> str | None:
    if speaker == NARRATOR:
        return None
    ids = [c["id"] for c in cast]
    return CAPTION_COLOURS[ids.index(speaker) % len(CAPTION_COLOURS)] if speaker in ids else None


def portrait_prompt(c: dict, style: str) -> str:
    who = c.get("look") or f"a {c.get('age') or 'adult'} {c.get('gender') or 'person'}"
    return (f"Head-and-shoulders portrait photo of {who}, looking straight into the camera, neutral relaxed "
            f"expression with closed mouth, even soft light, plain light grey background, sharp focus on the face. "
            f"{style.rstrip('. ')}. No text, no watermark.")


def voice_instruction(c: dict) -> str:
    v = c.get("voice") or ""
    if not v:
        g = {"female": "female", "male": "male"}.get(c.get("gender"), "")
        v = f"A natural {c.get('age') or 'adult'} {g} voice, clear and warm, moderate pace".replace("  ", " ")
    return v
