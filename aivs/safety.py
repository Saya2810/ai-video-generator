"""Guard for projects that use photos of real people.

Uploaded person photos (not generated portraits) put a real person's face into the video, and with lip-sync they
say words they never said. Sexual or nude scenes with such photos are not generated; the check runs when a project
is created and before every render, over the idea/script, all lines, visual descriptions, style and character looks.
"""
from __future__ import annotations

import re

_SEXUAL = re.compile(
    r"(?<![\w])(sex\w*|porn\w*|nackt\w*|nude\w*|naked|nudity|blowjob\w*|handjob\w*|orgasm\w*|erotik\w*|erotisch\w*|"
    r"erotic\w*|fick\w*|bumsen\w*|masturb\w*|penis\w*|vagina\w*|brüste\w*|boobs|tits|striptease\w*|strip-?tease|"
    r"undress\w*|entkleid\w*|lingerie|dessous|bdsm|sperma|cumshot\w*|oben-?ohne|topless|intimszene\w*|"
    r"geschlechtsverkehr\w*|intercourse)(?![\w])", re.IGNORECASE)

MESSAGE = ("This project uses uploaded photos of real people. Sexual or nude scenes with real people's photos are not "
           "generated. Remove the photo(s) or change the content.")


def real_person_refs(board: dict) -> list[dict]:
    return [r for r in board.get("references") or [] if r.get("kind") == "person" and not r.get("generated")]


def texts(board: dict, extra: list[str] | None = None) -> list[str]:
    out = list(extra or [])
    out += [board.get("title") or "", board.get("style") or ""]
    for s in board.get("shots") or []:
        out += [s.get("narration") or "", s.get("visual") or ""]
    for c in board.get("cast") or []:
        out += [c.get("look") or ""]
    return out


def find_sexual(strings: list[str]) -> str | None:
    for t in strings:
        m = _SEXUAL.search(t or "")
        if m:
            return m.group(0)
    return None


def check(board: dict, extra: list[str] | None = None, has_photos: bool | None = None) -> None:
    """Raise SystemExit when real-person photos and sexual content come together."""
    photos = real_person_refs(board) if has_photos is None else has_photos
    if not photos:
        return
    hit = find_sexual(texts(board, extra))
    if hit:
        raise SystemExit(f"{MESSAGE} (found: '{hit}')")
