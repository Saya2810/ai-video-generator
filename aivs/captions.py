"""Caption files (SRT sidecar + styled ASS for burn-in) from timed caption chunks."""
from __future__ import annotations

from pathlib import Path


def _srt_time(t: float) -> str:
    ms = int(round(max(t, 0) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _ass_time(t: float) -> str:
    cs = int(round(max(t, 0) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def tidy(chunks: list[dict], min_dur: float = 0.9, gap: float = 0.04) -> list[dict]:
    """Enforce readable minimum durations without overlapping the next caption."""
    out = [dict(c) for c in sorted(chunks, key=lambda c: c["start"])]
    for i, c in enumerate(out):
        limit = out[i + 1]["start"] - gap if i + 1 < len(out) else c["end"] + 1.0
        c["end"] = max(c["end"], min(c["start"] + min_dur, limit))
        c["end"] = min(c["end"] + 0.12, limit) if c["end"] + 0.12 <= limit else c["end"]
    return out


def write_srt(chunks: list[dict], path: Path) -> None:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines += [str(i), f"{_srt_time(c['start'])} --> {_srt_time(c['end'])}", c["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ass(chunks: list[dict], path: Path, size: tuple[int, int]) -> None:
    W, H = size
    portrait = H > W
    font_size = round(H * (0.036 if portrait else 0.052))
    margin_v = round(H * (0.16 if portrait else 0.075))
    margin_h = round(W * 0.08)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Source Sans 3 Semibold,{font_size},&H00FFFFFF,&H00FFFFFF,&HC0101010,&H80000000,0,0,0,0,100,100,0.3,0,1,{max(font_size * 0.07, 2):.1f},{max(font_size * 0.04, 1):.1f},2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for c in chunks:
        text = c["text"].replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")
        colour = f"\\c{c['colour']}" if c.get("colour") else ""  # one colour per speaker in dialogues
        events.append(f"Dialogue: 0,{_ass_time(c['start'])},{_ass_time(c['end'])},Cap,,0,0,0,,{{\\fad(90,90){colour}}}{text}")
    path.write_text(head + "\n".join(events) + "\n", encoding="utf-8")
