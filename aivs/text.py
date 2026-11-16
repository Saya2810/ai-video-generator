"""Deterministic text handling: sentence splitting, shot segmentation, caption chunking, word alignment."""
from __future__ import annotations

import difflib
import re
import unicodedata

_ABBR = {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "z.b", "bzw", "ca", "nr", "u.a", "d.h", "no"}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts, start = [], 0
    for m in re.finditer(r"[.!?…]+[\"'”’»)]*\s+", text):
        prev = text[start:m.start()].split()
        last = prev[-1].lower().rstrip(".") if prev else ""
        if last in _ABBR or re.fullmatch(r"\d+", last or "x") and text[m.start()] == "." and text[m.end():m.end() + 1].islower():
            continue
        parts.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def _split_long(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    # prefer a clause boundary near the middle
    best, mid = None, len(words) / 2
    for i, w in enumerate(words[:-1]):
        if w[-1] in ",;:–—" and 4 <= i + 1 <= len(words) - 4:
            if best is None or abs(i + 1 - mid) < abs(best - mid):
                best = i + 1
    if best is None:
        best = int(mid)
    left, right = " ".join(words[:best]), " ".join(words[best:])
    return _split_long(left, max_words) + _split_long(right, max_words)


def segment_script(text: str, min_words: int = 7, max_words: int = 26) -> list[str]:
    """Split narration into shot-sized beats (~3-9 s each) without changing a single word."""
    beats: list[str] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    for para in paragraphs:
        cur: list[str] = []
        for sent in split_sentences(para):
            for piece in _split_long(sent, max_words):
                n_cur = sum(len(s.split()) for s in cur)
                n_new = len(piece.split())
                if cur and (n_cur >= min_words and n_cur + n_new > max_words or n_cur >= min_words and n_new >= min_words):
                    beats.append(" ".join(cur))
                    cur = []
                cur.append(piece)
        if cur:
            if beats and sum(len(s.split()) for s in cur) < min_words - 3 and len(beats[-1].split()) < max_words:
                beats[-1] = beats[-1] + " " + " ".join(cur)
            else:
                beats.append(" ".join(cur))
    return beats


def norm_token(tok: str) -> str:
    tok = unicodedata.normalize("NFKC", tok).lower()
    return re.sub(r"[^\w]+", "", tok, flags=re.UNICODE)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = [t for t in map(norm_token, reference.split()) if t]
    hyp = [t for t in map(norm_token, hypothesis.split()) if t]
    if not ref:
        return 0.0
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return 1.0 - matched / max(len(ref), len(hyp), 1)


def align_words(script: str, heard: list[dict], duration: float) -> list[dict]:
    """Give every word of the *original script* a start/end time using ASR word timings of the spoken audio.

    Words the recogniser spelled differently are interpolated between their matched neighbours, so captions
    always show the author's text, never ASR guesses.
    """
    words = script.split()
    ref = [norm_token(w) for w in words]
    hyp = [norm_token(h["word"]) for h in heard]
    times: list[tuple[float, float] | None] = [None] * len(words)
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            if ref[a + k]:
                times[a + k] = (float(heard[b + k]["start"]), float(heard[b + k]["end"]))
    # interpolate gaps by character weight
    i, n = 0, len(words)
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        t0 = times[i - 1][1] if i > 0 else 0.0
        t1 = times[j][0] if j < n else duration
        t1 = max(t1, t0)
        weights = [max(len(w), 1) for w in words[i:j]]
        total, acc = sum(weights), 0.0
        for k, wgt in enumerate(weights):
            s = t0 + (t1 - t0) * acc / total
            acc += wgt
            e = t0 + (t1 - t0) * acc / total
            times[i + k] = (s, e)
        i = j
    out, last = [], 0.0
    for w, (s, e) in zip(words, times):
        s = min(max(s, last), duration)
        e = min(max(e, s), duration)
        out.append({"word": w, "start": round(s, 3), "end": round(e, 3)})
        last = s
    return out


_FUNC = set("""a an the of in on at to for and or but nor with by from as is are was were be been that which who whose
this these those its their his her our your my into onto than then so if not no
der die das den dem des ein eine einen einem einer eines und oder aber im am auf aus bei mit nach von vor zu zum zur
für über unter dass ist sind war nicht als wie sich
le la les un une des du de et ou à au aux en dans sur pour par avec ce cette ces qui que est sont ne se
el los las una y o del con por para al su sus lo es son se
il lo gli i uno e di da con per tra fra che è sono
o os as um uma do da dos das em no na com
een het van op te voor met en""".split())


def _split_balanced(seg: list[dict], max_chars: int) -> list[list[dict]]:
    """Break one clause into lines of similar length; prefer breaks after commas, never after function words."""
    lens = [len(w["word"]) for w in seg]
    total = sum(lens) + len(seg) - 1
    if total <= max_chars or len(seg) == 1:
        return [seg]
    n_lines = -(-total // max_chars)
    target = total / n_lines
    m = len(seg)
    INF = float("inf")
    best, back = [0.0] + [INF] * m, [0] * (m + 1)
    for j in range(1, m + 1):
        for i in range(j - 1, -1, -1):
            length = sum(lens[i:j]) + (j - i - 1)
            if length > max_chars and j - i > 1:
                break
            cost = 0.35 * (length - target) ** 2
            if j < m:
                last = seg[j - 1]["word"]
                if last[-1:] in ",;:" or last.endswith(("—", "–")):
                    cost -= 90
                if norm_token(last) in _FUNC:
                    cost += 220
                if j == m - 1 and lens[m - 1] < 9:
                    cost += 120  # do not strand a short last word on its own line
            if best[i] + cost < best[j]:
                best[j], back[j] = best[i] + cost, i
    lines, j = [], m
    while j > 0:
        lines.append(seg[back[j]:j])
        j = back[j]
    return lines[::-1]


def caption_chunks(words: list[dict], max_chars: int = 42, max_gap: float = 0.6) -> list[dict]:
    """Group timed words into readable caption lines: clauses first (sentence ends, pauses), then balanced lines."""
    segments, cur = [], []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > max_gap:
            segments.append(cur)
            cur = []
        cur.append(w)
        if w["word"][-1:] in ".!?…" or w["word"].endswith((".\"", "!\"", "?\"", ".”", "!”", "?”")):
            segments.append(cur)
            cur = []
    if cur:
        segments.append(cur)
    chunks = []
    for seg in segments:
        for line in _split_balanced(seg, max_chars):
            chunks.append({"text": " ".join(w["word"] for w in line), "start": line[0]["start"], "end": line[-1]["end"]})
    return chunks
