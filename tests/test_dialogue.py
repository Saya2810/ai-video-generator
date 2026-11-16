"""Run: envs/studio/bin/python tests/test_dialogue.py   (no models needed, ~10 s)

Offline tests for characters, dialogue, per-speaker voices, lip-sync stage and speaker captions.
All models (LLM, FLUX, TTS, voice design, Whisper, LatentSync, MediaPipe) are replaced by fakes."""
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="aivs-dlg-")).resolve()  # macOS: /var -> /private/var
os.environ["AIVS_PROJECTS"] = str(TMP / "projects")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from PIL import Image  # noqa: E402

from aivs import cast as C, captions, config, images, pipeline, planner, references as R  # noqa: E402
from aivs.util import digest, read_json  # noqa: E402

PASSED = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    PASSED.append(msg)


# ---------------------------------------------------------------- 1. script parsing
rows = C.parse_dialogue("""Erzähler: Ein kleines Café in Berlin.
Anna: Hast du das gehört?
Tom (leise): Ja. Und ich glaube es nicht.
Das ist eine Fortsetzung ohne Namen.
- Anna: Dann lass uns gehen.""")
check([r["speaker"] for r in rows] == ["narrator", "Anna", "Tom", "narrator", "Anna"], f"dialogue parsed: {[r['speaker'] for r in rows]}")
check(rows[2]["text"] == "Ja. Und ich glaube es nicht.", "stage direction in brackets dropped")
check(C.parse_dialogue("Es war einmal: ein Leuchtturm.\nEr stand am Meer.") is None, "plain narration is not a dialogue")
check(C.parse_dialogue("Es war einmal ein Café.\nAnna: Hallo.") is None, "needs two named lines")
check(C.parse_dialogue("Anna: Hallo.\nOff: Später.\nTom: Hi.")[1]["speaker"] == "narrator", "'Off' counts as narrator")

cast = [C.clean_entry({"name": "Anna", "gender": "weiblich"}), C.clean_entry({"name": "Tom", "gender": "Mann"}),
        C.clean_entry({"name": "Kim"}), C.clean_entry({"name": "Lea", "gender": "female", "kokoro_voice": "af_bella"})]
check([c["gender"] for c in cast] == ["female", "male", "", "female"], "gender normalised (German words)")
taken = set()
for c in cast:
    c["id"] = C.make_id(c["name"], taken)
C.assign_kokoro(cast)
voices = [c["kokoro_voice"] for c in cast]
check(len(set(voices)) == 4 and voices[1] in C.KOKORO_MALE and voices[0] in C.KOKORO_FEMALE and voices[3] == "af_bella",
      f"distinct English voices by gender, kept when set: {voices}")
check(C.make_id("Narrator", set()) != "narrator", "a character can't take the narrator id")
check(C.colour_for("narrator", cast) is None and C.colour_for(cast[1]["id"], cast) == C.CAPTION_COLOURS[1], "caption colours")

# ---------------------------------------------------------------- 2. planner: dialogue from an idea
asked = []


def fake_ask(reqs, wd):
    r = reqs[0]
    asked.append(r)
    if r["id"] == "dialogue":
        if len([a for a in asked if a["id"] == "dialogue"]) == 1:
            return {"dialogue": '{"characters": [], "lines": [{"speaker": "Ghost", "text": "Boo"}, {"speaker": "Anna", "text": "x"}]}'}
        return {"dialogue": json.dumps({"characters": [
            {"name": "Anna", "gender": "female", "age": "30s", "look": "LLM look for Anna", "voice": "LLM voice Anna"},
            {"name": "Tom", "gender": "male", "age": "40s", "look": "bald man with grey beard, denim jacket", "voice": "deep slow voice"},
            {"name": "Extra", "gender": "male", "look": "should be dropped", "voice": "x"}],
            "lines": [{"speaker": "Narrator", "text": "Ein Café am Morgen."}, {"speaker": "Anna", "text": "Hast du **das** gehört?"},
                      {"speaker": "Tom", "text": "Ja, ich habe es gehört."}, {"speaker": "Extra", "text": "Ich auch."},
                      {"speaker": "Anna", "text": "Dann lass uns gehen."}]})}
    n = int(re.search(r"There must be exactly (\d+) shots", r["user"]).group(1))
    pool = ["Anna sits at a window table holding a coffee cup, morning light", "Close-up of Tom leaning back in a leather armchair",
            "Rain streaks down the cafe window, street lamps glowing outside", "Over-the-shoulder view of a barista grinding beans",
            "Wide shot of the empty street with parked bicycles at dusk", "Macro of steam rising from a porcelain espresso cup"]
    shots = [{"n": i + 1, "visual": pool[i % len(pool)] + (f" variant {i}" if i >= len(pool) else ""), "motion": "push_in"} for i in range(n)]
    out = {"title": "Café", "recurring_subject": "something long enough here", "style": "warm film", "shots": shots}
    if "Also describe these characters" in r["user"]:
        out["characters"] = [{"name": "Tom", "gender": "male", "look": "described Tom", "voice": "described voice"}]
    return {"board": json.dumps(out)}


planner._ask = fake_ask
given = [{"name": "Anna", "look": "woman with red curls, green coat", "voice": ""}]
chars, lines = planner.write_dialogue("Zwei Freunde im Café", "de", 30, [C.clean_entry(g) for g in given], 2, True, TMP)
check([c["name"] for c in chars] == ["Anna", "Tom"], f"given character first, invented up to the count: {[c['name'] for c in chars]}")
check(chars[0]["look"] == "woman with red curls, green coat" and chars[0]["voice"] == "LLM voice Anna",
      "given fields win, missing fields filled by the LLM")
check(not chars[0]["generated"] and chars[1]["generated"], "invented characters are marked")
check([l["speaker"] for l in lines] == ["narrator", "Anna", "Tom", "Anna"], f"lines of dropped characters removed: {lines}")
check(lines[1]["text"] == "Hast du das gehört?", "markdown removed from lines")
check("Invent 1 more character" in asked[-1]["user"] and "German" in asked[-1]["user"], "prompt asks for the missing character in German")
check(len([a for a in asked if a["id"] == "dialogue"]) == 2, "unknown speaker -> retry")

asked.clear()
b = planner.plan(topic=None, script="Anna: Hast du das gehört?\nTom: Ja, laut und deutlich.\nAnna: Dann komm.",
                 lang="de", seconds=20, style=None, workdir=TMP)
check([c["name"] for c in b["cast"]] == ["Anna", "Tom"] and [c["id"] for c in b["cast"]] == ["anna", "tom"], "cast from script names")
check([s["speaker"] for s in b["shots"]] == ["anna", "tom", "anna"] and all(s["on_screen"] for s in b["shots"]), "speaker per shot")
check([s["narration"] for s in b["shots"]][1] == "Ja, laut und deutlich.", "dialogue text kept word for word")
check("[Anna speaks]" in asked[-1]["user"] and "Also describe these characters" in asked[-1]["user"], "storyboard knows who speaks")
check(next(c for c in b["cast"] if c["name"] == "Tom")["look"] == "described Tom", "storyboard step described the script characters")
check(b["recurring_subject"] == "", "no text recurring subject in dialogues")
asked.clear()
plain = planner.plan(topic=None, script="Der Leuchtturm steht seit hundert Jahren an der Küste und trotzt jedem Sturm.",
                     lang="de", seconds=20, style=None, workdir=TMP)
check("cast" not in plain and "speaker" not in plain["shots"][0] and "[" not in asked[-1]["user"].split("Narration beats:")[1][:5],
      "narrated videos unchanged")

# ---------------------------------------------------------------- 3. create_project + stage_cast
photo = TMP / "anna.jpg"
Image.new("RGB", (900, 1200), (180, 60, 60)).save(photo)
gen_calls = []


def fake_generate(prompt, out, size, seed, steps, log_path, reference=None):
    gen_calls.append({"prompt": prompt, "size": size, "refs": [Path(r).name for r in (reference if isinstance(reference, list) else [reference] if reference else [])]})
    Image.new("RGB", size, (90, 90, 200)).save(out)


images.generate = fake_generate
workers = []


def fake_worker(name, job, wd):
    workers.append((name, job))
    if name == "voice_design":
        for it in job["items"]:
            sf.write(it["out"], np.zeros(48000 * 8, np.float32), 48000)
        return {"items": []}
    if name == "tts":
        res = []
        for it in job["items"]:
            n = 48000 * (1 + len(it["text"]) // 30)
            sf.write(it["out"], (0.1 * np.sin(np.arange(n) / 9)).astype(np.float32), 48000)
            res.append({"id": it["id"], "out": it["out"], "duration": n / 48000})
        return {"items": res}
    if name == "stt":
        out = []
        for it in job["items"]:
            sid = it["id"]
            text = next(s["narration"] for s in CURRENT["shots"] if s["id"] == sid)
            words = [{"word": w, "start": 0.2 * i, "end": 0.2 * i + 0.18} for i, w in enumerate(text.split())]
            out.append({"id": sid, "text": text, "words": words})
        return {"items": out}
    if name == "depth":
        for it in job["items"]:
            Path(it["out"]).write_bytes(b"d")
        return {"items": []}
    raise AssertionError(name)


pipeline.run_worker = fake_worker
from aivs import fetch  # noqa: E402
fetched = []
fetch.fetch = lambda repo, *a, **k: fetched.append(repo)

pdir = pipeline.create_project(script="Anna: Hast du das gehört?\nTom: Ja, ich habe es gehört.\nErzähler: Draußen regnet es.\nAnna: Dann bleiben wir.",
                               lang="de", name="cafe", refs=[{"kind": "person", "name": "Anna", "path": str(photo)}])
proj = read_json(pdir / "project.json")
bd = proj["storyboard"]
check([c["name"] for c in bd["cast"]] == ["Anna", "Tom"], "cast in project")
check([s["speaker"] for s in bd["shots"]] == ["anna", "tom", "narrator", "anna"], "narrator line in a dialogue")
check(bd["shots"][2]["on_screen"] is False, "narrator is never on screen")
settings = pipeline.default_settings() | proj["settings"]
CURRENT = bd
pipeline.stage_cast(pdir, bd, settings)
saved = read_json(pdir / "project.json")["storyboard"]
anna, tom = saved["cast"]
check(anna["ref"] == "anna" and not next(r for r in saved["references"] if r["id"] == "anna").get("generated"),
      "uploaded photo linked to the character with that name (no portrait generated)")
tom_ref = next(r for r in saved["references"] if r["id"] == tom["ref"])
check(tom_ref["generated"] and (pdir / tom_ref["image"]).exists() and len(gen_calls) == 1 and gen_calls[0]["size"] == (1024, 1024),
      "portrait generated for the invented character")
check("described Tom" in gen_calls[0]["prompt"] and "looking straight into the camera" in gen_calls[0]["prompt"], "portrait prompt uses the look")
vd = [j for n, j in workers if n == "voice_design"]
check(len(vd) == 1 and len(vd[0]["items"]) == 2 and vd[0]["items"][0]["language"] == "german"
      and vd[0]["items"][0]["text"].startswith("Guten Morgen"), "one German voice sample designed per character")
_vd_local = fetch.resolve_model(config.MODELS["voice_design"]) != config.MODELS["voice_design"]  # already on disk
check(fetched == ([] if _vd_local else [config.MODELS["voice_design"]]),
      "voice design model fetched with the resumable downloader (only when not on disk yet)")
check(all((pdir / c["voice_design_sample"]).exists() for c in saved["cast"]), "voice samples stored in the project")
workers.clear(); gen_calls.clear()
CURRENT = bd = saved
pipeline.stage_cast(pdir, bd, settings)
check(not workers and not gen_calls, "second run: portraits and voices cached")
bd["cast"][1]["look"] = "young man with a red cap"
pipeline.stage_cast(pdir, bd, settings)
check(len(gen_calls) == 1 and "red cap" in gen_calls[0]["prompt"] and not workers, "changed look -> new portrait only")
bd["cast"][0]["voice"] = "very high cheerful voice"
workers.clear()
pipeline.stage_cast(pdir, bd, settings)
check(len(workers) == 1 and [i["id"] for i in workers[0][1]["items"]] == ["anna"], "changed voice -> only that voice redesigned")

# ---------------------------------------------------------------- 4. narration per speaker
bd = read_json(pdir / "project.json")["storyboard"]
CURRENT = bd
workers.clear()
pipeline.stage_narration(pdir, bd, settings)
tts_jobs = [j for n, j in workers if n == "tts"]
check(len(tts_jobs) == 1 and len(tts_jobs[0]["items"]) == 4, "all German lines in one Chatterbox run (one model load)")
items = {i["id"]: i for i in tts_jobs[0]["items"]}
check(items["s01"]["ref_audio"].endswith(bd["cast"][0]["voice_design_sample"].split("/")[-1])
      and items["s02"]["ref_audio"] != items["s01"]["ref_audio"], "each character speaks with its own voice sample")
check(items["s03"].get("ref_audio") is None and items["s03"]["cfg_weight"] == 0.0, "narrator keeps the default voice settings")
check(items["s01"]["cfg_weight"] == 0.5, "same-language sample -> normal cfg (no accent carry-over)")
stt_jobs = [j for n, j in workers if n == "stt"]
check(len(stt_jobs) == 1 and len(stt_jobs[0]["items"]) == 4, "one Whisper check for all lines")
# narrated project keys unchanged
lt = json.loads((ROOT / "examples" / "leuchtturm-de.json").read_text())
st = pipeline.default_settings()
setup = pipeline.tts_setup(lt["language"], st)
old_key = digest([lt["shots"][0]["narration"], {k: v for k, v in setup.items()} | {"seed": st["seed"]}])
check(digest([lt["shots"][0]["narration"], dict(pipeline.speaker_setup(TMP, lt, lt["shots"][0], st)) | {"seed": st["seed"]}]) == old_key,
      "narration cache keys of existing projects unchanged")
# English dialogue -> Kokoro preset per character
en = {"language": "en", "style": "s", "cast": [C.clean_entry({"name": "Ann", "gender": "female", "id": "ann"}),
                                               C.clean_entry({"name": "Bob", "gender": "male", "id": "bob"})]}
for c in en["cast"]:
    c["id"] = c["name"].lower()
C.assign_kokoro(en["cast"])
s_ann = pipeline.speaker_setup(TMP, en, {"speaker": "ann"}, st)
s_bob = pipeline.speaker_setup(TMP, en, {"speaker": "bob"}, st)
s_nar = pipeline.speaker_setup(TMP, en, {"speaker": "narrator"}, st)
check(s_ann["engine"] == s_bob["engine"] == "kokoro" and s_ann["voice"] != s_bob["voice"] and s_nar["voice"] == "af_heart",
      f"English: Kokoro voices {s_ann['voice']}/{s_bob['voice']}, narrator af_heart")

# ---------------------------------------------------------------- 5. timeline, keyframes with portraits
tl = pipeline.build_timeline(bd, settings)
gap = bd["shots"][1]["_narr_start"] - (bd["shots"][0]["_narr_start"] + bd["shots"][0]["_audio"]["duration"])
check(gap < 1.0, f"dialogue lines follow tightly ({gap:.2f}s between lines)")
gen_calls.clear()
bd["shots"][2]["visual"] = "Rain on the window while Tom stares outside"
pipeline.stage_keyframes(pdir, bd, settings)
check(gen_calls[0]["refs"] == ["anna.png"], f"speaker's photo in her shot: {gen_calls[0]['refs']}")
check(gen_calls[1]["refs"] == [Path(next(r for r in bd["references"] if r["id"] == bd["cast"][1]["ref"])["image"]).name],
      "invented character's portrait in his shot")
check(gen_calls[2]["refs"] == [Path(next(r for r in bd["references"] if r["id"] == bd["cast"][1]["ref"])["image"]).name],
      "narrator shot: character named in the visual is included")
check("Image 1 shows Anna" in gen_calls[0]["prompt"], "FLUX told which image is Anna")

# ---------------------------------------------------------------- 6. motion + lip-sync stage
from aivs import i2v, lipsync, motion, talk  # noqa: E402
i2v.unavailable_reason = lambda: "not installed in test"
talk.unavailable_reason = lambda: "not installed in test"   # never start the real 14B model from a test
motion.parallax_clip = lambda img, dep, tmp, *a: Path(tmp).write_bytes(b"v")
pipeline.stage_motion(pdir, bd, settings)
check(not any(s["_backend"] == "talk" for s in bd["shots"]), "talking model not installed -> no talk backend")
check("is speaking" in bd["shots"][0].get("_wan_action", "") and not bd["shots"][2].get("_wan_action"),
      f"speaking shots get a performance direction, the narrator shot does not: {bd['shots'][0].get('_wan_action')}")
# talking model installed: speaking on-screen shots are generated by it (faked), from keyframe + their line
import copy  # noqa: E402
bdt = copy.deepcopy(bd)
i2v.unavailable_reason = lambda: None
talk.unavailable_reason = lambda: None
talk_calls = []


def fake_talk(shot, audio16, out, *, prompt, frames, settings, workdir):
    x, sr = sf.read(audio16)
    talk_calls.append({"id": shot["id"], "frames": frames, "sr": sr, "len": x.size / sr, "prompt": prompt})
    if shot["id"] == "s04":
        raise RuntimeError("Metal out of memory")
    Path(out).write_bytes(b"t")


talk.generate = fake_talk
talk.conform = lambda raw, tmp, frames, size, fps, log: Path(tmp).write_bytes(b"c")
i2v.raw_clips = lambda shot, board, st, cdir, workdir, seconds: [p for p in i2v.raw_paths(shot, board, st, cdir, seconds)
                                                                  if not p.write_bytes(b"w")]
i2v.conform = lambda raws, tmp, frames, size, fps, log, native=True: (Path(tmp).write_bytes(b"v"), 1.0)[1]
bdt["shots"][1]["on_screen"] = True
pipeline.stage_motion(pdir, bdt, settings | {"motion": "wan"})
backends = [s["_backend"] for s in bdt["shots"]]
check([c["id"] for c in talk_calls] == ["s01", "s02", "s04"], f"talking model for on-screen speakers only: {talk_calls}")
check(backends == ["talk", "talk", "wan", "wan"], f"narrator shot -> Wan, failed talking shot -> Wan: {backends}")
check("talking model failed" in bdt["_motion_notes"]["s04"], "talking-model failure noted")
c0, s0 = talk_calls[0], bdt["shots"][0]
check(c0["sr"] == 16000 and abs(c0["len"] - c0["frames"] / 25) < 0.01
      and c0["frames"] >= s0["_frames"] / settings["fps"] * 25, "16 kHz line track covering the shot at 25 fps")
check("is speaking" in c0["prompt"] and "Anna" not in c0["prompt"], f"talking prompt: {c0['prompt'][:80]}")
lipsync.unavailable_reason = lambda: None
ls_seen = []
lipsync.run_shot = lambda clip, a16, out, **kw: (ls_seen.append(kw["shot_id"]), Path(out).write_bytes(b"l"))
pipeline.stage_lipsync(pdir, bdt, settings, tl)
check(ls_seen == ["s04"], f"lip-sync skips talking-model shots: {ls_seen}")
talk.unavailable_reason = lambda: "not installed in test"
i2v.unavailable_reason = lambda: "not installed in test"
lipsync.unavailable_reason = lambda: "envs/lipsync is missing"
pipeline.stage_lipsync(pdir, bd, settings, tl)
check(set(bd["_lipsync_notes"]) == {"s01", "s02", "s04"}, "no lip-sync installed -> noted for speaking shots, no crash")
bd["_lipsync_notes"] = {}
lipsync.unavailable_reason = lambda: None
ls_calls = []


def fake_run_shot(clip, audio16, out, **kw):
    x, sr = sf.read(audio16)
    ls_calls.append({"id": kw["shot_id"], "sr": sr, "len": x.size / sr, "first_sound": float(np.argmax(np.abs(x) > 0.01) / sr), **kw})
    if kw["shot_id"] == "s04":
        raise RuntimeError("Face not detected")
    Path(out).write_bytes(b"ls")


lipsync.run_shot = fake_run_shot
bd["shots"][1]["on_screen"] = False
pipeline.stage_lipsync(pdir, bd, settings, tl)
check([c["id"] for c in ls_calls] == ["s01", "s04"], "lip-sync only for characters on screen (not narrator, not off-screen)")
s1 = bd["shots"][0]
exp_off = s1["_narr_start"] - s1["_cut_frame"] / settings["fps"]
check(ls_calls[0]["sr"] == 16000 and abs(ls_calls[0]["len"] - s1["_frames"] / settings["fps"]) < 0.01
      and abs(ls_calls[0]["first_sound"] - exp_off) < 0.02, f"line placed at {exp_off:.2f}s inside a clip-long 16 kHz track")
check(ls_calls[0]["resolution"] == 512 and ls_calls[0]["frames"] == s1["_frames"], "512 px, exact frame count")
check(s1["_clip"].endswith(".lipsync.mp4") and s1.get("_lipsync"), "lip-synced clip replaces the motion clip")
_cast = {c["id"]: c for c in bd["cast"]}
_img = {r["id"]: r["image"] for r in bd["references"]}
check(ls_calls[0]["ref_face"] == pdir / _img[_cast[s1["speaker"]]["ref"]]
      and ls_calls[0]["other_faces"] == [pdir / _img[c["ref"]] for c in bd["cast"] if c["id"] != s1["speaker"]],
      "speaker's portrait + the other characters' portraits passed to pick the speaking face")
check("s04" in bd["_lipsync_notes"] and not bd["shots"][3]["_clip"].endswith(".lipsync.mp4"), "failed shot keeps its clip")
ls_calls.clear()
pipeline.stage_lipsync(pdir, bd, settings | {"lipsync": "off"}, tl)
check(not ls_calls, "--lipsync off")

# ---------------------------------------------------------------- 7. captions per speaker
chunks = pipeline._caption_chunks(bd, 42)
check(chunks[0]["text"].startswith("Anna: ") and chunks[0]["colour"] == C.CAPTION_COLOURS[0], "speaker name + colour on her line")
tom_first = next(c for c in chunks if c["text"].startswith("Tom: "))
check(tom_first["colour"] == C.CAPTION_COLOURS[1], "second speaker, second colour")
narr = next(c for c in chunks if "Draußen" in c["text"])
check(not narr["text"].startswith("Erzähler") and "colour" not in narr, "narrator plain white, no name")
ass = TMP / "c.ass"
captions.write_ass(chunks, ass, (1920, 1080))
txt = ass.read_text()
check("{\\fad(90,90)\\c&H9BE3FF&}Anna:" in txt and "{\\fad(90,90)}Draußen" in txt, "ASS colour tags")
srt = TMP / "c.srt"
captions.write_srt(chunks, srt)
check("&H" not in srt.read_text() and "Anna: Hast du" in srt.read_text(), "SRT without colour codes, with names")

# ---------------------------------------------------------------- 8. line_audio (real)
src = TMP / "line.wav"
sf.write(src, np.full(48000, 0.5, np.float32), 48000)
lipsync.line_audio(src, 0.75, 3.0, TMP / "l16.wav")
y, sr = sf.read(TMP / "l16.wav")
check(sr == 16000 and y.size == 48000 and abs(np.argmax(np.abs(y) > 0.1) / sr - 0.75) < 0.01 and abs(y[-100:]).max() < 1e-3,
      "line_audio: 16 kHz, clip-long, line at its offset, silence after")

# ---------------------------------------------------------------- 9. workers with fake model libraries
calls = []


class FakeModel:
    def generate(self, **kw):
        calls.append(kw)
        yield types.SimpleNamespace(audio=np.full(24000, 0.3, np.float32), sample_rate=24000)

    def generate_voice_design(self, **kw):
        calls.append(kw)
        yield types.SimpleNamespace(audio=np.full(24000 * 9, 0.3, np.float32), sample_rate=24000)


mlx = types.ModuleType("mlx"); mlx.core = types.ModuleType("mlx.core"); mlx.core.random = types.SimpleNamespace(seed=lambda s: None)
ma = types.ModuleType("mlx_audio"); ma_tts = types.ModuleType("mlx_audio.tts"); ma_u = types.ModuleType("mlx_audio.tts.utils")
ma_u.load_model = lambda m: FakeModel()
sys.modules.update({"mlx": mlx, "mlx.core": mlx.core, "mlx_audio": ma, "mlx_audio.tts": ma_tts, "mlx_audio.tts.utils": ma_u})
from aivs.workers import tts as W, voice_design as VD  # noqa: E402
job = {"engine": "chatterbox", "model": "m", "lang": "de", "ref_audio": "/job.wav", "cfg_weight": 0.5, "seed": 1,
       "items": [{"id": "a", "text": "Hallo", "out": str(TMP / "a.wav"), "ref_audio": "/anna.wav", "cfg_weight": 0.3},
                 {"id": "b", "text": "Hi", "out": str(TMP / "b.wav"), "ref_audio": None, "cfg_weight": 0.0},
                 {"id": "c", "text": "Na", "out": str(TMP / "c.wav")}]}
(TMP / "job.json").write_text(json.dumps(job))
W.main(str(TMP / "job.json"), str(TMP / "res.json"))
check(calls[0]["ref_audio"] == "/anna.wav" and calls[0]["cfg_weight"] == 0.3, "tts worker: per-line voice")
check("ref_audio" not in calls[1] and calls[1]["cfg_weight"] == 0.0, "tts worker: per-line default voice")
check(calls[2]["ref_audio"] == "/job.wav" and calls[2]["cfg_weight"] == 0.5, "tts worker: job defaults still apply")
calls.clear()
vjob = {"model": "m", "seed": 3, "items": [{"id": "anna", "text": "Guten Morgen", "language": "german", "instruct": "warm",
                                            "out": str(TMP / "v.wav")}]}
(TMP / "vjob.json").write_text(json.dumps(vjob))
VD.main(str(TMP / "vjob.json"), str(TMP / "vres.json"))
y, sr = sf.read(TMP / "v.wav")
check(calls[0]["instruct"] == "warm" and calls[0]["language"] == "german" and sr == 48000 and 7 < y.size / sr < 10,
      "voice design worker writes a 48 kHz sample")

# ---------------------------------------------------------------- 10. MediaPipe stand-in for InsightFace
mp = types.ModuleType("mediapipe")
mp.ImageFormat = types.SimpleNamespace(SRGB=1)
mp.Image = lambda image_format, data: data
results = []


class FakeLandmarker:
    def detect(self, img):  # full 1000x1000 frames get the queued result; tiles/tracking crops see no face
        return results.pop(0) if img.shape[:2] == (1000, 1000) else types.SimpleNamespace(face_landmarks=[])


vision = types.SimpleNamespace(FaceLandmarkerOptions=lambda **k: k, RunningMode=types.SimpleNamespace(IMAGE=0),
                               FaceLandmarker=types.SimpleNamespace(create_from_options=lambda o: FakeLandmarker()))
mtp = types.ModuleType("mediapipe.tasks.python"); mtp.BaseOptions = lambda **k: k; mtp.vision = vision
sys.modules.update({"mediapipe": mp, "mediapipe.tasks": types.ModuleType("mediapipe.tasks"), "mediapipe.tasks.python": mtp})
sys.path.insert(0, str(ROOT / "scripts"))
import lipsync_run as LR  # noqa: E402


def face(cx, cy, scale):
    pts = [types.SimpleNamespace(x=(cx + scale * ((i % 21) - 10) / 10) / 1000, y=(cy + scale * ((i // 21) - 11) / 11) / 1000) for i in range(478)]
    for i in LR.BROW_A:
        pts[i] = types.SimpleNamespace(x=(cx + 0.4 * scale) / 1000, y=(cy - 0.4 * scale) / 1000)  # image-right
    for i in LR.BROW_B:
        pts[i] = types.SimpleNamespace(x=(cx - 0.4 * scale) / 1000, y=(cy - 0.4 * scale) / 1000)  # image-left
    for i in LR.NOSE:
        pts[i] = types.SimpleNamespace(x=cx / 1000, y=cy / 1000)
    return pts


det = LR.MediaPipeFaceDetector(model_path="x")
frame = np.zeros((1000, 1000, 3), np.uint8)
results[:] = [types.SimpleNamespace(face_landmarks=[face(300, 300, 50), face(600, 500, 200)]),
              types.SimpleNamespace(face_landmarks=[]), types.SimpleNamespace(face_landmarks=[])]
bbox, lmk = det(frame)
check(abs(lmk[LR.L106].mean(0)[0] - 520) < 1 and abs(lmk[LR.R106].mean(0)[0] - 680) < 1, "eyebrows ordered image-left/right")
check(abs(lmk[LR.N106].mean(0)[1] - 500) < 1 and bbox[2] - bbox[0] > 300, "largest face (speaker) chosen, nose point")
again = det(frame)
check(again[1] is lmk and det.reused == 1, "missing face -> last detection reused")
det2 = LR.MediaPipeFaceDetector(model_path="x")
results[:] = [types.SimpleNamespace(face_landmarks=[])]
check(det2(frame) == (None, None), "no face at all -> None (LatentSync then fails and the shot keeps its clip)")


class FakeIdentity:  # "embedding" = x of the face centre; the speaker is the face at x=300, the other character at x=600
    def embed(self, rgb, pts):
        return np.array([pts[:, 0].mean()])

    def similarity(self, a, b):
        return 1.0 - abs(float(a[0]) - float(b[0])) / 1000


det3 = LR.MediaPipeFaceDetector(model_path="x")
det3.ident, det3.speaker, det3.others = FakeIdentity(), np.array([300.0]), [np.array([600.0])]
results[:] = [types.SimpleNamespace(face_landmarks=[face(300, 300, 50), face(600, 500, 200)])]
bbox, lmk = det3(frame)
check(abs(lmk[LR.N106].mean(0)[0] - 300) < 1 and "speaker's photo match" in det3.choice,
      f"several faces: the one like the speaker's portrait wins over the larger one ({det3.choice})")
results[:] = [types.SimpleNamespace(face_landmarks=[face(305, 300, 50), face(600, 500, 200)])]
det3.lm = types.SimpleNamespace(detect=lambda img: results.pop(0) if img.shape[:2] == (1000, 1000) else
                                types.SimpleNamespace(face_landmarks=[]))
bbox, lmk = det3(frame)
check(abs(lmk[LR.N106].mean(0)[0] - 305) < 1, "the chosen face is followed by position in the next frame")
det4 = LR.MediaPipeFaceDetector(model_path="x")
det4.ident, det4.speaker, det4.others = FakeIdentity(), np.array([300.0]), [np.array([600.0])]
results[:] = [types.SimpleNamespace(face_landmarks=[face(600, 500, 200)])]
check(det4(frame) == (None, None) and "speaker not found" in det4.choice,
      "only the other character visible -> no face (the wrong mouth is never animated)")
LR._install_shims()
import decord, insightface.app  # noqa: E401,E402
try:
    decord.AudioReader("x")
    check(False, "decord shim raises")
except RuntimeError:
    check(True, "decord/insightface never really imported")

# ---------------------------------------------------------------- 10b. real-person photos: no sexual scenes
from aivs import safety  # noqa: E402
try:
    pipeline.create_project(topic="Anna zieht sich nackt aus", name="bad1", lang="de",
                            refs=[{"kind": "person", "name": "Anna", "path": str(photo)}])
    check(False, "sexual prompt with a real person's photo refused")
except SystemExit as exc:
    check("real people" in str(exc) and not (TMP / "projects" / "bad1" / "project.json").exists(),
          "sexual prompt with a real person's photo refused before planning")
ok_board = {"references": [{"kind": "person", "generated": False}], "shots": [{"narration": "Der Wind bläst über das Meer", "visual": "Anna at a dinner, intimate conversation"}]}
safety.check(ok_board)
check(True, "no false alarm on 'bläst' / 'intimate conversation'")
check(safety.find_sexual(["Eine Sexszene im Bett"]) == "Sexszene" and safety.find_sexual(["Sussex coast"]) is None,
      "German compound caught, 'Sussex' not")
gen_only = {"references": [{"kind": "person", "generated": True}], "shots": [{"narration": "sexy", "visual": ""}]}
safety.check(gen_only)
check(True, "generated portraits (invented people) are not blocked")
bd_bad = read_json(pdir / "project.json")["storyboard"]
bd_bad["shots"][0]["visual"] = "Anna naked in bed"
try:
    safety.check(bd_bad)
    check(False, "edited storyboard checked")
except SystemExit:
    check(True, "edited storyboard with a real photo is checked before rendering")

# ---------------------------------------------------------------- 11. CLI + web
from aivs import cli  # noqa: E402
got = {}
real_create = pipeline.create_project
pipeline.create_project = lambda **kw: (got.update(kw), TMP / "x")[1]
pipeline.render = lambda p: None
cli.need = lambda t: t
vs = TMP / "voice.wav"
sf.write(vs, np.zeros(48000, np.float32), 48000)
castf = TMP / "cast.json"
castf.write_text(json.dumps([{"name": "Anna", "look": "red curls", "photo": "anna.jpg", "voice_sample": "voice.wav"},
                             {"name": "Tom", "voice": "deep"}]))
cli.main(["make", "--prompt", "Streit im Café", "--lang", "de", "--characters", "3", "--narrator", "--cast", str(castf),
          "--lipsync", "off"])
check(got["characters"] == 3 and got["narrator"] and [c["name"] for c in got["cast"]] == ["Anna", "Tom"], "CLI: characters, narrator, cast")
check(got["refs"][0]["kind"] == "person" and got["refs"][0]["path"] == str(photo) and "photo" not in got["cast"][0], "cast photo -> person reference")
check(got["cast"][0]["voice_sample"] == str(vs) and got["settings"]["lipsync"] == "off", "voice sample path + lipsync setting")
pipeline.create_project = real_create

from aivs.web import server  # noqa: E402
started = []
server._start = lambda args, name: (started.append(args), {"ok": True, "name": name})[1]
srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"


def req(path, body=None, raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(base + path, data=data, headers=headers or {"Content-Type": "application/json"},
                               method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


up = json.loads(req("/api/upload", raw=photo.read_bytes(), headers={"X-Filename": "anna.jpg"})[1])
st_, body = req("/api/make", {"mode": "prompt", "text": "Streit im Café", "lang": "de", "characters": 2, "narrator": True,
                             "lipsync": "off", "references": [{"token": up["token"], "kind": "person", "name": "Anna",
                                                               "voice": "helle schnelle Stimme", "consent": True}]})
a = started[-1]
check(st_ == 200 and a[a.index("--characters") + 1] == "2" and "--narrator" in a and a[a.index("--lipsync") + 1] == "off", f"web make: {a}")
cj = json.loads(Path(a[a.index("--cast") + 1]).read_text())
check(cj == [{"name": "Anna", "voice": "helle schnelle Stimme", "look": ""}], "voice given for an uploaded person -> cast entry")
st_, body = req("/api/make", {"mode": "prompt", "text": "x", "characters": "viele"})
check(st_ == 400, "bad character count refused")
voice_file = read_json(pdir / "project.json")["storyboard"]["cast"][0]["voice_design_sample"]
st_, body = req(f"/media/cafe/{voice_file}")
check(st_ == 200 and body[:4] == b"RIFF", "designed voice sample playable in the editor")
st_, body = req("/media/cafe/refs/voices/../../project.json")
check(st_ == 404, "no traversal via voice route")
st_, body = req("/api/project/cafe/render", {"shots": [{"id": "s02", "speaker": "anna", "on_screen": False},
                                                      {"id": "s03", "speaker": "hacker"}],
                                            "cast": [{"id": "tom", "look": "  tall   man ", "voice": "low"}, {"id": "nobody", "look": "x"}]})
pj = read_json(pdir / "project.json")["storyboard"]
s2 = next(s for s in pj["shots"] if s["id"] == "s02")
check(s2["speaker"] == "anna" and s2["on_screen"] is False, "editor: speaker + on-screen saved")
check(next(s for s in pj["shots"] if s["id"] == "s03")["speaker"] == "narrator", "unknown speaker ignored")
check(next(c for c in pj["cast"] if c["id"] == "tom")["look"] == "tall man", "editor: character look saved (normalised)")
st_, body = req("/api/project/cafe/render", {"shots": [{"id": "s01", "visual": "Anna naked in the shower"}]})
check(st_ == 400 and b"real people" in body, "web editor refuses sexual edits for real-person photos")
st_, body = req("/")
check(b'id="characters"' in body and b'id="lipsync"' in body and b"castBox" in body, "UI has characters, lip-sync, cast editor")
srv.shutdown()

print(f"\nALL {len(PASSED)} CHECKS PASSED")
shutil.rmtree(TMP, ignore_errors=True)
