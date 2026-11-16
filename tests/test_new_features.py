"""Run: envs/studio/bin/python tests/test_new_features.py   (no models needed, ~10 s)

Offline tests for: Wan as default motion (with per-shot fallback) and user reference images.
Models are replaced by fakes; everything else (file handling, cache keys, planner parsing, CLI, web server) is real."""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="aivs-test-")).resolve()  # macOS: /var -> /private/var
os.environ["AIVS_PROJECTS"] = str(TMP / "projects")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from aivs import config, images, pipeline, planner, references as R  # noqa: E402
from aivs.util import digest, read_json, write_json  # noqa: E402

PASSED = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    PASSED.append(msg)


# ---------------------------------------------------------------- fixtures
def make_photo(path, size=(3000, 2000), color=(200, 30, 30), exif_rotate=False, rgba=False):
    img = Image.new("RGBA" if rgba else "RGB", size, color + ((128,) if rgba else ()))
    if exif_rotate:
        exif = Image.Exif()
        exif[0x0112] = 6      # orientation: rotate 90° CW on display
        exif[0x8825] = {2: (50.0, 0.0, 0.0)}  # GPS block that must not survive
        img.save(path, "JPEG", exif=exif.tobytes())
    else:
        img.save(path)
    return path


SRC = TMP / "src"
SRC.mkdir()
anna = make_photo(SRC / "anna.jpg", exif_rotate=True)
kitchen = make_photo(SRC / "kitchen.png", size=(2400, 1600), color=(30, 120, 200))
logo = make_photo(SRC / "mug.png", size=(800, 800), color=(10, 200, 10), rgba=True)
tiny = make_photo(SRC / "tiny.png", size=(20, 20))

# ---------------------------------------------------------------- 1. references module
out = TMP / "norm.png"
sha = R.normalize_image(anna, out)
im = Image.open(out)
check(max(im.size) == R.MAX_SIDE and im.size == (1024, 1536), f"EXIF rotation applied + scaled to {R.MAX_SIDE}: {im.size}")
check(not im.info.get("exif") and "GPSInfo" not in str(im.getexif()), "EXIF/GPS metadata stripped")
check(len(sha) == 64, "sha256 returned")
R.normalize_image(logo, TMP / "mug_norm.png")
check(Image.open(TMP / "mug_norm.png").mode == "RGB", "transparent PNG flattened to RGB")
try:
    R.normalize_image(tiny, TMP / "tiny_norm.png")
    check(False, "tiny image rejected")
except ValueError:
    check(True, "tiny image rejected")
bad = SRC / "notimage.jpg"
bad.write_bytes(b"hello")
try:
    R.normalize_image(bad, TMP / "bad.png")
    check(False, "non-image rejected")
except ValueError:
    check(True, "non-image rejected")

R.fit_image(out, TMP / "fit.png", (1920, 1088))
check(Image.open(TMP / "fit.png").size == (1920, 1088), "fit_image cover-crops to the keyframe size")

spec = R.parse_cli_spec("person", f"Anna={anna}")
check(spec["name"] == "Anna" and spec["kind"] == "person", "CLI spec NAME=PATH")
check(R.parse_cli_spec("background", str(kitchen))["name"] == "kitchen", "CLI spec name from file name")
check(R.clean_spec({"kind": "Hintergrund", "name": "x", "path": "p"})["kind"] == "background", "German kind alias")

pdir0 = TMP / "p0"
refs = R.import_refs(pdir0, [{"kind": "person", "name": "Anna", "path": str(anna)},
                             {"kind": "person", "name": "Anna", "path": str(kitchen)}])
check([r["id"] for r in refs] == ["anna", "anna-2"], f"unique ids for duplicate names: {[r['id'] for r in refs]}")
refs2 = R.import_refs(pdir0, [{"kind": "person", "name": "Anna", "path": str(anna)}], refs)
check([r["id"] for r in refs2] == ["anna-2", "anna"] and len(refs2) == 2, "re-import of same name replaces, no duplicates")

shots = [{"id": "s01", "narration": "Anna betritt die Küche.", "visual": "Anna walks into the Kitchen"},
         {"id": "s02", "narration": "Annabelle ist nicht gemeint.", "visual": "a cat"},
         {"id": "s03", "narration": "x", "visual": "y", "refs": ["kitchen"]}]
rl = [{"id": "anna", "kind": "person", "name": "Anna", "description": "", "image": "refs/anna.png", "sha": "1"},
      {"id": "kitchen", "kind": "background", "name": "Kitchen", "description": "bright kitchen", "image": "refs/k.png",
       "sha": "2", "as_is": True},
      {"id": "mug", "kind": "object", "name": "Mug", "description": "", "image": "refs/m.png", "sha": "3"},
      {"id": "bob", "kind": "person", "name": "Bob", "description": "", "image": "refs/b.png", "sha": "4"}]
R.assign_by_name(shots, rl)
check(shots[0]["refs"] == ["anna", "kitchen"], f"name match, case-insensitive: {shots[0]['refs']}")
check(shots[1]["refs"] == [], "no partial-word match (Annabelle != Anna)")
check(shots[2]["refs"] == ["kitchen"], "existing assignment kept")
by_id = {r["id"]: r for r in rl}
order = R.for_shot({"id": "s9", "refs": ["kitchen", "mug", "anna", "bob"]}, by_id, Path("/p"))
check([r["id"] for r in order] == ["anna", "bob", "mug"], f"people first, capped at {R.MAX_PER_SHOT}: {[r['id'] for r in order]}")
check(order[0]["path"] == "/p/refs/anna.png", "absolute path")
check(R.use_as_is([by_id["kitchen"]]) and not R.use_as_is([by_id["kitchen"], by_id["anna"]]), "as-is only for a lone place")
p = R.flux_prompt("SCENE", [dict(by_id["anna"]), dict(by_id["kitchen"])], look_reference=True)
check("Image 1 shows Anna" in p and "Image 2 shows the location Kitchen (bright kitchen)" in p
      and "Image 3 only defines" in p and p.endswith("Scene: SCENE"), "FLUX prompt numbers the images")
check(R.plain_text("Anna smiles in the Kitchen", [by_id["anna"], by_id["kitchen"]]) == "the person smiles in the bright kitchen",
      "names replaced for Wan")
R.sanitize_shot_refs([s := {"refs": ["anna", "zzz", "anna", 3]}], rl)
check(s["refs"] == ["anna"], "sanitize drops unknown/duplicate ids")

# ---------------------------------------------------------------- 2. keyframe cache keys unchanged without refs
board = json.loads(Path(__file__).resolve().parent.parent.joinpath("examples").joinpath("leuchtturm-de.json").read_text())
settings = pipeline.default_settings() | {"aspect": "9:16"}
_, ks = pipeline.sizes(settings)
consistent = pipeline.use_reference(board, settings)
expect, rk = [], None
for i, s in enumerate(board["shots"]):  # the original formula, copied verbatim
    key = digest([s["visual"], board["style"], ks, settings["seed"] + i, settings["image_steps"], config.MODELS["image"], rk,
                  board.get("recurring_subject") if consistent else None])
    if consistent and i == 0:
        rk = key
    expect.append(key)
calls = []
images.generate = lambda prompt, out, size, seed, steps, log_path, reference=None: (calls.append((prompt, reference, size)), Image.new("RGB", size).save(out))
pk = TMP / "keys"; pk.mkdir()
b2 = json.loads(json.dumps(board))
pipeline.stage_keyframes(pk, b2, settings)
check([s["_ikey"] for s in b2["shots"]] == expect, "keyframe cache keys of projects without references unchanged")
check(calls[0][1] is None and all(c[1] and len(c[1]) == 1 for c in calls[1:]), "look-reference behaviour unchanged")

# ---------------------------------------------------------------- 3. create_project from a storyboard with references
sb = {"title": "Anna kocht", "language": "de", "style": "warm film look", "shots": [
    {"narration": "Anna kommt nach Hause.", "visual": "Anna opens the front door at dusk"},
    {"narration": "Die Küche wartet schon.", "visual": "Empty kitchen in morning light", "refs": ["kitchen"]},
    {"narration": "Sie kocht Kaffee in der Küche.", "visual": "Anna pours coffee into the mug in the kitchen"},
    {"narration": "Ein ruhiger Moment.", "visual": "Close-up of steam rising"}]}
specs = [{"kind": "person", "name": "Anna", "path": str(anna), "description": "woman, thirties, red coat"},
         {"kind": "background", "name": "Kitchen", "path": str(kitchen)},
         {"kind": "object", "name": "mug", "path": str(logo)}]
pdir = pipeline.create_project(storyboard=sb, name="anna-test", settings={"aspect": "16:9"}, refs=specs)
proj = read_json(pdir / "project.json")
bd = proj["storyboard"]
check(proj["settings"]["motion"] == "wan", "new projects default to Wan motion")
check([r["id"] for r in bd["references"]] == ["anna", "kitchen", "mug"], "references stored in the storyboard")
check(all((pdir / r["image"]).is_file() for r in bd["references"]), "reference PNGs copied into <project>/refs")
check(bd["shots"][0]["refs"] == ["anna"] and bd["shots"][1]["refs"] == ["kitchen"]
      and set(bd["shots"][2]["refs"]) == {"anna", "kitchen", "mug"} and bd["shots"][3]["refs"] == [],
      f"shots assigned: {[s['refs'] for s in bd['shots']]}")
check("references" in proj["request"], "request records the references")
again = pipeline.create_project(storyboard=sb, name="anna-test", settings={"aspect": "16:9"}, refs=specs)
check(read_json(again / "project.json")["storyboard"]["references"] == bd["references"], "same make twice = no replan")

# keyframes for that project
calls.clear()
settings = pipeline.default_settings() | proj["settings"]
pipeline.stage_keyframes(pdir, bd, settings)
by_shot = {c[2]: c for c in calls}
check(len(calls) == 3, f"3 generated keyframes, the lone-kitchen shot uses the photo: {len(calls)} calls")
s2img = Image.open(bd["shots"][1]["image"])
check(s2img.size == (1920, 1088) and s2img.getpixel((10, 10))[2] > 150, "kitchen photo became keyframe s02 (blue, cropped)")
first = calls[0]
check([Path(x).name for x in first[1]] == ["anna.png"] and "Image 1 shows Anna (woman, thirties, red coat)" in first[0],
      "s01 generated with Anna's photo")
third = calls[1]
check([Path(x).name for x in third[1]] == ["anna.png", "mug.png", "kitchen.png"], f"s03 refs order person, object, place: {[Path(x).name for x in third[1]]}")
check(calls[2][1] is None, "s04 without references = plain text-to-image (no look reference with user photos)")
check(not pipeline.use_reference(bd, settings), "auto look-reference off when user photos exist")
keys1 = [s["_ikey"] for s in bd["shots"]]
# changing a reference image invalidates only the shots that use it
pipeline.add_references(pdir, [{"kind": "object", "name": "mug", "path": str(kitchen), "id": "mug"}])
bd2 = read_json(pdir / "project.json")["storyboard"]
calls.clear()
pipeline.stage_keyframes(pdir, bd2, settings)
keys2 = [s["_ikey"] for s in bd2["shots"]]
check([a == b for a, b in zip(keys1, keys2)] == [True, True, False, True], "replacing the mug re-renders only s03")
check(len(calls) == 1, "only one keyframe regenerated")

# ---------------------------------------------------------------- 4. motion: Wan default, fallback per shot
from aivs import i2v, motion  # noqa: E402

bd3 = read_json(pdir / "project.json")["storyboard"]
bd3["shots"][3]["animate"] = "camera"
for s in bd3["shots"]:
    s["_audio"] = {"duration": 2.0, "words": []}
tl = pipeline.build_timeline(bd3, settings)
pipeline.stage_keyframes(pdir, bd3, settings)
wan_calls, prompts = [], []
i2v.unavailable_reason = lambda: None
from aivs import talk  # noqa: E402
talk.unavailable_reason = lambda: "not installed in test"   # never start the real 14B model from a test


def fake_raw(shot, board, st, cdir, workdir, seconds):
    prompts.append(i2v.build_prompt(shot, board["style"]))
    wan_calls.append(shot["id"])
    if shot["id"] == "s02":
        raise RuntimeError("Metal out of memory")
    ps = i2v.raw_paths(shot, board, st, cdir, seconds)
    for p in ps:
        p.write_bytes(b"x")
    return ps


real_conform = i2v.conform
i2v.raw_clips = fake_raw
made = []
i2v.conform = lambda raws, tmp, frames, size, fps, log, native=True: (made.append("wan"), Path(tmp).write_bytes(b"v"), 1.0)[2]
motion.parallax_clip = lambda img, dep, tmp, *a: (made.append("parallax"), Path(tmp).write_bytes(b"v"))
motion.camera_clip = lambda img, tmp, *a: (made.append("camera"), Path(tmp).write_bytes(b"v"))
pipeline.run_worker = lambda name, job, wd: [Path(it["out"]).write_bytes(b"d") for it in job["items"]] and {"items": []}
pipeline.stage_motion(pdir, bd3, settings)
check([s["_backend"] for s in bd3["shots"]] == ["wan", "parallax", "wan", "camera"],
      f"backends: wan default, failed s02 -> parallax, per-shot override camera: {[s['_backend'] for s in bd3['shots']]}")
check("s02" in bd3["_motion_notes"] and "Metal out of memory" in bd3["_motion_notes"]["s02"], "fallback reason recorded")
check(wan_calls == ["s01", "s02", "s03"], "camera-override shot never sent to Wan")
check("Anna" not in prompts[0] and "woman, thirties, red coat" in prompts[0], f"Wan prompt uses the description: {prompts[0][:90]}")
check(sorted(made) == ["camera", "parallax", "wan", "wan"], f"clips made: {made}")
# second run: Wan cached, nothing regenerated
wan_calls.clear(); made.clear()
for s in bd3["shots"]:
    s.pop("_backend", None)
pipeline.stage_motion(pdir, bd3, settings)
check(made == [], f"rerun reuses finished clips: {made}")
# Wan not installed -> everything falls back, no crash
i2v.unavailable_reason = lambda: "envs/wan is missing"
bd4 = json.loads(json.dumps({k: v for k, v in bd3.items()}))
for s in bd4["shots"]:
    s.pop("animate", None)
pipeline.stage_keyframes(pdir, bd4, settings)
pipeline.stage_motion(pdir, bd4, settings)
check(all(s["_backend"] == "parallax" for s in bd4["shots"]), "Wan unavailable -> 2.5D camera for all shots, no crash")
check(pipeline.default_settings()["motion"] == "wan", "default motion is wan")
# native length: chunks at real speed instead of one stretched 2 s clip
check(i2v.chunk_plan(2.0) == [49] and i2v.chunk_plan(3.3) == [81] and i2v.chunk_plan(6.4) == [81, 81]
      and i2v.chunk_plan(5.0, 121) == [121] and i2v.chunk_plan(12.0, 121) == [97, 97, 97],
      f"chunk plans: {i2v.chunk_plan(2.0)} {i2v.chunk_plan(6.4)} {i2v.chunk_plan(12.0, 121)}")
check(all(sum(n - 1 for n in i2v.chunk_plan(t, m)) + 1 >= t * 24 + 1 and max(i2v.chunk_plan(t, m)) <= m
          for t in (0.5, 3.3, 7.7, 19.9) for m in (49, 81, 121)), "chunks always cover the shot, never above the cap")
check(pipeline.default_settings()["wan_length"] == "native", "native length is the default")
paths = i2v.raw_paths(bd3["shots"][0], bd3, settings, pdir / "clips", 6.4)
check(len(paths) == 2 and paths[0].name != paths[1].name and ".wan1." in paths[1].name, "one cache file per chunk")
check(len(i2v.raw_paths(bd3["shots"][0], bd3, settings | {"wan_length": "stretch"}, pdir / "clips", 6.4)) == 1,
      "stretch mode: one clip")
spk = dict(bd3["shots"][0], speaker="anna", on_screen=True, _wan_action="Anna is speaking calmly, lips moving")
check("is speaking" in i2v.build_prompt(spk, "x") and "is speaking" not in i2v.build_prompt(bd3["shots"][0], "x"),
      "speaking shots direct the performance in the Wan prompt")
# real ffmpeg join of two 24 fps chunks (first frame of chunk 2 dropped) -> exact frame count at 30 fps, no interpolation
import subprocess  # noqa: E402
c1, c2 = TMP / "c1.mp4", TMP / "c2.mp4"
for c, n in ((c1, 49), (c2, 49)):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=832x480:rate=24", "-frames:v", str(n),
                    "-pix_fmt", "yuv420p", str(c)], check=True)
joined = TMP / "joined.mp4"
check(real_conform([c1, c2], joined, 120, (640, 360), 30, TMP / "conform.log") == 1.0, "native conform: no slow-down")
nf = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                     "stream=nb_read_frames,width,r_frame_rate", "-of", "csv=p=0", str(joined)], capture_output=True, text=True).stdout.strip()
check(nf == "640,30/1,120", f"joined chunks: exactly 120 frames at 30 fps, output size ({nf})")

# ---------------------------------------------------------------- 5. planner with references
captured = {}


def fake_ask(reqs, wd):
    captured["user"] = reqs[0]["user"]
    if reqs[0]["id"] == "script":
        return {"script": "Anna kommt am Abend müde nach Hause zurück. Danach kocht sie sich in aller Ruhe einen starken Kaffee."}
    import re as _re
    n = int(_re.search(r"There must be exactly (\d+) shots", reqs[0]["user"]).group(1))
    shots = [{"n": 1, "visual": "Anna opens the door of her flat at dusk", "motion": "push_in", "refs": ["anna", "ghost"]},
             {"n": 2, "visual": "Steam rises from a cup on the counter", "motion": "pan_left"},
             {"n": 3, "visual": "Wide shot of a quiet street with parked bicycles", "motion": "pan_up"}][:n]
    return {"board": json.dumps({"title": "T", "recurring_subject": "a woman with red coat and brown hair walking",
                                 "style": "s", "shots": shots})}


planner._ask = fake_ask
b = planner.plan(topic=None, script="Anna kommt am Abend müde nach Hause zurück. Danach kocht sie sich in aller Ruhe einen starken Kaffee.", lang="de", seconds=20, style=None,
                 workdir=TMP / "w", refs=[dict(r) for r in bd["references"]])
check('id "anna" (person) called "Anna"' in captured["user"] and '"refs": ["<id>"]' in captured["user"], "planner prompt lists the photos")
check(b["shots"][0]["refs"] == ["anna", "ghost"] and all(x["refs"] == [] for x in b["shots"][1:]), f"planner refs parsed (filtered later), {len(b['shots'])} shots")
check(b["recurring_subject"] == "", "recurring text subject dropped when photos exist")
b_plain = planner.plan(topic=None, script="Anna kommt am Abend müde nach Hause zurück. Danach kocht sie sich in aller Ruhe einen starken Kaffee.", lang="de", seconds=20, style=None, workdir=TMP / "w")
check("refs" not in b_plain["shots"][0] and '"refs"' not in captured["user"], "planner unchanged without references")
pipeline.planner.plan = planner.plan
p2 = pipeline.create_project(script="Anna kommt am Abend müde nach Hause zurück. Danach kocht sie sich in aller Ruhe einen starken Kaffee.", lang="de", name="planned",
                             refs=[specs[0]])
s0 = read_json(p2 / "project.json")["storyboard"]["shots"]
check(s0[0]["refs"] == ["anna"], f"unknown planner id removed, name match added: {[s['refs'] for s in s0]}")

# ---------------------------------------------------------------- 6. CLI
from aivs import cli  # noqa: E402

got = {}
pipeline.create_project = lambda **kw: (got.update(kw), TMP / "x")[1]
pipeline.render = lambda p: None
reff = TMP / "refs.json"
reff.write_text(json.dumps([{"kind": "background", "name": "Strand", "path": "src/kitchen.png", "as_is": False}]))
cli.need = lambda t: t
cli.main(["make", "--prompt", "Anna am Strand", "--person", f"Anna={anna}", "--object", str(logo),
          "--references", str(reff), "--motion", "parallax"])
check([(r["kind"], r["name"]) for r in got["refs"]] == [("person", "Anna"), ("object", "mug"), ("background", "Strand")],
      f"CLI collects --person/--object/--references: {[(r['kind'], r['name']) for r in got['refs']]}")
check(got["refs"][2]["path"] == str(kitchen) and got["refs"][2]["as_is"] is False, "JSON paths relative to the file")
check(got["settings"]["motion"] == "parallax", "--motion parallax still selectable")
try:
    cli.main(["make", "--prompt", "x", "--person", "Ghost=/nope.jpg"])
    check(False, "missing image rejected")
except SystemExit as exc:
    check("not found" in str(exc), "missing image rejected with a clear message")

# ---------------------------------------------------------------- 7. web server
from aivs.web import server  # noqa: E402

started = []
server._start = lambda args, name: (started.append(args), {"ok": True, "name": name})[1]
srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{port}"


def req(path, body=None, raw=None, headers=None, method=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"Content-Type": "application/json"} if body is not None else {}
    h.update(headers or {})
    r = urllib.request.Request(base + path, data=data, headers=h, method=method or ("POST" if data is not None else "GET"))
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


st, body = req("/api/upload", raw=anna.read_bytes(), headers={"X-Filename": "Anna_Portrait.jpg", "Content-Type": "image/jpeg"})
up = json.loads(body)
check(st == 200 and up["name"] == "Anna Portrait" and (up["width"], up["height"]) == (1024, 1536), f"upload normalised: {up}")
st, body = req(up["url"])
check(st == 200 and body[:8] == b"\x89PNG\r\n\x1a\n", "uploaded PNG served for the thumbnail")
st, body = req("/api/upload", raw=b"not an image", headers={"X-Filename": "x.jpg"})
check(st == 400 and b"could not read" in body, "non-image upload refused")
st, body = req("/api/upload", raw=anna.read_bytes(), headers={"X-Filename": "a.jpg", "Origin": "http://evil.example"})
check(st == 403, "cross-origin upload refused")
up_k = json.loads(req("/api/upload", raw=kitchen.read_bytes(), headers={"X-Filename": "k.png"})[1])

st, body = req("/api/make", {"mode": "prompt", "text": "Anna in der Küche", "references": [
    {"token": up["token"], "kind": "person", "name": "Anna"}]})
check(st == 400 and b"agreed" in body, "person photo without consent refused")
st, body = req("/api/make", {"mode": "prompt", "text": "Anna in der Küche", "lang": "de", "motion": "wan", "references": [
    {"token": up["token"], "kind": "person", "name": "Anna", "consent": True},
    {"token": up_k["token"], "kind": "background", "name": "Küche", "description": "helle Küche"}]})
args = started[-1]
check(st == 200 and "--references" in args and args[args.index("--motion") + 1] == "wan", f"make passes refs + motion: {args}")
written = json.loads(Path(args[args.index("--references") + 1]).read_text())
check([w["name"] for w in written] == ["Anna", "Küche"] and all(Path(w["path"]).is_file() for w in written), "refs.json points at uploads")
st, body = req("/api/make", {"mode": "prompt", "text": "x", "references": [{"token": "../../etc/passwd", "kind": "object", "name": "x"}]})
check(st == 400, "bad token refused")

# existing project: add / assign / remove
st, body = req("/api/project/anna-test/references", {"token": up_k["token"], "kind": "background", "name": "Balkon"})
check(st == 200 and [r["id"] for r in json.loads(body)["references"]][-1] == "balkon", "reference added to an existing project")
st, body = req("/media/anna-test/refs/balkon.png")
check(st == 200, "project reference image served")
st, body = req("/media/anna-test/refs/../project.json")
check(st == 404, "no path traversal through /media")
st, body = req("/api/project/anna-test/render", {"shots": [{"id": "s04", "refs": ["balkon", "nope"], "animate": "parallax"},
                                                           {"id": "s01", "animate": ""}]})
pj = read_json(TMP / "projects" / "anna-test" / "project.json")
s04 = next(s for s in pj["storyboard"]["shots"] if s["id"] == "s04")
check(s04["refs"] == ["balkon"] and s04["animate"] == "parallax", "editor saves per-shot refs + motion type")
st, body = req("/api/project/anna-test/references/remove", {"id": "balkon"})
pj = read_json(TMP / "projects" / "anna-test" / "project.json")
check("balkon" not in [r["id"] for r in pj["storyboard"]["references"]]
      and all("balkon" not in (s.get("refs") or []) for s in pj["storyboard"]["shots"]), "reference removed from project and shots")
st, body = req("/api/projects")
check(st == 200 and all(p["name"] not in ("_uploads", "_jobs") for p in json.loads(body)["projects"]), "upload folder not listed as a project")
st, body = req("/")
check(st == 200 and b'id="newRefFile"' in body and b'value="wan"' in body, "UI served with image upload + motion choice")
srv.shutdown()

print(f"\nALL {len(PASSED)} CHECKS PASSED")
shutil.rmtree(TMP, ignore_errors=True)
