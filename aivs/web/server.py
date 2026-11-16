"""Local web studio (stdlib only). One render job at a time; everything stays on this machine."""
from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .. import config, pipeline, references
from ..util import read_json, slugify, write_json

HERE = Path(__file__).parent
JOB = {"proc": None, "log": None, "name": None, "started": None, "cmd": None}
LOCK = threading.Lock()
SAFE = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}$")
TOKEN = re.compile(r"^[0-9a-f]{20}$")
MEDIA = {"final.mp4", "captions.srt", "report.json"}
UPLOADS = config.PROJECTS / "_uploads"  # normalised uploads waiting to be attached to a project


def _upload_path(token: str) -> Path | None:
    return UPLOADS / f"{token}.png" if isinstance(token, str) and TOKEN.match(token) else None


def _spec_from(item: dict) -> dict:
    """Web form entry {token, kind, name, description, as_is, consent} -> reference spec (raises ValueError)."""
    path = _upload_path(item.get("token"))
    if not path or not path.is_file():
        raise ValueError("uploaded image not found — please upload it again")
    kind = str(item.get("kind") or "person")
    if kind == "person" and not item.get("consent"):
        raise ValueError("photos of people need the confirmation that the person agreed to appear in the video")
    spec = {"kind": kind, "name": str(item.get("name") or "").strip(), "description": item.get("description") or "",
            "path": str(path)}
    if "as_is" in item:
        spec["as_is"] = bool(item["as_is"])
    return references.clean_spec(spec)


def _job_state() -> dict:
    proc = JOB["proc"]
    state = "idle"
    if proc is not None:
        code = proc.poll()
        state = "running" if code is None else ("done" if code == 0 else "failed")
    tail = ""
    if JOB["log"] and Path(JOB["log"]).exists():
        tail = "\n".join(Path(JOB["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
    return {"state": state, "name": JOB["name"], "log": tail,
            "elapsed": round(time.time() - JOB["started"]) if JOB["started"] and state == "running" else None}


def _start(args: list[str], name: str) -> dict:
    with LOCK:
        if JOB["proc"] is not None and JOB["proc"].poll() is None:
            return {"error": "a render is already running"}
        logdir = config.PROJECTS / "_jobs"
        logdir.mkdir(parents=True, exist_ok=True)
        log_path = logdir / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.log"
        fh = open(log_path, "w", encoding="utf-8")
        cmd = [sys.executable, "-u", "-m", "aivs.cli", *args]
        JOB.update(proc=subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=config.worker_env()),
                   log=str(log_path), name=name, started=time.time(), cmd=cmd)
    return {"ok": True, "name": name}


def _projects() -> list[dict]:
    out = []
    if not config.PROJECTS.exists():
        return out
    for d in sorted(config.PROJECTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        proj = read_json(d / "project.json")
        if not proj:
            continue
        rep = read_json(d / "report.json") or {}
        out.append({"name": d.name, "title": proj["storyboard"].get("title", d.name),
                    "language": proj["storyboard"].get("language"), "shots": len(proj["storyboard"]["shots"]),
                    "rendered": (d / "final.mp4").exists(), "duration": rep.get("duration_s"),
                    "mtime": int((d / "final.mp4").stat().st_mtime) if (d / "final.mp4").exists() else 0})
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "aivs"

    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if rng and (rng.group(1) or rng.group(2)):
            if rng.group(1):
                start = int(rng.group(1))
                end = min(int(rng.group(2)), size - 1) if rng.group(2) else size - 1
            else:
                start = max(size - int(rng.group(2)), 0)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _local_only(self) -> bool:
        """Reject cross-site requests: state-changing calls must come from this page."""
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and urlparse(origin).netloc != host:
            self._json({"error": "cross-origin request refused"}, 403)
            return False
        return True

    def _upload(self):
        size = int(self.headers.get("Content-Length", "0") or 0)
        if size <= 0:
            return self._json({"error": "empty upload"}, 400)
        if size > references.MAX_UPLOAD_BYTES:
            self.rfile.read(size)
            return self._json({"error": f"image too large (max {references.MAX_UPLOAD_BYTES // 2**20} MB)"}, 413)
        data = self.rfile.read(size)
        fname = unquote(self.headers.get("X-Filename", "upload"))
        suffix = Path(fname).suffix.lower()[:6] if re.fullmatch(r"\.[A-Za-z0-9]{1,5}", Path(fname).suffix or "") else ""
        UPLOADS.mkdir(parents=True, exist_ok=True)
        import hashlib
        token = hashlib.sha256(data).hexdigest()[:20]
        raw = Path(tempfile.mkdtemp(prefix="aivs-up-")) / f"original{suffix}"
        raw.write_bytes(data)
        try:
            references.normalize_image(raw, UPLOADS / f"{token}.png")
        except Exception as exc:
            return self._json({"error": f"could not read this image: {exc}"}, 400)
        finally:
            raw.unlink(missing_ok=True)
        from PIL import Image
        with Image.open(UPLOADS / f"{token}.png") as im:
            w, h = im.size
        stem = re.sub(r"[_-]+", " ", Path(fname).stem).strip()[:40]
        return self._json({"ok": True, "token": token, "url": f"/media/_uploads/{token}.png", "width": w, "height": h,
                           "name": stem})

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self._file(HERE / "index.html")
        if path == "/api/projects":
            return self._json({"projects": _projects(), "job": _job_state()})
        if path == "/api/job":
            return self._json(_job_state())
        m = re.match(r"^/api/project/([^/]+)$", path)
        if m and SAFE.match(m.group(1)):
            d = config.PROJECTS / m.group(1)
            proj = read_json(d / "project.json")
            if not proj:
                return self._json({"error": "not found"}, 404)
            frames = {}
            for s in proj["storyboard"]["shots"]:
                hits = sorted((d / "frames").glob(f"{s['id']}-*.png"), key=lambda p: p.stat().st_mtime) if (d / "frames").exists() else []
                hits = [h for h in hits if ".tmp" not in h.name and ".depth" not in h.name]
                if hits:
                    frames[s["id"]] = hits[-1].name
            return self._json({"project": proj, "report": read_json(d / "report.json"), "frames": frames})
        m = re.match(r"^/media/([^/]+)/refs/voices/([a-z0-9][a-z0-9-]{0,80}\.(?:wav|mp3|m4a|flac))$", path)
        if m and SAFE.match(m.group(1)):
            return self._file(config.PROJECTS / m.group(1) / "refs" / "voices" / m.group(2))
        m = re.match(r"^/media/_uploads/([0-9a-f]{20})\.png$", path)
        if m:
            return self._file(UPLOADS / f"{m.group(1)}.png")
        m = re.match(r"^/media/([^/]+)/(frames/|refs/)?([^/]+)$", path)
        if m and SAFE.match(m.group(1)):
            name = m.group(3)
            if m.group(2) == "frames/" and re.fullmatch(r"s\d+-[0-9a-f]+\.png", name):
                return self._file(config.PROJECTS / m.group(1) / "frames" / name)
            if m.group(2) == "refs/" and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}\.png", name):
                return self._file(config.PROJECTS / m.group(1) / "refs" / name)
            if not m.group(2) and name in MEDIA:
                return self._file(config.PROJECTS / m.group(1) / name)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._local_only():
            return
        path = unquote(urlparse(self.path).path)
        if path == "/api/upload":  # raw image bytes in the body, original file name in X-Filename
            return self._upload()
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or 0)) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad JSON"}, 400)
        m = re.match(r"^/api/project/([^/]+)/references(/remove)?$", path)
        if m and SAFE.match(m.group(1)):
            d = config.PROJECTS / m.group(1)
            if not (d / "project.json").exists():
                return self._json({"error": "not found"}, 404)
            try:
                if m.group(2):
                    refs = pipeline.remove_reference(d, str(body.get("id") or ""))
                else:
                    refs = pipeline.add_references(d, [_spec_from(body)])
            except (ValueError, OSError) as exc:
                return self._json({"error": str(exc)}, 400)
            return self._json({"ok": True, "references": refs})
        if path == "/api/make":
            text = (body.get("text") or "").strip()
            if not text:
                return self._json({"error": "enter a prompt or a script"}, 400)
            name = slugify(body.get("name") or text[:40])
            args = ["make", "--name", name, "--lang", str(body.get("lang") or "en")]
            if body.get("mode") == "script":
                tmp = Path(tempfile.mkdtemp(prefix="aivs-")) / "script.txt"
                tmp.write_text(text, encoding="utf-8")
                args += ["--script", str(tmp)]
            else:
                args += ["--prompt", text, "--seconds", str(int(body.get("seconds") or 40))]
            args += _option_args(body)
            if body.get("style"):
                args += ["--style", str(body["style"])]
            try:
                n_chars = max(0, min(int(body.get("characters") or 0), 12))
            except (TypeError, ValueError):
                return self._json({"error": "characters must be a number"}, 400)
            if n_chars:
                args += ["--characters", str(n_chars)]
            if body.get("narrator") and n_chars:
                args += ["--narrator"]
            voices = [{"name": str(it.get("name") or "").strip(), "voice": str(it.get("voice") or "").strip()[:300],
                       "look": str(it.get("description") or "").strip()[:300]}
                      for it in body.get("references") or []
                      if isinstance(it, dict) and it.get("kind") == "person" and str(it.get("voice") or "").strip()]
            if voices:  # voice descriptions given for uploaded people -> character definitions
                cast_file = Path(tempfile.mkdtemp(prefix="aivs-")) / "cast.json"
                cast_file.write_text(json.dumps(voices, ensure_ascii=False), encoding="utf-8")
                args += ["--cast", str(cast_file)]
            items = body.get("references") or []
            if items:
                try:
                    specs = [_spec_from(it) for it in items if isinstance(it, dict)]
                except ValueError as exc:
                    return self._json({"error": str(exc)}, 400)
                refs_file = Path(tempfile.mkdtemp(prefix="aivs-")) / "references.json"
                refs_file.write_text(json.dumps(specs, ensure_ascii=False), encoding="utf-8")
                args += ["--references", str(refs_file)]
            return self._json(_start(args, name))
        m = re.match(r"^/api/project/([^/]+)/render$", path)
        if m and SAFE.match(m.group(1)):
            d = config.PROJECTS / m.group(1)
            proj = read_json(d / "project.json")
            if not proj:
                return self._json({"error": "not found"}, 404)
            if isinstance(body.get("shots"), list):  # storyboard edits from the UI
                by_id = {s["id"]: s for s in proj["storyboard"]["shots"]}
                valid_refs = {r["id"] for r in proj["storyboard"].get("references") or []}
                for s in body["shots"]:
                    cur = by_id.get(s.get("id"))
                    if cur:
                        for k in ("narration", "visual", "motion"):
                            if isinstance(s.get(k), str) and s[k].strip():
                                cur[k] = s[k].strip()
                        if isinstance(s.get("refs"), list):
                            cur["refs"] = [r for r in s["refs"] if r in valid_refs]
                        if s.get("animate") in ("wan", "parallax", "camera"):
                            cur["animate"] = s["animate"]
                        elif "animate" in s:
                            cur.pop("animate", None)
                        speakers = {"narrator"} | {c["id"] for c in proj["storyboard"].get("cast") or []}
                        if s.get("speaker") in speakers:
                            cur["speaker"] = s["speaker"]
                        if isinstance(s.get("on_screen"), bool):
                            cur["on_screen"] = s["on_screen"]
                if isinstance(body.get("cast"), list):  # character edits: look and voice description
                    by_cid = {c["id"]: c for c in proj["storyboard"].get("cast") or []}
                    for c in body["cast"]:
                        cur = by_cid.get(c.get("id")) if isinstance(c, dict) else None
                        if cur:
                            for k, n in (("look", 300), ("voice", 300)):
                                if isinstance(c.get(k), str):
                                    cur[k] = " ".join(c[k].split())[:n]
                if isinstance(body.get("style"), str) and body["style"].strip():
                    proj["storyboard"]["style"] = body["style"].strip()
                try:
                    from .. import safety
                    safety.check(proj["storyboard"])
                except SystemExit as exc:
                    return self._json({"error": str(exc)}, 400)
                proj["request"] = {"edited": True}
                write_json(d / "project.json", proj)
            return self._json(_start(["render", str(d), *_option_args(body)], m.group(1)))
        if path == "/api/cancel":
            proc = JOB["proc"]
            if proc is not None and proc.poll() is None:
                proc.terminate()
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)


def _option_args(body: dict) -> list[str]:
    args = []
    if body.get("aspect") in config.ASPECTS:
        args += ["--aspect", body["aspect"]]
    if body.get("music"):
        args += ["--music", str(body["music"])]
    if body.get("captions") in ("burn", "srt", "off"):
        args += ["--captions", body["captions"]]
    if body.get("voice"):
        args += ["--voice", str(body["voice"])]
    if body.get("motion") in ("wan", "parallax", "camera"):
        args += ["--motion", body["motion"]]
    if body.get("draft"):
        args += ["--draft"]
    if body.get("lipsync") in ("auto", "off"):
        args += ["--lipsync", body["lipsync"]]
    return args


def serve(host: str = "127.0.0.1", port: int = 8760) -> None:
    os.chdir(config.ROOT)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"AI Video Studio running at http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        proc = JOB["proc"]
        if proc is not None and proc.poll() is None:
            proc.terminate()
