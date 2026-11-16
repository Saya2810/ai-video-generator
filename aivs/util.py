"""Small shared helpers: subprocess, hashing, ffprobe, worker launching."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from . import config


def log(msg: str) -> None:
    print(f"[aivs {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slugify(text: str, maxlen: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (text[:maxlen].rstrip("-")) or "project"


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def need(tool: str) -> str:
    exe = shutil.which(tool)
    if not exe:
        raise SystemExit(f"'{tool}' not found on PATH. Install it (macOS: brew install ffmpeg).")
    return exe


def run(cmd: list, log_path: Path | None = None, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    else:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    if check and proc.returncode != 0:
        tail = ""
        if log_path and log_path.exists():
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        elif proc.stdout:
            tail = "\n".join(proc.stdout.splitlines()[-25:])
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n{tail}")
    return proc


def run_worker(name: str, job: dict, workdir: Path) -> dict:
    """Run aivs.workers.<name> in a fresh process so only one model is ever resident in memory."""
    workdir.mkdir(parents=True, exist_ok=True)
    job_path = workdir / f"{name}.job.json"
    out_path = workdir / f"{name}.result.json"
    out_path.unlink(missing_ok=True)
    write_json(job_path, job)
    t0 = time.time()
    run([sys.executable, "-m", f"aivs.workers.{name}", job_path, out_path],
        log_path=workdir / f"{name}.log", env=config.worker_env())
    result = read_json(out_path)
    if result is None:
        raise RuntimeError(f"worker {name} produced no result; see {workdir / (name + '.log')}")
    result["_seconds"] = round(time.time() - t0, 1)
    return result


def probe(path: Path) -> dict:
    out = subprocess.run(
        [need("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def media_duration(path: Path) -> float:
    return float(probe(path)["format"]["duration"])
