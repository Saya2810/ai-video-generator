"""Resumable multi-connection model downloader (stdlib only).

Hugging Face's default client can stall on slow or throttled links. This fetches every file of a repo into
models/local/<org>--<name>/ using many HTTP range requests in parallel, resumes after interruption, and verifies
the SHA-256 of every LFS file against the hub metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config

CHUNK = 8 * 1024 * 1024
UA = {"User-Agent": "aivs-fetch/0.1"}
LOCAL = config.ROOT / "models" / "local"


# Files a model needs at runtime that live in a sibling repo: {repo: [(source_repo, filename)]}
# Chatterbox Multilingual ships no built-in voice; the default speaker conditionals (same MIT upstream,
# ResembleAI/chatterbox conds.pt) are published in MLX format with the English conversion.
EXTRA_FILES = {
    "mlx-community/chatterbox-multilingual-v3": [("mlx-community/chatterbox-fp16", "conds.safetensors")],
}


def local_dir(repo: str) -> Path:
    return LOCAL / repo.replace("/", "--")


def resolve_model(repo: str) -> str:
    """Return a local directory if the repo was fetched (or is a path), else the repo id (HF cache / auto-download)."""
    if os.path.isdir(repo):
        return repo
    d = local_dir(repo)
    return str(d) if (d / ".complete").exists() else repo


_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first(*args, **kwargs):
    """urllib has no happy-eyeballs: on networks with broken IPv6 every request would wait for a timeout first."""
    res = _orig_getaddrinfo(*args, **kwargs)
    return sorted(res, key=lambda r: r[0] != socket.AF_INET)


socket.getaddrinfo = _ipv4_first


def _get(url: str, headers: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def _file_list(repo: str) -> list[dict]:
    with _get(f"https://huggingface.co/api/models/{repo}?blobs=true") as r:
        meta = json.load(r)
    return [{"name": s["rfilename"], "size": s.get("size") or 0, "sha256": (s.get("lfs") or {}).get("sha256")}
            for s in meta.get("siblings", [])]


class _Progress:
    def __init__(self, total: int, done: int):
        self.total, self.done, self.lock, self.t0, self.start = total, done, threading.Lock(), time.time(), done
        self.last = 0.0

    def add(self, n: int):
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last > 5:
                self.last = now
                rate = (self.done - self.start) / max(now - self.t0, 1e-6) / 1e6
                eta = (self.total - self.done) / 1e6 / max(rate, 1e-6)
                print(f"  {self.done / 1e9:6.2f}/{self.total / 1e9:.2f} GB  {rate:5.1f} MB/s  eta {eta / 60:5.1f} min", flush=True)


class _File:
    def __init__(self, repo: str, f: dict, dest: Path):
        self.url = f"https://huggingface.co/{repo}/resolve/main/{f['name']}"
        self.name, self.size, self.sha, self.dest = f["name"], f["size"], f["sha256"], dest
        self.parts_path = dest.parent / (dest.name + ".parts")
        self.lock = threading.Lock()
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.complete = dest.exists() and dest.stat().st_size == self.size and not self.parts_path.exists()
        self.done = set()
        if not self.complete:
            if self.parts_path.exists() and dest.exists():
                self.done = set(json.loads(self.parts_path.read_text()))
            self.parts_path.write_text(json.dumps(sorted(self.done)))
            self.fd = os.open(dest, os.O_RDWR | os.O_CREAT)
            os.ftruncate(self.fd, self.size)
        self.offsets = list(range(0, self.size, CHUNK)) or [0]

    def part(self, off: int, prog: _Progress) -> None:
        end = min(off + CHUNK, self.size) - 1
        last = None
        for attempt in range(12):
            try:
                hdr = {"Range": f"bytes={off}-{end}"} if self.size else {}
                with _get(self.url, hdr) as r:
                    data = r.read()
                if self.size and len(data) != end - off + 1:
                    raise IOError(f"short read {len(data)}")
                os.pwrite(self.fd, data, off)
                with self.lock:
                    self.done.add(off)
                    self.parts_path.write_text(json.dumps(sorted(self.done)))
                prog.add(len(data))
                return
            except Exception as exc:  # noqa: BLE001 - any network hiccup (incl. IncompleteRead, SSL) is retried
                last = exc
                if attempt >= 2:
                    print(f"  retry {attempt + 1} for {self.name} @ {off}: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"giving up on {self.name} @ {off}: {last}")

    def finish(self) -> None:
        if self.complete:
            return
        os.close(self.fd)
        if self.sha:
            h = hashlib.sha256()
            with open(self.dest, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 22), b""):
                    h.update(block)
            if h.hexdigest() != self.sha:
                self.parts_path.unlink(missing_ok=True)
                self.dest.unlink()
                raise RuntimeError(f"checksum mismatch for {self.name} — deleted, run again")
        self.parts_path.unlink(missing_ok=True)


def fetch(repo: str, connections: int = 32, include: tuple[str, ...] = ()) -> Path:
    d = local_dir(repo)
    for src, name in EXTRA_FILES.get(repo, []):
        if not (d / name).exists():
            d.mkdir(parents=True, exist_ok=True)
            with _get(f"https://huggingface.co/{src}/resolve/main/{name}") as r:
                (d / name).write_bytes(r.read())
            print(f"{repo}: added {name} from {src}", flush=True)
    if (d / ".complete").exists():
        print(f"{repo}: already complete")
        return d
    meta = _file_list(repo)
    if include:
        meta = [f for f in meta if any(f["name"].startswith(p) for p in include)]
    total = sum(f["size"] for f in meta)
    print(f"{repo}: {len(meta)} files, {total / 1e9:.2f} GB -> {d}", flush=True)
    files = [_File(repo, f, d / f["name"]) for f in sorted(meta, key=lambda f: f["size"])]  # small files first
    already = sum(f.size for f in files if f.complete) + sum(
        min(CHUNK, f.size - o) for f in files if not f.complete for o in f.done)
    prog = _Progress(total, already)
    tasks = [(f, o) for f in files if not f.complete for o in f.offsets if o not in f.done]
    with ThreadPoolExecutor(connections) as pool:
        list(pool.map(lambda t: t[0].part(t[1], prog), tasks))
    for f in files:
        f.finish()
    (d / ".complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{repo}: done ({total / 1e9:.2f} GB, sha256 verified)", flush=True)
    return d


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = [config.MODELS[k] for k in ("planner", "image", "stt", "depth", "tts_chatterbox")]
    conns = int(os.environ.get("AIVS_FETCH_CONNECTIONS", "32"))
    for repo in argv:
        fetch(repo, conns)


if __name__ == "__main__":
    main()
