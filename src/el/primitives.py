"""OS primitives — the effector layer of el.

Each primitive is a narrow function with a timeout and a documented effect.
Primitives are composed by the executor into action sequences. Destructive
primitives (delete, install, overwrite) respect Config.confirm_destructive.

Every primitive returns a PrimResult so outcomes are uniformly logged and
fed into the training corpus for the action transformer.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request as urlrequest


@dataclass
class PrimResult:
    name: str
    ok: bool
    duration_ms: int
    stdout: str = ""
    stderr: str = ""
    data: Any = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "stdout": self.stdout[:2000],
            "stderr": self.stderr[:2000],
            "data": self.data if _json_safe(self.data) else str(self.data)[:2000],
            "error": self.error,
        }


def _json_safe(x: Any) -> bool:
    try:
        json.dumps(x)
        return True
    except Exception:
        return False


def _timed(fn: Callable[[], PrimResult]) -> PrimResult:
    t0 = time.monotonic()
    try:
        r = fn()
    except Exception as exc:
        r = PrimResult(name="<unknown>", ok=False, duration_ms=0, error=f"{type(exc).__name__}: {exc}")
    r.duration_ms = int((time.monotonic() - t0) * 1000)
    return r


DANGEROUS_SH_TOKENS: tuple[str, ...] = (
    "rm -rf", "rm -fr", "rm -r ", " rm ", "mkfs", "dd if=", "dd of=",
    ":(){ :|:& };:", "shutdown", "reboot", " halt",
    "pip install", "pip3 install", "conda install", "apt-get install",
    "apt install", "brew install", "npm install -g", "curl | sh",
    "curl | bash", "wget | sh", "wget | bash", "chmod -R 777", "chown -R",
    "> /dev/sda", "mv /", "sudo ",
)


def sh_looks_dangerous(cmd: str) -> bool:
    c = " " + cmd.lower() + " "
    return any(tok in c for tok in DANGEROUS_SH_TOKENS)


def sh_looks_networked(cmd: str) -> bool:
    c = cmd.lower()
    return any(tok in c for tok in ("curl ", "wget ", "git clone ", "git pull", "git fetch", "git push", "pip install", "npm install", "yarn install", "pnpm install"))


def sh(cmd: str, *, timeout: float = 30.0, cwd: str | None = None, confirmed: bool = False) -> PrimResult:
    def run() -> PrimResult:
        if sh_looks_dangerous(cmd) and not confirmed:
            return PrimResult(
                name="sh",
                ok=False,
                duration_ms=0,
                error=f"refusing destructive shell: {cmd!r}; pass confirmed=True or --yes",
            )
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return PrimResult(
            name="sh",
            ok=proc.returncode == 0,
            duration_ms=0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            data={"returncode": proc.returncode, "cmd": cmd},
        )
    return _timed(run)


def sh_spawn(cmd: str, *, cwd: str | None = None) -> PrimResult:
    """Start a background subprocess and return its pid without waiting."""
    def run() -> PrimResult:
        if sh_looks_dangerous(cmd):
            return PrimResult(name="sh_spawn", ok=False, duration_ms=0, error="refused destructive")
        proc = subprocess.Popen(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return PrimResult(name="sh_spawn", ok=True, duration_ms=0, data={"pid": proc.pid, "cmd": cmd})
    return _timed(run)


def process_kill(pid: int, *, confirmed: bool = False) -> PrimResult:
    def run() -> PrimResult:
        if not confirmed:
            return PrimResult(name="process_kill", ok=False, duration_ms=0, error="confirmation required")
        try:
            os.kill(int(pid), 15)
            return PrimResult(name="process_kill", ok=True, duration_ms=0, data={"pid": int(pid), "signal": 15})
        except ProcessLookupError:
            return PrimResult(name="process_kill", ok=True, duration_ms=0, data={"pid": int(pid), "already_gone": True})
    return _timed(run)


def which(binary: str) -> PrimResult:
    def run() -> PrimResult:
        loc = shutil.which(binary)
        return PrimResult(name="which", ok=loc is not None, duration_ms=0, stdout=loc or "", data={"path": loc})
    return _timed(run)


def pip_install(package: str, *, confirmed: bool = False, timeout: float = 300.0) -> PrimResult:
    def run() -> PrimResult:
        if not confirmed:
            return PrimResult(name="pip_install", ok=False, duration_ms=0, error="confirmation required")
        proc = subprocess.run(
            ["pip", "install", package],
            capture_output=True, text=True, timeout=timeout,
        )
        return PrimResult(
            name="pip_install",
            ok=proc.returncode == 0,
            duration_ms=0,
            stdout=proc.stdout[-2000:], stderr=proc.stderr[-2000:],
            data={"package": package, "returncode": proc.returncode},
        )
    return _timed(run)


def file_read(path: str, *, max_bytes: int = 200_000) -> PrimResult:
    def run() -> PrimResult:
        p = Path(path).expanduser()
        data = p.read_bytes()[:max_bytes]
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = "<binary>"
        return PrimResult(name="file_read", ok=True, duration_ms=0, stdout=text, data={"size": len(data)})
    return _timed(run)


def file_write(path: str, content: str, *, overwrite: bool = False) -> PrimResult:
    def run() -> PrimResult:
        p = Path(path).expanduser()
        if p.exists() and not overwrite:
            return PrimResult(name="file_write", ok=False, duration_ms=0, error=f"exists: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return PrimResult(name="file_write", ok=True, duration_ms=0, data={"path": str(p), "bytes": len(content)})
    return _timed(run)


def file_append(path: str, content: str) -> PrimResult:
    def run() -> PrimResult:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return PrimResult(name="file_append", ok=True, duration_ms=0, data={"path": str(p)})
    return _timed(run)


def file_list(path: str = ".", *, pattern: str | None = None) -> PrimResult:
    def run() -> PrimResult:
        p = Path(path).expanduser()
        if not p.exists():
            return PrimResult(name="file_list", ok=False, duration_ms=0, error=f"missing: {p}")
        entries = []
        it = p.rglob(pattern) if pattern else p.iterdir()
        for e in it:
            entries.append(
                {
                    "name": e.name,
                    "path": str(e),
                    "is_dir": e.is_dir(),
                    "size": e.stat().st_size if e.is_file() else None,
                }
            )
        return PrimResult(name="file_list", ok=True, duration_ms=0, data=entries[:500])
    return _timed(run)


def file_move(src: str, dst: str) -> PrimResult:
    def run() -> PrimResult:
        s, d = Path(src).expanduser(), Path(dst).expanduser()
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return PrimResult(name="file_move", ok=True, duration_ms=0, data={"from": str(s), "to": str(d)})
    return _timed(run)


def file_copy(src: str, dst: str) -> PrimResult:
    def run() -> PrimResult:
        s, d = Path(src).expanduser(), Path(dst).expanduser()
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(str(s), str(d), dirs_exist_ok=True)
        else:
            shutil.copy2(str(s), str(d))
        return PrimResult(name="file_copy", ok=True, duration_ms=0, data={"from": str(s), "to": str(d)})
    return _timed(run)


def file_delete(path: str, *, confirmed: bool = False) -> PrimResult:
    def run() -> PrimResult:
        if not confirmed:
            return PrimResult(name="file_delete", ok=False, duration_ms=0, error="confirmation required")
        p = Path(path).expanduser()
        if not p.exists():
            return PrimResult(name="file_delete", ok=True, duration_ms=0, data={"missing": str(p)})
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return PrimResult(name="file_delete", ok=True, duration_ms=0, data={"path": str(p)})
    return _timed(run)


def mkdir(path: str) -> PrimResult:
    def run() -> PrimResult:
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)
        return PrimResult(name="mkdir", ok=True, duration_ms=0, data={"path": path})
    return _timed(run)


def http_get(url: str, *, timeout: float = 15.0, max_bytes: int = 1_000_000) -> PrimResult:
    def run() -> PrimResult:
        try:
            with urlrequest.urlopen(url, timeout=timeout) as resp:
                data = resp.read(max_bytes)
                status = resp.status if hasattr(resp, "status") else 200
                ctype = resp.headers.get("Content-Type", "")
        except urlerror.URLError as exc:
            return PrimResult(name="http_get", ok=False, duration_ms=0, error=f"url_error: {exc.reason}")
        except Exception as exc:
            return PrimResult(name="http_get", ok=False, duration_ms=0, error=f"{type(exc).__name__}: {exc}")
        try:
            body = data.decode("utf-8", errors="replace")
        except Exception:
            body = "<binary>"
        return PrimResult(
            name="http_get",
            ok=200 <= status < 400,
            duration_ms=0,
            stdout=body[:max_bytes],
            data={"status": status, "content_type": ctype, "bytes": len(data), "url": url},
        )
    return _timed(run)


def http_download(url: str, dst: str, *, timeout: float = 60.0) -> PrimResult:
    def run() -> PrimResult:
        p = Path(dst).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlrequest.urlopen(url, timeout=timeout) as resp, p.open("wb") as out:
                shutil.copyfileobj(resp, out)
        except Exception as exc:
            return PrimResult(name="http_download", ok=False, duration_ms=0, error=f"{type(exc).__name__}: {exc}")
        return PrimResult(name="http_download", ok=True, duration_ms=0, data={"path": str(p), "size": p.stat().st_size})
    return _timed(run)


def file_search(root: str, query: str, *, max_hits: int = 200) -> PrimResult:
    def run() -> PrimResult:
        r = Path(root).expanduser()
        if not r.exists():
            return PrimResult(name="file_search", ok=False, duration_ms=0, error=f"missing: {r}")
        hits = []
        needle = query.lower()
        for p in r.rglob("*"):
            if p.is_file() and needle in p.name.lower():
                hits.append(str(p))
                if len(hits) >= max_hits:
                    break
        return PrimResult(name="file_search", ok=True, duration_ms=0, data=hits)
    return _timed(run)


def grep(root: str, pattern: str, *, ext: str | None = None, max_hits: int = 100) -> PrimResult:
    import re as _re

    def run() -> PrimResult:
        r = Path(root).expanduser()
        if not r.exists():
            return PrimResult(name="grep", ok=False, duration_ms=0, error=f"missing: {r}")
        rx = _re.compile(pattern, _re.IGNORECASE)
        hits = []
        for p in r.rglob("*"):
            if not p.is_file():
                continue
            if ext and p.suffix.lstrip(".") != ext.lstrip("."):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append({"path": str(p), "line": i, "text": line[:200]})
                    if len(hits) >= max_hits:
                        return PrimResult(name="grep", ok=True, duration_ms=0, data=hits)
        return PrimResult(name="grep", ok=True, duration_ms=0, data=hits)
    return _timed(run)


def disk_usage(path: str = ".") -> PrimResult:
    def run() -> PrimResult:
        usage = shutil.disk_usage(str(Path(path).expanduser()))
        total, used, free = usage.total, usage.used, usage.free
        return PrimResult(
            name="disk_usage",
            ok=True,
            duration_ms=0,
            data={"total": total, "used": used, "free": free, "pct_used": round(used / total * 100, 2)},
        )
    return _timed(run)


def process_list(*, pattern: str | None = None) -> PrimResult:
    def run() -> PrimResult:
        out = subprocess.run(["ps", "-eo", "pid,ppid,etime,comm"], capture_output=True, text=True, timeout=5)
        lines = out.stdout.splitlines()
        if pattern:
            lines = [lines[0]] + [line for line in lines[1:] if pattern.lower() in line.lower()]
        return PrimResult(name="process_list", ok=out.returncode == 0, duration_ms=0, stdout="\n".join(lines[:200]))
    return _timed(run)


def pdf_to_text(path: str) -> PrimResult:
    def run() -> PrimResult:
        p = Path(path).expanduser()
        if not p.exists():
            return PrimResult(name="pdf_to_text", ok=False, duration_ms=0, error=f"missing: {p}")
        if shutil.which("pdftotext"):
            out = subprocess.run(
                ["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True, timeout=60
            )
            return PrimResult(name="pdf_to_text", ok=out.returncode == 0, duration_ms=0, stdout=out.stdout)
        try:
            raw = p.read_bytes()
            text_parts = []
            for chunk in raw.split(b"BT")[1:]:
                end = chunk.find(b"ET")
                if end > 0:
                    segment = chunk[:end]
                    for line in segment.split(b"\n"):
                        if b"Tj" in line or b"TJ" in line:
                            piece = line.split(b"(")
                            for frag in piece[1:]:
                                close = frag.find(b")")
                                if close > 0:
                                    text_parts.append(frag[:close].decode("latin-1", errors="ignore"))
            text = " ".join(text_parts)
            return PrimResult(name="pdf_to_text", ok=bool(text), duration_ms=0, stdout=text or "<empty>")
        except Exception as exc:
            return PrimResult(name="pdf_to_text", ok=False, duration_ms=0, error=str(exc))
    return _timed(run)


def git_status(repo: str = ".") -> PrimResult:
    def run() -> PrimResult:
        out = subprocess.run(
            ["git", "-C", str(Path(repo).expanduser()), "status", "--porcelain", "-b"],
            capture_output=True, text=True, timeout=10,
        )
        return PrimResult(name="git_status", ok=out.returncode == 0, duration_ms=0, stdout=out.stdout)
    return _timed(run)


def git_log(repo: str = ".", *, n: int = 20) -> PrimResult:
    def run() -> PrimResult:
        out = subprocess.run(
            ["git", "-C", str(Path(repo).expanduser()), "log", f"-n{n}", "--oneline", "--decorate"],
            capture_output=True, text=True, timeout=10,
        )
        return PrimResult(name="git_log", ok=out.returncode == 0, duration_ms=0, stdout=out.stdout)
    return _timed(run)


def env_get(key: str) -> PrimResult:
    def run() -> PrimResult:
        val = os.environ.get(key, "")
        return PrimResult(name="env_get", ok=bool(val), duration_ms=0, data={"key": key, "present": bool(val)})
    return _timed(run)


def summarize_text(text: str, *, max_lines: int = 20, max_chars: int = 4000) -> PrimResult:
    def run() -> PrimResult:
        if not text:
            return PrimResult(name="summarize_text", ok=False, duration_ms=0, error="empty")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        picked: list[str] = []
        total = 0
        for ln in lines:
            if total + len(ln) > max_chars:
                break
            picked.append(ln)
            total += len(ln)
            if len(picked) >= max_lines:
                break
        summary = "\n".join(picked)
        return PrimResult(name="summarize_text", ok=True, duration_ms=0, stdout=summary, data={"chars": len(summary)})
    return _timed(run)


def clipboard_write(text: str) -> PrimResult:
    def run() -> PrimResult:
        for cmd in (["pbcopy"], ["xclip", "-selection", "clipboard"], ["wl-copy"]):
            if shutil.which(cmd[0]):
                proc = subprocess.run(cmd, input=text, text=True, timeout=5)
                return PrimResult(
                    name="clipboard_write",
                    ok=proc.returncode == 0,
                    duration_ms=0,
                    data={"tool": cmd[0], "chars": len(text)},
                )
        return PrimResult(name="clipboard_write", ok=False, duration_ms=0, error="no clipboard tool")
    return _timed(run)


def now_iso() -> PrimResult:
    def run() -> PrimResult:
        from datetime import datetime, timezone

        return PrimResult(
            name="now_iso",
            ok=True,
            duration_ms=0,
            stdout=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    return _timed(run)


def noop() -> PrimResult:
    return PrimResult(name="noop", ok=True, duration_ms=0)


def http_post(url: str, body: str = "", *, timeout: float = 15.0, content_type: str = "application/json") -> PrimResult:
    def run() -> PrimResult:
        req = urlrequest.Request(url, data=body.encode("utf-8"), method="POST", headers={"Content-Type": content_type, "User-Agent": "el/0.1"})
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                data = resp.read(1_000_000).decode("utf-8", errors="replace")
                return PrimResult(name="http_post", ok=True, duration_ms=0, stdout=data, data={"status": resp.status, "url": url})
        except urlerror.HTTPError as exc:
            return PrimResult(name="http_post", ok=False, duration_ms=0, error=f"HTTP {exc.code}")
        except Exception as exc:
            return PrimResult(name="http_post", ok=False, duration_ms=0, error=str(exc))
    return _timed(run)


def clipboard_read() -> PrimResult:
    def run() -> PrimResult:
        for cmd in (["xclip", "-selection", "clipboard", "-o"], ["pbpaste"], ["wl-paste"]):
            if shutil.which(cmd[0]):
                try:
                    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                    if out.returncode == 0:
                        return PrimResult(name="clipboard_read", ok=True, duration_ms=0, stdout=out.stdout, data={"len": len(out.stdout)})
                except Exception:
                    continue
        return PrimResult(name="clipboard_read", ok=False, duration_ms=0, error="no clipboard backend")
    return _timed(run)


def env_set(key: str, value: str) -> PrimResult:
    def run() -> PrimResult:
        os.environ[key] = value
        return PrimResult(name="env_set", ok=True, duration_ms=0, data={"key": key})
    return _timed(run)


def env_list(*, prefix: str | None = None) -> PrimResult:
    def run() -> PrimResult:
        keys = [k for k in os.environ if (prefix is None or k.startswith(prefix))]
        keys.sort()
        return PrimResult(name="env_list", ok=True, duration_ms=0, data=keys)
    return _timed(run)


def hash_file(path: str, *, algo: str = "sha256") -> PrimResult:
    import hashlib

    def run() -> PrimResult:
        p = Path(path).expanduser()
        if not p.exists():
            return PrimResult(name="hash_file", ok=False, duration_ms=0, error=f"missing: {p}")
        h = hashlib.new(algo)
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        digest = h.hexdigest()
        return PrimResult(name="hash_file", ok=True, duration_ms=0, stdout=digest, data={"algo": algo, "digest": digest})
    return _timed(run)


def b64_encode(text: str) -> PrimResult:
    import base64

    def run() -> PrimResult:
        out = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return PrimResult(name="b64_encode", ok=True, duration_ms=0, stdout=out)
    return _timed(run)


def b64_decode(text: str) -> PrimResult:
    import base64

    def run() -> PrimResult:
        try:
            out = base64.b64decode(text.encode("ascii")).decode("utf-8", errors="replace")
            return PrimResult(name="b64_decode", ok=True, duration_ms=0, stdout=out)
        except Exception as exc:
            return PrimResult(name="b64_decode", ok=False, duration_ms=0, error=str(exc))
    return _timed(run)


def regex_match(text: str, pattern: str) -> PrimResult:
    import re

    def run() -> PrimResult:
        try:
            matches = re.findall(pattern, text)
            return PrimResult(name="regex_match", ok=True, duration_ms=0, stdout="\n".join(str(m) for m in matches[:200]), data={"count": len(matches)})
        except re.error as exc:
            return PrimResult(name="regex_match", ok=False, duration_ms=0, error=str(exc))
    return _timed(run)


def json_parse(text: str) -> PrimResult:
    def run() -> PrimResult:
        try:
            data = json.loads(text)
            return PrimResult(name="json_parse", ok=True, duration_ms=0, data=data)
        except Exception as exc:
            return PrimResult(name="json_parse", ok=False, duration_ms=0, error=str(exc))
    return _timed(run)


def csv_parse(path: str, *, max_rows: int = 500) -> PrimResult:
    import csv

    def run() -> PrimResult:
        p = Path(path).expanduser()
        if not p.exists():
            return PrimResult(name="csv_parse", ok=False, duration_ms=0, error=f"missing: {p}")
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)
        return PrimResult(name="csv_parse", ok=True, duration_ms=0, data={"rows": rows, "count": len(rows)})
    return _timed(run)


def archive_create(src: str, dst: str, *, fmt: str = "zip") -> PrimResult:
    def run() -> PrimResult:
        s = Path(src).expanduser()
        base = Path(dst).expanduser()
        archived = shutil.make_archive(str(base), fmt, root_dir=str(s.parent), base_dir=str(s.name))
        return PrimResult(name="archive_create", ok=True, duration_ms=0, data={"archive": archived})
    return _timed(run)


def archive_extract(src: str, dst: str) -> PrimResult:
    def run() -> PrimResult:
        s = Path(src).expanduser()
        d = Path(dst).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(s), str(d))
        return PrimResult(name="archive_extract", ok=True, duration_ms=0, data={"dst": str(d)})
    return _timed(run)


def ffmpeg_probe(path: str, *, timeout: float = 15.0) -> PrimResult:
    def run() -> PrimResult:
        if shutil.which("ffprobe") is None:
            return PrimResult(name="ffmpeg_probe", ok=False, duration_ms=0, error="ffprobe not installed")
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
            capture_output=True, text=True, timeout=timeout,
        )
        ok = proc.returncode == 0
        try:
            data = json.loads(proc.stdout) if ok else None
        except Exception:
            data = None
        return PrimResult(name="ffmpeg_probe", ok=ok, duration_ms=0, stdout=proc.stdout[:2000], stderr=proc.stderr[:1000], data=data)
    return _timed(run)


def ocr_image(path: str, *, timeout: float = 30.0) -> PrimResult:
    def run() -> PrimResult:
        if shutil.which("tesseract") is None:
            return PrimResult(name="ocr_image", ok=False, duration_ms=0, error="tesseract not installed")
        proc = subprocess.run(
            ["tesseract", path, "-", "-l", "eng"],
            capture_output=True, text=True, timeout=timeout,
        )
        return PrimResult(
            name="ocr_image",
            ok=proc.returncode == 0,
            duration_ms=0,
            stdout=proc.stdout, stderr=proc.stderr,
        )
    return _timed(run)


def index_search(root: str, query: str, *, max_hits: int = 200) -> PrimResult:
    def run() -> PrimResult:
        r = Path(root).expanduser()
        q = query.lower()
        hits = []
        for p in r.rglob("*"):
            if len(hits) >= max_hits:
                break
            if q in p.name.lower():
                hits.append({"path": str(p), "name": p.name})
        return PrimResult(name="index_search", ok=True, duration_ms=0, data=hits)
    return _timed(run)


def head(path: str, *, n: int = 20) -> PrimResult:
    def run() -> PrimResult:
        p = Path(path).expanduser()
        if not p.exists():
            return PrimResult(name="head", ok=False, duration_ms=0, error=f"missing: {p}")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[:n]
        return PrimResult(name="head", ok=True, duration_ms=0, stdout="\n".join(lines), data={"count": len(lines)})
    return _timed(run)


def tail(path: str, *, n: int = 20) -> PrimResult:
    def run() -> PrimResult:
        p = Path(path).expanduser()
        if not p.exists():
            return PrimResult(name="tail", ok=False, duration_ms=0, error=f"missing: {p}")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        return PrimResult(name="tail", ok=True, duration_ms=0, stdout="\n".join(lines), data={"count": len(lines)})
    return _timed(run)


def wc_lines(path: str) -> PrimResult:
    def run() -> PrimResult:
        p = Path(path).expanduser()
        if not p.exists():
            return PrimResult(name="wc_lines", ok=False, duration_ms=0, error=f"missing: {p}")
        with p.open("rb") as fh:
            n = sum(1 for _ in fh)
        return PrimResult(name="wc_lines", ok=True, duration_ms=0, stdout=str(n), data={"lines": n})
    return _timed(run)


def uname() -> PrimResult:
    def run() -> PrimResult:
        import platform
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        }
        return PrimResult(name="uname", ok=True, duration_ms=0, stdout=json.dumps(info), data=info)
    return _timed(run)


def cpu_info() -> PrimResult:
    def run() -> PrimResult:
        info = {
            "count": os.cpu_count(),
            "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
        }
        return PrimResult(name="cpu_info", ok=True, duration_ms=0, stdout=json.dumps(info), data=info)
    return _timed(run)


PRIMITIVES: dict[str, Callable[..., PrimResult]] = {
    "sh": sh,
    "sh_spawn": sh_spawn,
    "which": which,
    "pip_install": pip_install,
    "process_kill": process_kill,
    "file_read": file_read,
    "file_write": file_write,
    "file_append": file_append,
    "file_list": file_list,
    "file_move": file_move,
    "file_copy": file_copy,
    "file_delete": file_delete,
    "mkdir": mkdir,
    "http_get": http_get,
    "http_post": http_post,
    "http_download": http_download,
    "file_search": file_search,
    "index_search": index_search,
    "grep": grep,
    "head": head,
    "tail": tail,
    "wc_lines": wc_lines,
    "disk_usage": disk_usage,
    "process_list": process_list,
    "pdf_to_text": pdf_to_text,
    "git_status": git_status,
    "git_log": git_log,
    "env_get": env_get,
    "env_set": env_set,
    "env_list": env_list,
    "summarize_text": summarize_text,
    "clipboard_write": clipboard_write,
    "clipboard_read": clipboard_read,
    "now_iso": now_iso,
    "hash_file": hash_file,
    "b64_encode": b64_encode,
    "b64_decode": b64_decode,
    "regex_match": regex_match,
    "json_parse": json_parse,
    "csv_parse": csv_parse,
    "archive_create": archive_create,
    "archive_extract": archive_extract,
    "ffmpeg_probe": ffmpeg_probe,
    "ocr_image": ocr_image,
    "uname": uname,
    "cpu_info": cpu_info,
    "noop": noop,
}


DESTRUCTIVE: frozenset[str] = frozenset({
    "file_delete", "file_move", "pip_install", "process_kill",
    "archive_extract", "env_set",
})
NETWORK: frozenset[str] = frozenset({"http_get", "http_post", "http_download", "pip_install"})


@dataclass(frozen=True)
class Action:
    name: str
    kwargs: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def call(self) -> PrimResult:
        fn = PRIMITIVES.get(self.name)
        if fn is None:
            return PrimResult(name=self.name, ok=False, duration_ms=0, error=f"unknown primitive: {self.name}")
        kwargs = dict(self.kwargs)
        return fn(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kwargs": [list(p) for p in self.kwargs]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Action":
        kwargs = tuple((str(k), v) for k, v in (d.get("kwargs") or []))
        return cls(name=d["name"], kwargs=kwargs)

    @classmethod
    def make(cls, name: str, **kwargs: Any) -> "Action":
        return cls(name=name, kwargs=tuple((k, v) for k, v in kwargs.items()))


def is_destructive(action: Action) -> bool:
    if action.name in DESTRUCTIVE:
        return True
    if action.name in {"sh", "sh_spawn"}:
        cmd = dict(action.kwargs).get("cmd", "")
        if isinstance(cmd, str) and sh_looks_dangerous(cmd):
            return True
    return False


def is_networked(action: Action) -> bool:
    if action.name in NETWORK:
        return True
    if action.name in {"sh", "sh_spawn"}:
        cmd = dict(action.kwargs).get("cmd", "")
        if isinstance(cmd, str) and sh_looks_networked(cmd):
            return True
    return False
