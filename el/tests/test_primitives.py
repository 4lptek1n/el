from pathlib import Path

from el.primitives import (
    Action,
    PRIMITIVES,
    disk_usage,
    file_append,
    file_list,
    file_read,
    file_write,
    grep,
    mkdir,
    noop,
    now_iso,
    sh,
    summarize_text,
)


def test_registry_has_primitives():
    for name in ("sh", "file_read", "file_write", "file_list", "http_get"):
        assert name in PRIMITIVES


def test_noop_ok():
    r = noop()
    assert r.ok and r.name == "noop"


def test_now_iso_ok():
    r = now_iso()
    assert r.ok and r.stdout


def test_file_write_then_read(tmp_path: Path):
    p = tmp_path / "a.txt"
    w = file_write(str(p), "hello", overwrite=True)
    assert w.ok
    r = file_read(str(p))
    assert r.ok and "hello" in r.stdout


def test_file_write_no_overwrite(tmp_path: Path):
    p = tmp_path / "a.txt"
    file_write(str(p), "hi", overwrite=True)
    assert not file_write(str(p), "bye").ok


def test_file_append(tmp_path: Path):
    p = tmp_path / "log.txt"
    file_append(str(p), "a\n")
    file_append(str(p), "b\n")
    assert p.read_text().splitlines() == ["a", "b"]


def test_file_list_and_grep(tmp_path: Path):
    (tmp_path / "x.md").write_text("TODO: fix me")
    (tmp_path / "y.py").write_text("print('ok')")
    listing = file_list(str(tmp_path))
    assert listing.ok and len(listing.data) == 2
    hits = grep(str(tmp_path), "TODO")
    assert hits.ok and any("x.md" in h["path"] for h in hits.data)


def test_disk_usage_here():
    r = disk_usage(".")
    assert r.ok and r.data["total"] > 0


def test_sh_echo():
    r = sh("echo hi", timeout=3)
    assert r.ok and "hi" in r.stdout


def test_mkdir_idempotent(tmp_path: Path):
    p = tmp_path / "nested" / "deep"
    assert mkdir(str(p)).ok
    assert p.exists()


def test_summarize_text():
    r = summarize_text("a\nb\nc\n")
    assert r.ok and "a" in r.stdout


def test_action_roundtrip():
    a = Action.make("file_write", path="x", content="y", overwrite=True)
    d = a.to_dict()
    b = Action.from_dict(d)
    assert a == b
