"""Curated demo command catalog.

Each Demo has a natural-language command string and an optional online flag.
The CLI's `el demo --all` iterates over this list so reproducibility and CI
stay honest: the demo catalog IS the benchmark.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Demo:
    name: str
    command: str
    online: bool = False
    description: str = ""


DEMOS: tuple[Demo, ...] = (
    Demo(
        name="list_current_folder",
        command="list this folder",
        description="List entries in the current working directory.",
    ),
    Demo(
        name="find_readme",
        command='find "readme"',
        description="Substring-match against filenames under the current dir.",
    ),
    Demo(
        name="disk_report",
        command="report disk usage here",
        description="Write a short disk-usage report markdown.",
    ),
    Demo(
        name="summarize_current_folder",
        command="summarize this folder",
        description="Produce summary.md from the current dir listing.",
    ),
    Demo(
        name="organize_folder",
        command="organize this folder",
        description="Prepare _by_type subfolders.",
    ),
    Demo(
        name="git_status_here",
        command="git status",
        description="Report working-tree status of the current repo.",
    ),
    Demo(
        name="inspect_pyproject",
        command="inspect ./pyproject.toml",
        description="Read the first 8KB of pyproject.toml.",
    ),
    Demo(
        name="count_py_files",
        command="count files",
        description="List and count files in the current dir.",
    ),
    Demo(
        name="now",
        command="help",
        description="Help verb hits the noop plan; covers the unknown-fallback path.",
    ),
    Demo(
        name="clean_tmp",
        command="clean this folder",
        description="List *.tmp files (non-destructive preview).",
    ),
    Demo(
        name="grep_todo",
        command='find "todo"',
        description="File search for todo.",
    ),
    Demo(
        name="inspect_license",
        command="inspect ./LICENSE",
        description="Read the MIT license file.",
    ),
    Demo(
        name="list_src",
        command="list ./src",
        description="List the src directory if present.",
    ),
    Demo(
        name="report_cwd",
        command="report this folder",
        description="Produce a report-<slug>.md with listing + disk usage.",
    ),
    Demo(
        name="online_fetch_example",
        command="research https://example.com",
        online=True,
        description="Fetch a tiny public page and summarize it (requires network).",
    ),
)


def demo_by_name(name: str) -> Demo | None:
    for d in DEMOS:
        if d.name == name:
            return d
    return None


def ephemeral_workspace() -> Path:
    import subprocess

    tmp = Path(tempfile.mkdtemp(prefix="el-demo-"))
    (tmp / "README.md").write_text("# demo workspace\n", encoding="utf-8")
    (tmp / "note.md").write_text("todo: finish this\n", encoding="utf-8")
    (tmp / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (tmp / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp / "src").mkdir()
    (tmp / "src" / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp / "trash.tmp").write_text("junk\n", encoding="utf-8")
    try:
        subprocess.run(
            ["git", "init", "-q", "-b", "main"],
            cwd=tmp,
            check=False,
            timeout=5,
            capture_output=True,
        )
    except Exception:
        pass
    return tmp
