"""Seed the skill registry from bundled corpora.

`bundled_skills()` returns the canonical starter skills — one or two per
verb — so that even a fresh install has a usable baseline registry before
self-play has fired a single time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..intent import Intent
from ..primitives import Action
from ..registry import SkillRegistry


SEED_DIR = Path(__file__).resolve().parents[3] / "seed_data"


def _intent(verb: str, *, obj: str = "", scope: str = "", args: tuple[tuple[str, str], ...] = ()) -> Intent:
    return Intent(verb=verb, obj=obj, scope=scope, args=args, raw="", confidence=1.0)


def bundled_skills() -> list[tuple[Intent, list[Action], str]]:
    s: list[tuple[Intent, list[Action], str]] = []

    s.append((
        _intent("list", scope="this_folder"),
        [Action.make("file_list", path=".")],
        "seed",
    ))
    s.append((
        _intent("list", obj="folder"),
        [Action.make("file_list", path=".")],
        "seed",
    ))

    s.append((
        _intent("find"),
        [Action.make("file_search", root=".", query="")],
        "seed",
    ))

    s.append((
        _intent("count", obj="file"),
        [Action.make("file_list", path=".")],
        "seed",
    ))

    s.append((
        _intent("disk", scope="this_folder"),
        [Action.make("disk_usage", path=".")],
        "seed",
    ))
    s.append((
        _intent("report", obj="disk"),
        [
            Action.make("disk_usage", path="."),
            Action.make("file_write", path="./disk-report.md", content="# disk\nsee stdout\n", overwrite=True),
        ],
        "seed",
    ))
    s.append((
        _intent("report", scope="this_folder"),
        [
            Action.make("file_list", path="."),
            Action.make("disk_usage", path="."),
            Action.make(
                "file_write",
                path="./report-cwd.md",
                content="# report\n\nsee events for details.\n",
                overwrite=True,
            ),
        ],
        "seed",
    ))

    s.append((
        _intent("summarize", scope="this_folder"),
        [
            Action.make("file_list", path="."),
            Action.make("file_write", path="./summary.md", content="# summary\n", overwrite=True),
        ],
        "seed",
    ))
    s.append((
        _intent("summarize", obj="pdf", scope="this_folder"),
        [
            Action.make("file_list", path=".", pattern="*.pdf"),
            Action.make("file_write", path="./pdf-summary.md", content="# pdf summary\n", overwrite=True),
        ],
        "seed",
    ))

    s.append((
        _intent("organize", scope="this_folder"),
        [
            Action.make("file_list", path="."),
            Action.make("mkdir", path="./_by_type"),
        ],
        "seed",
    ))
    s.append((
        _intent("organize", scope="downloads"),
        [
            Action.make("file_list", path="~/Downloads"),
            Action.make("mkdir", path="~/Downloads/_by_type"),
        ],
        "seed",
    ))

    s.append((
        _intent("clean", scope="this_folder"),
        [Action.make("file_list", path=".", pattern="*.tmp")],
        "seed",
    ))

    s.append((
        _intent("git_status"),
        [Action.make("git_status", repo=".")],
        "seed",
    ))
    s.append((
        _intent("git_status", scope="this_folder"),
        [Action.make("git_status", repo=".")],
        "seed",
    ))
    s.append((
        _intent("commit"),
        [Action.make("git_status", repo="."), Action.make("git_log", repo=".", n=5)],
        "seed",
    ))

    s.append((
        _intent("inspect"),
        [Action.make("file_read", path=".", max_bytes=8000)],
        "seed",
    ))

    s.append((
        _intent("help"),
        [Action.make("noop")],
        "seed",
    ))

    s.append((
        _intent("research"),
        [
            Action.make("http_get", url="https://example.com"),
            Action.make("summarize_text", text=""),
            Action.make("file_write", path="./research.md", content="# research\n", overwrite=True),
        ],
        "seed",
    ))

    s.append((
        _intent("download"),
        [Action.make("http_download", url="https://example.com", dst="./example.html")],
        "seed",
    ))

    return s


def seed_registry(
    registry: SkillRegistry,
    *,
    use_bundled: bool = True,
    extra_path: Path | None = None,
    extra_examples: Iterable[tuple[Intent, list[Action], str]] = (),
) -> dict:
    added = 0
    if use_bundled:
        for intent, actions, origin in bundled_skills():
            if registry.has_exact_intent(intent):
                continue
            registry.add_skill(intent, actions, origin=origin, weight=0.6)
            added += 1
        for name in ("core_skills.jsonl",):
            p = SEED_DIR / name
            if p.exists():
                for intent, actions, origin in _load_triples(p):
                    if registry.has_exact_intent(intent):
                        continue
                    registry.add_skill(intent, actions, origin=origin, weight=0.55)
                    added += 1
    if extra_path is not None and extra_path.exists():
        for intent, actions, origin in _load_triples(extra_path):
            registry.add_skill(intent, actions, origin=origin, weight=0.5)
            added += 1
    for intent, actions, origin in extra_examples:
        registry.add_skill(intent, actions, origin=origin, weight=0.5)
        added += 1
    stats = registry.stats()
    stats["added"] = added
    return stats


def _load_triples(path: Path) -> Iterable[tuple[Intent, list[Action], str]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            intent = Intent.from_dict(row["intent"])
            actions = [Action.from_dict(a) for a in row.get("actions", [])]
            yield intent, actions, row.get("source", "seed")
