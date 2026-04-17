"""Self-play planner.

When the skill registry has no match for an Intent, SelfPlay proposes N
candidate action sequences by composing primitives according to
verb-specific templates. Each candidate is scored by a dry pre-check
(static feasibility) and the top candidate is returned to the executor for
real execution.

This is intentionally simple: the point is to bootstrap the skill registry
cheaply. As the registry grows, self-play fires less often; as the action
transformer trains on accumulated triples, its proposals displace static
templates.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .intent import Intent
from .primitives import Action


PlanBuilder = Callable[[Intent, random.Random], list[Action]]


def _scope_path(intent: Intent) -> str:
    path = intent.arg("path")
    if path:
        return path
    if intent.scope == "this_folder":
        return "."
    if intent.scope == "downloads":
        return "~/Downloads"
    if intent.scope == "home":
        return "~"
    return "."


def _summarize_plan(intent: Intent, rng: random.Random) -> list[Action]:
    target = _scope_path(intent)
    url = intent.arg("url")
    out_path = intent.arg("out") or f"{target}/summary.md" if target != "." else "summary.md"
    if url:
        return [
            Action.make("http_get", url=url),
            Action.make("summarize_text", text="", max_lines=30),
            Action.make("file_write", path=out_path, content="# summary (stub)\n", overwrite=True),
        ]
    if intent.obj == "pdf":
        return [
            Action.make("file_list", path=target, pattern="*.pdf"),
            Action.make("noop"),
            Action.make("file_write", path=out_path, content="# PDF summary (stub)\n", overwrite=True),
        ]
    return [
        Action.make("file_list", path=target),
        Action.make("file_write", path=out_path, content="# summary (stub)\n", overwrite=True),
    ]


def _organize_plan(intent: Intent, rng: random.Random) -> list[Action]:
    target = _scope_path(intent)
    return [
        Action.make("file_list", path=target),
        Action.make("mkdir", path=f"{target}/_by_type"),
    ]


def _download_plan(intent: Intent, rng: random.Random) -> list[Action]:
    url = intent.arg("url")
    if not url:
        return []
    out = intent.arg("path") or f"./downloads/{url.rsplit('/', 1)[-1] or 'index.html'}"
    return [Action.make("http_download", url=url, dst=out)]


def _research_plan(intent: Intent, rng: random.Random) -> list[Action]:
    url = intent.arg("url") or "https://example.com/"
    query = intent.arg("query") or intent.obj or "topic"
    out = f"./research-{_slug(query)}.md"
    return [
        Action.make("http_get", url=url),
        Action.make("summarize_text", text="", max_lines=40),
        Action.make("file_write", path=out, content=f"# research: {query}\n", overwrite=True),
    ]


def _list_plan(intent: Intent, rng: random.Random) -> list[Action]:
    return [Action.make("file_list", path=_scope_path(intent))]


def _find_plan(intent: Intent, rng: random.Random) -> list[Action]:
    query = intent.arg("query") or intent.obj or ""
    return [Action.make("file_search", root=_scope_path(intent), query=query)]


def _disk_plan(intent: Intent, rng: random.Random) -> list[Action]:
    return [Action.make("disk_usage", path=_scope_path(intent))]


def _process_plan(intent: Intent, rng: random.Random) -> list[Action]:
    pattern = intent.arg("query") or intent.obj or None
    kwargs = {"pattern": pattern} if pattern else {}
    return [Action.make("process_list", **kwargs)]


def _git_plan(intent: Intent, rng: random.Random) -> list[Action]:
    repo = _scope_path(intent)
    if intent.verb == "git_status":
        return [Action.make("git_status", repo=repo)]
    return [Action.make("git_status", repo=repo), Action.make("git_log", repo=repo, n=10)]


def _run_plan(intent: Intent, rng: random.Random) -> list[Action]:
    cmd = intent.arg("query") or intent.arg("path") or ""
    if not cmd:
        return []
    return [Action.make("sh", cmd=cmd, timeout=20)]


def _inspect_plan(intent: Intent, rng: random.Random) -> list[Action]:
    path = intent.arg("path") or _scope_path(intent)
    return [Action.make("file_read", path=path, max_bytes=8_000)]


def _report_plan(intent: Intent, rng: random.Random) -> list[Action]:
    target = _scope_path(intent)
    out = f"./report-{_slug(target)}.md"
    return [
        Action.make("file_list", path=target),
        Action.make("disk_usage", path=target),
        Action.make(
            "file_write",
            path=out,
            content=f"# report: {target}\n\nsee registry events for details.\n",
            overwrite=True,
        ),
    ]


def _clean_plan(intent: Intent, rng: random.Random) -> list[Action]:
    target = _scope_path(intent)
    return [
        Action.make("file_list", path=target, pattern="*.tmp"),
    ]


def _help_plan(intent: Intent, rng: random.Random) -> list[Action]:
    return [Action.make("noop")]


def _now_plan(intent: Intent, rng: random.Random) -> list[Action]:
    return [Action.make("now_iso")]


DEFAULT_TEMPLATES: dict[str, PlanBuilder] = {
    "summarize": _summarize_plan,
    "organize": _organize_plan,
    "download": _download_plan,
    "research": _research_plan,
    "list": _list_plan,
    "find": _find_plan,
    "count": _list_plan,
    "disk": _disk_plan,
    "report": _report_plan,
    "clean": _clean_plan,
    "inspect": _inspect_plan,
    "extract": _inspect_plan,
    "git_status": _git_plan,
    "commit": _git_plan,
    "run": _run_plan,
    "build": _run_plan,
    "test": _run_plan,
    "lint": _run_plan,
    "help": _help_plan,
    "now": _now_plan,
    "process_list": _process_plan,
}


@dataclass
class SelfPlayRunner:
    seed: int = 0
    num_candidates: int = 5
    templates: dict[str, PlanBuilder] = field(default_factory=lambda: dict(DEFAULT_TEMPLATES))

    def candidates(self, intent: Intent) -> list[list[Action]]:
        """Return up to N distinct candidate plans for the given intent."""
        builder = self.templates.get(intent.verb)
        if builder is None:
            return []
        rng = random.Random(self.seed + hash(intent.canonical_key()) % 1_000_000)
        seen: set[tuple] = set()
        out: list[list[Action]] = []
        attempts = 0
        while len(out) < max(1, self.num_candidates) and attempts < self.num_candidates * 4:
            attempts += 1
            plan = builder(intent, rng)
            if not plan:
                continue
            key = tuple((a.name, tuple(sorted(a.kwargs))) for a in plan)
            if key in seen:
                plan = list(plan) + [Action.make("noop")]
                key = tuple((a.name, tuple(sorted(a.kwargs))) for a in plan)
                if key in seen:
                    continue
            seen.add(key)
            out.append(plan)
        return out

    def plan(self, intent: Intent) -> list[Action]:
        """Back-compat single-plan path used when subprocess sandbox is unavailable."""
        cands = self.candidates(intent)
        if not cands:
            return []
        return max(cands, key=lambda p: self._static_score(p, intent))

    def _static_score(self, actions: list[Action], intent: Intent) -> float:
        score = 1.0
        score -= 0.05 * max(0, len(actions) - 3)
        if any(a.name == "noop" for a in actions):
            score -= 0.2
        if any(a.name in {"http_download", "file_write"} for a in actions) and intent.verb in {
            "summarize", "download", "research", "report", "write_code"
        }:
            score += 0.3
        return score


SelfPlay = SelfPlayRunner


def _slug(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_/":
            out.append("-")
    slug = "".join(out).strip("-")
    return slug[:40] or "run"


def is_path_like(text: str) -> bool:
    return any(text.startswith(prefix) for prefix in ("./", "/", "~/"))


def safe_path(text: str) -> Path:
    return Path(text).expanduser()
