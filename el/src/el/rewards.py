"""Reward model for action outcomes.

Three signals, combined into a scalar in [0, 1]:

- automatic: exit codes, output presence, duration, primitive ok-flag
- heuristic: intent-specific checks (e.g. summarize must produce a file)
- explicit: user thumbs-up/down via `el rate`

Rewards feed the skill registry's local plasticity update and the action
transformer's training labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .intent import Intent
from .primitives import PrimResult


@dataclass(frozen=True)
class Outcome:
    results: tuple[PrimResult, ...]
    intent: Intent

    @property
    def all_ok(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)

    @property
    def any_output(self) -> bool:
        return any(bool(r.stdout.strip()) or bool(r.data) for r in self.results)

    @property
    def total_ms(self) -> int:
        return sum(r.duration_ms for r in self.results)


def score_outcome(results: Iterable[PrimResult], intent: Intent) -> float:
    results_t = tuple(results)
    if not results_t:
        return 0.0
    outcome = Outcome(results_t, intent)
    auto = _auto_score(outcome)
    heur = _heuristic_score(outcome)
    return round(min(1.0, 0.7 * auto + 0.3 * heur), 4)


def _auto_score(o: Outcome) -> float:
    ok_ratio = sum(1 for r in o.results if r.ok) / len(o.results)
    output_bonus = 0.1 if o.any_output else 0.0
    speed_bonus = 0.1 if o.total_ms < 2000 else 0.0
    return min(1.0, ok_ratio + output_bonus + speed_bonus)


def _heuristic_score(o: Outcome) -> float:
    verb = o.intent.verb
    produced_file = any(
        r.name in {"file_write", "file_append", "http_download", "file_copy", "file_move"} and r.ok
        for r in o.results
    )
    produced_text = any(r.stdout.strip() for r in o.results)
    if verb in {"summarize", "report", "write_code", "make_image", "extract", "convert", "download"}:
        return 1.0 if produced_file else (0.3 if produced_text else 0.0)
    if verb in {"list", "find", "inspect", "count", "grep", "git_status", "research"}:
        return 1.0 if produced_text else 0.0
    if verb in {"organize", "move", "copy", "delete", "clean", "rename"}:
        return 1.0 if any(r.ok for r in o.results) else 0.0
    if verb in {"run", "build", "test", "lint"}:
        return 1.0 if all(r.ok for r in o.results) else 0.0
    return 0.5
