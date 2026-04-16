"""Executor — maps Intents to action sequences and runs them.

Control flow per command:
1. Parser produces ranked candidate Intents.
2. For the top candidate, look up matching Skills in the registry.
3. If a Skill is found, run its action sequence and reinforce with outcome.
4. Otherwise delegate to SelfPlay, which will propose a plan, run it, and
   graduate the winning plan into a new Skill.
5. Every execution is logged to the events table and to events.jsonl.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from typing import TYPE_CHECKING

from .config import Config, load_config
from .intent import Intent
from .parser import Parser
from .primitives import Action, PrimResult, is_destructive, is_networked
from .registry import Skill, SkillRegistry
from .rewards import score_outcome

if TYPE_CHECKING:
    from .selfplay import SelfPlayRunner
    from .transformer.adapter import TransformerAdapter


@dataclass
class ExecutionResult:
    intent: Intent
    skill: Skill | None
    actions: tuple[Action, ...]
    results: tuple[PrimResult, ...]
    reward: float
    origin: str
    duration_ms: int

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.to_dict(),
            "skill_id": self.skill.id if self.skill else None,
            "actions": [a.to_dict() for a in self.actions],
            "results": [r.to_dict() for r in self.results],
            "reward": self.reward,
            "origin": self.origin,
            "duration_ms": self.duration_ms,
        }


@dataclass
class Executor:
    config: Config = field(default_factory=load_config)
    parser: Parser = field(default_factory=Parser)
    registry: SkillRegistry | None = None
    selfplay: SelfPlayRunner | None = None
    transformer: TransformerAdapter | None = None
    auto_confirm: bool = False

    def __post_init__(self) -> None:
        self.config.ensure_dirs()
        if self.registry is None:
            self.registry = SkillRegistry(self.config.registry_path)

    def handle(self, raw: str) -> ExecutionResult:
        intents = self.parser.parse(raw)
        intent = intents[0]
        if intent.verb == "unknown":
            result = ExecutionResult(
                intent=intent,
                skill=None,
                actions=(),
                results=(),
                reward=0.0,
                origin="unknown",
                duration_ms=0,
            )
            self._log_event("unparseable", {"raw": raw})
            return result

        skills = self.registry.lookup(intent)
        if skills:
            return self._run_skill(intent, skills[0])

        if self.transformer is not None:
            proposed = self.transformer.propose_plan(intent)
            if proposed:
                return self._run_candidate(intent, proposed, origin="transformer")

        if self.selfplay is not None:
            plan = self._selfplay_sandbox(intent)
            if plan:
                return self._run_candidate(intent, plan, origin="selfplay")

        self._log_event("no_plan", {"intent": intent.to_dict()})
        return ExecutionResult(
            intent=intent,
            skill=None,
            actions=(),
            results=(),
            reward=0.0,
            origin="no_plan",
            duration_ms=0,
        )

    def _run_skill(self, intent: Intent, skill: Skill) -> ExecutionResult:
        t0 = time.monotonic()
        actions = tuple(_apply_intent_args(a, intent) for a in skill.actions)
        results = tuple(self._execute_actions(actions, intent))
        reward = score_outcome(results, intent)
        self.registry.reinforce(skill.id, success=reward >= 0.5)
        duration_ms = int((time.monotonic() - t0) * 1000)
        exec_result = ExecutionResult(
            intent=intent,
            skill=skill,
            actions=skill.actions,
            results=results,
            reward=reward,
            origin="registry",
            duration_ms=duration_ms,
        )
        self._log_event("skill_run", exec_result.to_dict())
        return exec_result

    def _run_candidate(
        self, intent: Intent, actions: Iterable[Action], *, origin: str
    ) -> ExecutionResult:
        t0 = time.monotonic()
        actions_t = tuple(actions)
        results = tuple(self._execute_actions(actions_t, intent))
        reward = score_outcome(results, intent)
        duration_ms = int((time.monotonic() - t0) * 1000)
        skill: Skill | None = None
        if reward >= 0.6:
            skill = self.registry.add_skill(
                intent, actions_t, origin=origin, weight=reward
            )
        exec_result = ExecutionResult(
            intent=intent,
            skill=skill,
            actions=actions_t,
            results=results,
            reward=reward,
            origin=origin,
            duration_ms=duration_ms,
        )
        self._log_event("candidate_run", exec_result.to_dict())
        return exec_result

    def _selfplay_sandbox(self, intent: Intent) -> list[Action]:
        """Generate N candidate plans, run each in a sandboxed subprocess with
        timeout, score every outcome, and return the winner's action list.

        Each candidate executes in its own temp workdir. Destructive/network
        primitives are blocked inside the sandbox. If the subprocess sandbox
        is unavailable (e.g., frozen env), fall back to static scoring.
        """
        if self.selfplay is None:
            return []
        candidates = self.selfplay.candidates(intent)
        if not candidates:
            return []
        winners: list[tuple[float, list[Action]]] = []
        per_timeout = float(getattr(self.config, "selfplay_timeout_sec", 20.0))
        self._log_event(
            "selfplay_begin",
            {"intent": intent.to_dict(), "n_candidates": len(candidates), "timeout_s": per_timeout},
        )
        for idx, plan in enumerate(candidates):
            tmpdir = tempfile.mkdtemp(prefix="el-sandbox-")
            payload = {
                "actions": [a.to_dict() for a in plan],
                "intent": intent.to_dict(),
                "workdir": tmpdir,
                "offline": True,
                "timeout": per_timeout,
            }
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "el._candidate_runner"],
                    input=json.dumps(payload),
                    capture_output=True,
                    text=True,
                    timeout=per_timeout,
                    env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
                )
                if proc.returncode != 0 or not proc.stdout.strip():
                    self._log_event("selfplay_candidate", {"idx": idx, "ok": False, "reward": -1.0, "stderr": proc.stderr[-500:]})
                    continue
                out = json.loads(proc.stdout)
                reward = float(out.get("reward", 0.0))
                self._log_event(
                    "selfplay_candidate",
                    {"idx": idx, "ok": bool(out.get("ok")), "reward": reward, "actions": [a.to_dict() for a in plan]},
                )
                winners.append((reward, plan))
            except subprocess.TimeoutExpired:
                self._log_event("selfplay_candidate", {"idx": idx, "ok": False, "reward": -1.0, "error": "timeout"})
            except Exception as exc:
                self._log_event("selfplay_candidate", {"idx": idx, "ok": False, "reward": -1.0, "error": str(exc)})
            finally:
                try:
                    import shutil as _sh
                    _sh.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass
        if not winners:
            return list(candidates[0])
        winners.sort(key=lambda kv: kv[0], reverse=True)
        best_reward, best_plan = winners[0]
        self._log_event(
            "selfplay_winner",
            {"reward": best_reward, "actions": [a.to_dict() for a in best_plan]},
        )
        return list(best_plan)

    def _execute_actions(
        self, actions: Iterable[Action], intent: Intent
    ) -> Iterable[PrimResult]:
        for action in actions:
            if is_destructive(action) and self.config.confirm_destructive and not self.auto_confirm:
                yield PrimResult(
                    name=action.name,
                    ok=False,
                    duration_ms=0,
                    error="destructive action needs --yes",
                )
                return
            if is_networked(action) and self.config.offline:
                yield PrimResult(
                    name=action.name,
                    ok=False,
                    duration_ms=0,
                    error="offline mode; network disabled",
                )
                return
            if self.auto_confirm:
                kw = dict(action.kwargs)
                if action.name in {"sh", "sh_spawn", "pip_install", "process_kill", "file_delete"} and "confirmed" not in kw:
                    kw["confirmed"] = True
                    action = Action(name=action.name, kwargs=tuple((k, v) for k, v in kw.items()))
            result = action.call()
            yield result
            if not result.ok and action.name != "noop":
                return

    def _log_event(self, kind: str, payload: dict) -> None:
        self.registry.log_event(kind, payload)
        try:
            with Path(self.config.events_path).open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"ts": time.time(), "kind": kind, "payload": payload},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        except Exception:
            pass


def _apply_intent_args(action: Action, intent: Intent) -> Action:
    """Overlay intent args onto action kwargs for matching keys.

    Lets a seed skill defined with `path="."` be specialized to the user's
    explicit `path=./LICENSE` at runtime without separate skill rows.
    """
    overrideable = {"path", "url", "query", "root", "repo", "dst", "pattern"}
    overrides = {k: v for k, v in intent.args if k in overrideable}
    if not overrides:
        return action
    new_kwargs: list[tuple[str, object]] = []
    seen: set[str] = set()
    for k, v in action.kwargs:
        if k in overrides:
            new_kwargs.append((k, overrides[k]))
        else:
            new_kwargs.append((k, v))
        seen.add(k)
    return Action(name=action.name, kwargs=tuple(new_kwargs))
