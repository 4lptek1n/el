"""Skill registry — the symbolic half of dual-memory.

A SQLite-backed store of triples: (trigger_key, action_sequence, stats).
Local plasticity rule (STDP-inspired): on success the skill's weight
increments; on failure it decrements; unused skills slowly decay. No
gradient, no backprop.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .intent import Intent
from .primitives import Action


SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_key         TEXT NOT NULL,
    intent_json         TEXT NOT NULL,
    actions_json        TEXT NOT NULL,
    success_count       INTEGER NOT NULL DEFAULT 0,
    failure_count       INTEGER NOT NULL DEFAULT 0,
    weight              REAL NOT NULL DEFAULT 0.5,
    origin              TEXT NOT NULL DEFAULT 'seed',
    created_at          REAL NOT NULL,
    last_used_at        REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_skills_trigger ON skills (trigger_key);
CREATE INDEX IF NOT EXISTS idx_skills_weight ON skills (weight DESC);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    kind            TEXT NOT NULL,
    payload_json    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts);
"""


@dataclass
class Skill:
    id: int
    trigger_key: str
    intent: Intent
    actions: tuple[Action, ...]
    success_count: int
    failure_count: int
    weight: float
    origin: str
    created_at: float
    last_used_at: float

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total else 0.5


class SkillRegistry:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add_skill(
        self,
        intent: Intent,
        actions: Iterable[Action],
        *,
        origin: str = "seed",
        weight: float = 0.5,
    ) -> Skill:
        actions_tuple = tuple(actions)
        now = time.time()
        cur = self._conn.execute(
            """
            INSERT INTO skills (trigger_key, intent_json, actions_json, weight, origin, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                intent.canonical_key(),
                intent.to_json(),
                json.dumps([a.to_dict() for a in actions_tuple], ensure_ascii=False),
                weight,
                origin,
                now,
            ),
        )
        self._conn.commit()
        return self.get_skill(cur.lastrowid)

    def get_skill(self, skill_id: int) -> Skill:
        row = self._conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        return _row_to_skill(row)

    def has_exact_intent(self, intent: Intent) -> bool:
        """Exact trigger-key presence check (no verb-level fallback).

        Used by seeding to preserve all distinct same-verb variants.
        """
        row = self._conn.execute(
            "SELECT 1 FROM skills WHERE trigger_key = ? LIMIT 1",
            (intent.canonical_key(),),
        ).fetchone()
        return row is not None

    def lookup(self, intent: Intent, *, limit: int = 3) -> list[Skill]:
        rows = self._conn.execute(
            """
            SELECT * FROM skills
            WHERE trigger_key = ?
            ORDER BY weight DESC, last_used_at DESC
            LIMIT ?
            """,
            (intent.canonical_key(), limit),
        ).fetchall()
        if rows:
            return [_row_to_skill(r) for r in rows]
        rows = self._conn.execute(
            """
            SELECT * FROM skills
            WHERE trigger_key LIKE ?
            ORDER BY weight DESC
            LIMIT ?
            """,
            (f"{intent.verb}|%", limit),
        ).fetchall()
        return [_row_to_skill(r) for r in rows]

    def all_skills(self, *, limit: int = 1000) -> list[Skill]:
        rows = self._conn.execute(
            "SELECT * FROM skills ORDER BY weight DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_skill(r) for r in rows]

    def reinforce(self, skill_id: int, *, success: bool, lr: float = 0.15) -> Skill:
        skill = self.get_skill(skill_id)
        new_w = skill.weight + lr * (1.0 - skill.weight) if success else skill.weight - lr * skill.weight
        new_w = max(0.0, min(1.0, new_w))
        sc = skill.success_count + (1 if success else 0)
        fc = skill.failure_count + (0 if success else 1)
        self._conn.execute(
            """
            UPDATE skills
               SET success_count = ?, failure_count = ?, weight = ?, last_used_at = ?
             WHERE id = ?
            """,
            (sc, fc, new_w, time.time(), skill_id),
        )
        self._conn.commit()
        return self.get_skill(skill_id)

    def decay(self, *, factor: float = 0.99, floor: float = 0.0) -> int:
        self._conn.execute(
            "UPDATE skills SET weight = MAX(?, weight * ?)",
            (floor, factor),
        )
        cur = self._conn.execute(
            "DELETE FROM skills WHERE weight < 0.02 AND success_count = 0"
        )
        self._conn.commit()
        return cur.rowcount

    def decay_and_retire(
        self,
        *,
        factor: float = 0.99,
        floor: float = 0.0,
        retire_threshold: float = 0.1,
    ) -> dict:
        """Decay all skill weights and retire any whose weight falls below
        `retire_threshold` and whose net performance is negative.
        """
        before = self._conn.execute("SELECT COUNT(*) AS n FROM skills").fetchone()["n"]
        self._conn.execute(
            "UPDATE skills SET weight = MAX(?, weight * ?)",
            (floor, factor),
        )
        cur = self._conn.execute(
            "DELETE FROM skills WHERE weight < ? AND failure_count >= success_count",
            (retire_threshold,),
        )
        self._conn.commit()
        after = self._conn.execute("SELECT COUNT(*) AS n FROM skills").fetchone()["n"]
        return {
            "before": before,
            "after": after,
            "retired": cur.rowcount,
            "factor": factor,
            "retire_threshold": retire_threshold,
        }

    def log_event(self, kind: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT INTO events (ts, kind, payload_json) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(payload, ensure_ascii=False, default=str)),
        )
        self._conn.commit()

    def events(self, *, kind: str | None = None, limit: int = 500) -> list[dict]:
        if kind:
            rows = self._conn.execute(
                "SELECT ts, kind, payload_json FROM events WHERE kind = ? ORDER BY ts DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT ts, kind, payload_json FROM events ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            rec = {"ts": r["ts"], "kind": r["kind"]}
            try:
                rec["payload"] = json.loads(r["payload_json"])
            except Exception:
                rec["payload"] = {}
            out.append(rec)
        return out

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) AS n FROM skills").fetchone()["n"]
        top = self._conn.execute(
            "SELECT trigger_key, weight, success_count, failure_count FROM skills ORDER BY weight DESC LIMIT 10"
        ).fetchall()
        return {
            "count": total,
            "top": [
                {
                    "key": r["trigger_key"],
                    "weight": round(r["weight"], 3),
                    "success": r["success_count"],
                    "failure": r["failure_count"],
                }
                for r in top
            ],
        }

    def export_training_rows(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT intent_json, actions_json, weight, success_count, failure_count, origin FROM skills"
        ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "intent": json.loads(r["intent_json"]),
                    "actions": json.loads(r["actions_json"]),
                    "weight": r["weight"],
                    "reward": _reward_from_stats(r["success_count"], r["failure_count"]),
                    "origin": r["origin"],
                }
            )
        return out


def _row_to_skill(row: sqlite3.Row) -> Skill:
    intent = Intent.from_dict(json.loads(row["intent_json"]))
    actions = tuple(Action.from_dict(a) for a in json.loads(row["actions_json"]))
    return Skill(
        id=row["id"],
        trigger_key=row["trigger_key"],
        intent=intent,
        actions=actions,
        success_count=row["success_count"],
        failure_count=row["failure_count"],
        weight=row["weight"],
        origin=row["origin"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
    )


def _reward_from_stats(success: int, failure: int) -> float:
    total = success + failure
    if total == 0:
        return 0.0
    base = success / total
    confidence = 1.0 - math.exp(-total / 5.0)
    return round(base * confidence, 4)
