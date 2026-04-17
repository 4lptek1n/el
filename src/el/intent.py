"""Intent: the deterministic structured form of a user command.

An Intent is produced by the parser from free-text input, consumed by the
executor, and also used as the key into the skill registry. Intents are
designed to be JSON-serializable so they can be tokenized into the action
transformer's input stream and logged to events.jsonl.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Intent:
    verb: str
    obj: str = ""
    scope: str = ""
    args: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    raw: str = ""
    confidence: float = 1.0

    def arg(self, key: str, default: str = "") -> str:
        for k, v in self.args:
            if k == key:
                return v
        return default

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["args"] = [list(pair) for pair in self.args]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def canonical_key(self) -> str:
        return f"{self.verb}|{self.obj}|{self.scope}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Intent":
        args = tuple((str(k), str(v)) for k, v in (d.get("args") or []))
        return cls(
            verb=d["verb"],
            obj=d.get("obj", ""),
            scope=d.get("scope", ""),
            args=args,
            raw=d.get("raw", ""),
            confidence=float(d.get("confidence", 1.0)),
        )
