"""Seed dataset assembly for the action transformer.

Three seed streams (each optional at runtime):

1. NL2Bash subset — `data/nl2bash.jsonl` if present: lines of
   `{"nl": ..., "bash": ...}`. The loader converts each to a synthetic
   `(intent, [Action(sh, cmd=bash)], reward=0.7)` triple.
2. tldr-pages — a small curated subset of tldr command examples bundled at
   `seed_data/tldr_subset.jsonl`.
3. Synthesized man pages — `seed_data/man_synth.jsonl`.

Plus the skill registry's accumulated real rows (exported via
`el export-training`) mix into the same format.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from ..intent import Intent
from ..primitives import Action


BUNDLED_DIR = Path(__file__).resolve().parents[2].parent / "seed_data"


@dataclass
class SeedExample:
    intent: Intent
    actions: tuple[Action, ...]
    reward: float
    source: str = "seed"

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "reward": self.reward,
            "source": self.source,
        }


def dump_jsonl(examples: Iterable[SeedExample], path: Path) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: Path) -> list[SeedExample]:
    out: list[SeedExample] = []
    if not Path(path).exists():
        return out
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            intent = Intent.from_dict(row["intent"])
            actions = tuple(Action.from_dict(a) for a in row.get("actions") or [])
            out.append(
                SeedExample(
                    intent=intent,
                    actions=actions,
                    reward=float(row.get("reward", 0.0)),
                    source=row.get("source", "seed"),
                )
            )
    return out


def nl2bash_to_example(nl: str, bash: str) -> SeedExample:
    intent = Intent(verb="run", obj="", scope="", args=(("query", nl),), raw=nl, confidence=0.5)
    actions = (Action.make("sh", cmd=bash, timeout=20),)
    return SeedExample(intent=intent, actions=actions, reward=0.7, source="nl2bash")


def build_training_examples(
    *,
    extra_paths: Iterable[Path] = (),
    include_bundled: bool = True,
) -> list[SeedExample]:
    examples: list[SeedExample] = []
    if include_bundled:
        for name in ("core_skills.jsonl", "tldr_subset.jsonl", "man_synth.jsonl"):
            p = BUNDLED_DIR / name
            examples.extend(load_jsonl(p))
    for p in extra_paths:
        p = Path(p)
        if p.suffix == ".jsonl":
            if p.name.startswith("nl2bash"):
                examples.extend(_load_nl2bash_jsonl(p))
            else:
                examples.extend(load_jsonl(p))
    return examples


def _load_nl2bash_jsonl(path: Path) -> Iterator[SeedExample]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            nl = row.get("nl") or row.get("invocation") or ""
            bash = row.get("bash") or row.get("cmd") or ""
            if nl and bash:
                yield nl2bash_to_example(nl, bash)
