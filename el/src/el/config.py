"""Runtime configuration for el: state dir, timeouts, sandbox limits."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_STATE_DIR = Path(os.environ.get("EL_STATE_DIR", str(Path.home() / ".el_state")))


@dataclass(frozen=True)
class Config:
    state_dir: Path = DEFAULT_STATE_DIR
    registry_path: Path = field(init=False)
    events_path: Path = field(init=False)
    logs_dir: Path = field(init=False)
    sandbox_dir: Path = field(init=False)

    primitive_timeout_sec: float = 30.0
    selfplay_candidates: int = 5
    selfplay_timeout_sec: float = 45.0
    max_skill_actions: int = 12

    confirm_destructive: bool = True
    offline: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "registry_path", self.state_dir / "registry.sqlite3")
        object.__setattr__(self, "events_path", self.state_dir / "events.jsonl")
        object.__setattr__(self, "logs_dir", self.state_dir / "logs")
        object.__setattr__(self, "sandbox_dir", self.state_dir / "sandbox")

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)


def load_config(**overrides: object) -> Config:
    base = Config()
    if not overrides:
        return base
    kwargs = {
        "state_dir": base.state_dir,
        "primitive_timeout_sec": base.primitive_timeout_sec,
        "selfplay_candidates": base.selfplay_candidates,
        "selfplay_timeout_sec": base.selfplay_timeout_sec,
        "max_skill_actions": base.max_skill_actions,
        "confirm_destructive": base.confirm_destructive,
        "offline": base.offline,
    }
    kwargs.update(overrides)
    return Config(**kwargs)
