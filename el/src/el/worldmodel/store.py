"""Persistence wrapper for the world model.

Provides a single ``WorldModelStore`` with lazy load/save tied to
el's state directory.
"""
from __future__ import annotations

from pathlib import Path

from .hdc import HDC
from .planner import ActiveInferencePlanner
from .world import WorldModel


class WorldModelStore:
    """Thin facade: HDC + WorldModel + Planner with disk persistence."""

    FILENAME = "worldmodel.npz"

    def __init__(self, state_dir: Path, dim: int = 10_000, capacity: int = 4096):
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / self.FILENAME
        self.hdc = HDC(dim=dim)
        self.world = WorldModel(hdc=self.hdc, capacity=capacity)
        self.planner = ActiveInferencePlanner(world=self.world, hdc=self.hdc)
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            self.world.load(self.path)
        except Exception:
            # Corrupt or schema-mismatched file: start fresh, do not crash.
            self.world.reset()
        self._loaded = True

    def save(self) -> None:
        self.world.save(self.path)

    def stats(self) -> dict:
        s = self.world.stats()
        s["path"] = str(self.path)
        return s
