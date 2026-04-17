"""MultiModalSubstrate — pattern memory + sequence chain + persistence
on ONE shared Field with both C (Hebb) and SkipBank (long-range) channels.

The kızıl elma claim made physical: a single substrate object that
hosts pattern recall (via PatternMemory's C-channel writes), long-range
sequence chains (via SkipBank's nonlocal edges), and full save/load
persistence — all coexisting without one overwriting the other.

Three channels:
  - C (symmetric, local) — written by PatternMemory.store
  - B (directional, local) — touched by sequence STDP if used
  - SkipBank (sparse, long-range) — written by chain training

Save format: zip-style multi-file (.npz dir) holding pattern_memory.npz
and skip_bank.npz separately.
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from pathlib import Path
import numpy as np

from .field import Field, FieldConfig
from .pattern_memory import PatternMemory, Pattern
from .skip_bank import SkipBank


@dataclass
class MultiModalSubstrate:
    cfg: FieldConfig = dc_field(default_factory=FieldConfig)
    seed: int = 0
    K: int = 4
    md_min: int = 3
    long_decay: float = 0.95
    skip_eta: float = 0.05

    # Internal — built in __post_init__
    field: Field | None = None
    memory: PatternMemory | None = None
    bank: SkipBank | None = None

    def __post_init__(self) -> None:
        if self.field is None:
            self.field = Field(self.cfg, seed=self.seed)
        if self.memory is None:
            self.memory = PatternMemory(cfg=self.cfg, seed=self.seed,
                                        field=self.field)
        if self.bank is None:
            self.bank = SkipBank(self.cfg.rows, self.cfg.cols,
                                 K=self.K, md_min=self.md_min, seed=self.seed)
        self._E_long = np.zeros(self.cfg.rows * self.cfg.cols, dtype=np.float32)

    # ------------------------------------------------------------------
    # Pattern memory façade
    # ------------------------------------------------------------------
    def store_pattern(self, pattern: Pattern) -> None:
        self.memory.store(pattern)

    def recall(self, cue: Pattern):
        return self.memory.recall(cue)

    # ------------------------------------------------------------------
    # Sequence chain via SkipBank
    # ------------------------------------------------------------------
    def reset_runtime(self) -> None:
        self.field.reset_temp()
        self._E_long.fill(0)

    def step_with_skip(self) -> None:
        self.field.step()
        T_flat = self.field.T.reshape(-1)
        T_flat += self.bank.propagate(T_flat, eta=self.skip_eta)
        self._E_long = (self.long_decay * self._E_long
                        + (1.0 - self.long_decay) * np.abs(T_flat))

    def train_chain(self, anchors: list[tuple[int, int]], *,
                    n_epochs: int = 120, lr: float = 0.20,
                    hold: int = 4, gap: int = 3) -> None:
        for _ in range(n_epochs):
            for j in range(len(anchors) - 1):
                A, B = anchors[j], anchors[j + 1]
                self.reset_runtime()
                self.field.inject([A], [1.0])
                for _ in range(hold + gap):
                    self.step_with_skip()
                self.field.inject([B], [1.0])
                for _ in range(hold):
                    T_flat = self.field.T.reshape(-1)
                    self.bank.stdp_update(T_flat, self._E_long,
                                          lr=lr / hold)
                    self.step_with_skip()

    def probe_chain_link(self, A: tuple[int, int], B: tuple[int, int],
                         *, hold: int = 4, rd: int = 8) -> float:
        """Inject A, propagate, return T at B minus the same probe on a
        fresh substrate (empty bank)."""
        self.reset_runtime()
        self.field.inject([A], [1.0])
        for _ in range(hold + rd):
            self.step_with_skip()
        cue = float(self.field.T[B[0], B[1]])
        # baseline: empty bank, fresh field, same seed
        f0 = Field(self.cfg, seed=self.seed)
        empty = SkipBank(self.cfg.rows, self.cfg.cols, K=self.K,
                         md_min=self.md_min, seed=self.seed)
        f0.inject([A], [1.0])
        for _ in range(hold + rd):
            f0.step()
            T_flat = f0.T.reshape(-1)
            T_flat += empty.propagate(T_flat, eta=self.skip_eta)
        base = float(f0.T[B[0], B[1]])
        # restore field state for further use
        self.reset_runtime()
        return cue - base

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, dir_path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        self.memory.save(d / "pattern_memory.npz")
        self.bank.save(d / "skip_bank.npz")
        np.savez(d / "meta.npz",
                 K=self.K, md_min=self.md_min,
                 long_decay=self.long_decay, skip_eta=self.skip_eta,
                 seed=self.seed)

    @classmethod
    def load(cls, dir_path) -> "MultiModalSubstrate":
        d = Path(dir_path)
        meta = np.load(str(d / "meta.npz"))
        memory = PatternMemory.load(d / "pattern_memory.npz")
        bank = SkipBank.load(d / "skip_bank.npz")
        sub = cls.__new__(cls)
        sub.cfg = memory.cfg
        sub.seed = int(meta["seed"])
        sub.K = int(meta["K"])
        sub.md_min = int(meta["md_min"])
        sub.long_decay = float(meta["long_decay"])
        sub.skip_eta = float(meta["skip_eta"])
        sub.memory = memory
        sub.field = memory.field
        sub.bank = bank
        sub._E_long = np.zeros(sub.cfg.rows * sub.cfg.cols, dtype=np.float32)
        return sub
