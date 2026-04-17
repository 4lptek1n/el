"""Associative pattern memory on the thermofield substrate.

The point of all the substrate machinery (C/B edges, STDP, vertical
coincidence) is to support real cognitive functions. Pattern
memory — store P spatial patterns, then recall the closest one from a
partial / noisy cue — is the simplest non-trivial function that
real associative memories (Hopfield nets, RRAM crossbars, cortical
microcircuits) are evaluated on.

Storage protocol (Hebb-style on top of the existing C grid):
  - Each pattern is a list of "active" cell indices on a Field.
  - To store, we clamp those cells hot and run a few relaxation steps
    while applying Hebbian potentiation to in-grid edges between
    co-active cells (the existing `hebbian_step_field` does exactly
    this on neighbour edges).

Recall protocol:
  - Inject a partial cue (subset of the original active cells), let
    the field relax for K steps with no plasticity, then read out the
    top-N hottest cells. Match against each stored pattern by Jaccard /
    overlap and return the best.

This is a substrate-native, backprop-free associative memory — the
software analogue of an RRAM crossbar that has been programmed by
local Hebbian writes.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Sequence

import numpy as np

from .field import Field, FieldConfig
from .inhibition import global_gain_step, kwta_step
from .plasticity import hebbian_update


Pattern = list[tuple[int, int]]  # list of (row, col) active cells


@dataclass
class PatternMemory:
    cfg: FieldConfig = dc_field(default_factory=FieldConfig)
    seed: int = 0
    write_steps: int = 6           # relaxation steps while clamping during write
    write_lr: float = 0.10         # Hebb LR during write
    recall_steps: int = 8          # relaxation steps after cue injection
    # Inhibition / WTA — set wta_k > 0 to enable competitive recall.
    # When enabled, after every relaxation step the substrate keeps the
    # top-k hottest cells and decays the rest by `wta_suppression`.
    wta_k: int = 0
    wta_suppression: float = 0.5
    use_global_gain: bool = False
    target_mean: float = 0.05
    field: Field | None = None
    patterns: list[Pattern] = dc_field(default_factory=list)

    def __post_init__(self) -> None:
        if self.field is None:
            self.field = Field(self.cfg, seed=self.seed)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def store(self, pattern: Pattern) -> None:
        """Imprint one pattern into the substrate via clamped Hebb.

        We clamp the pattern cells to 1.0 and let the field relax under
        Hebbian potentiation; cells that fire together strengthen the
        edges between them. After write we erase the residual heat so
        the next operation starts cold.
        """
        f = self.field
        f.reset_temp()
        f.inject(list(pattern), [1.0] * len(pattern))
        for _ in range(self.write_steps):
            f.step()
            # Apply inhibition during write too — so the Hebb step only
            # potentiates edges between cells that survived competition,
            # producing sparse stored patterns instead of dense smear.
            if self.wta_k > 0:
                T_flat = f.T.reshape(-1)
                kwta_step(T_flat, self.wta_k, self.wta_suppression)
            if self.use_global_gain:
                global_gain_step(f.T.reshape(-1), self.target_mean)
            hebbian_update(f, lr=self.write_lr, decay=0.0)
        f.reset_temp()
        # Persist the canonical pattern for later matching
        self.patterns.append(list(pattern))

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def recall(self, cue: Pattern, top_n: int | None = None) -> tuple[int, float, np.ndarray]:
        """Inject `cue`, relax, return (best_pattern_idx, overlap_score, hot_set).

        `top_n` defaults to the size of the largest stored pattern.
        Overlap score is Jaccard between hot_set and the matched stored
        pattern.
        """
        if not self.patterns:
            raise RuntimeError("no patterns stored")

        f = self.field
        f.reset_temp()
        # Crucially: inject the cue WITHOUT clamping. If we clamp, the
        # cue cells stay pinned hot and the top-N readout trivially
        # returns the cue itself — defeats the purpose of recall.
        # No clamp = the cue is an INITIAL CONDITION; the substrate
        # then decides what stays hot based on the stored Hebb edges.
        f.inject(list(cue), [1.0] * len(cue), clamp=False)
        for _ in range(self.recall_steps):
            f.step()
            if self.wta_k > 0:
                kwta_step(f.T.reshape(-1), self.wta_k, self.wta_suppression)
            if self.use_global_gain:
                global_gain_step(f.T.reshape(-1), self.target_mean)

        if top_n is None:
            top_n = max(len(p) for p in self.patterns)

        T = f.T.copy()
        flat = T.ravel()
        # Top-N hottest cells
        idx = np.argpartition(flat, -top_n)[-top_n:]
        hot = set(
            (int(i // self.cfg.cols), int(i % self.cfg.cols)) for i in idx
        )

        best_i, best_score = -1, -1.0
        for i, p in enumerate(self.patterns):
            ps = set(p)
            inter = len(hot & ps)
            union = len(hot | ps)
            score = inter / union if union else 0.0
            if score > best_score:
                best_score = score
                best_i = i

        f.reset_temp()
        return best_i, best_score, np.array(sorted(hot))


def random_pattern(rows: int, cols: int, k: int, rng: np.random.Generator) -> Pattern:
    """A random pattern of `k` distinct active cells on a rows×cols grid."""
    n = rows * cols
    idx = rng.choice(n, size=k, replace=False)
    return [(int(i // cols), int(i % cols)) for i in idx]


def corrupt(pattern: Pattern, drop_frac: float, rng: np.random.Generator) -> Pattern:
    """Drop a fraction of the pattern's active cells (partial cue)."""
    if drop_frac <= 0:
        return list(pattern)
    keep_n = max(1, int(round(len(pattern) * (1.0 - drop_frac))))
    keep_idx = rng.choice(len(pattern), size=keep_n, replace=False)
    return [pattern[i] for i in sorted(keep_idx)]
