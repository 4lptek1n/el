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
from .plasticity import covariance_update, hebbian_update


Pattern = list[tuple[int, int]]  # list of (row, col) active cells


@dataclass
class PatternMemory:
    cfg: FieldConfig = dc_field(default_factory=FieldConfig)
    seed: int = 0
    write_steps: int = 6           # relaxation steps while clamping during write
    write_lr: float = 0.10         # Hebb LR during write
    # Decay applied to C during write (per Hebb step). Default 0.0 keeps
    # the legacy single-task tuning. Setting decay > 0 prevents C from
    # saturating at 1.0 everywhere, which is the root cause of multi-task
    # interference: a saturated C floods diffusion uniformly and washes
    # out the directional B-bias that sequence STDP produced. Non-zero
    # decay keeps C at a moderate level so the B-bias still steers heat,
    # at the cost of slightly noisier pattern recall.
    write_decay: float = 0.0
    recall_steps: int = 8          # relaxation steps after cue injection
    # Inhibition / WTA — set wta_k > 0 to enable competitive recall.
    # When enabled, after every relaxation step the substrate keeps the
    # top-k hottest cells and decays the rest by `wta_suppression`.
    wta_k: int = 0
    wta_suppression: float = 0.5
    use_global_gain: bool = False
    target_mean: float = 0.05
    # Plasticity rule: "hebb" (default, positive-only) or "covariance"
    # (Hebb + anti-Hebb in one principled rule that decorrelates
    # non-coactive cells — should reduce the smear that plain Hebb
    # produces on a positive-only substrate).
    rule: str = "hebb"
    cov_baseline: float | None = None   # if None, per-step mean(T)
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
            if self.rule == "covariance":
                covariance_update(f, lr=self.write_lr, decay=self.write_decay,
                                  baseline=self.cov_baseline)
            else:
                hebbian_update(f, lr=self.write_lr, decay=self.write_decay)
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


    # ------------------------------------------------------------------
    # Persistence — Eşik 4: substrate state must outlive a single run
    # ------------------------------------------------------------------
    def save(self, path) -> None:
        """Serialize substrate state (C, B), stored patterns, and config to .npz.

        After save+load, the memory must produce the same recall accuracy
        as the original — the substrate state is the model.
        """
        from pathlib import Path
        f = self.field
        patterns_arr = np.array(
            [(i, r, c) for i, p in enumerate(self.patterns) for (r, c) in p],
            dtype=np.int32,
        ) if self.patterns else np.zeros((0, 3), dtype=np.int32)
        np.savez(
            str(Path(path)),
            cfg_rows=self.cfg.rows, cfg_cols=self.cfg.cols,
            cfg_diffusion_rate=self.cfg.diffusion_rate,
            cfg_decay=self.cfg.decay, cfg_n_steps=self.cfg.n_steps,
            cfg_nonlinear_alpha=self.cfg.nonlinear_alpha,
            seed=self.seed,
            write_steps=self.write_steps, write_lr=self.write_lr,
            write_decay=self.write_decay,
            recall_steps=self.recall_steps,
            wta_k=self.wta_k, wta_suppression=self.wta_suppression,
            use_global_gain=self.use_global_gain,
            target_mean=self.target_mean,
            rule=self.rule,
            cov_baseline=(self.cov_baseline if self.cov_baseline is not None
                          else np.nan),
            C_right=f.C_right, C_down=f.C_down,
            B_right=f.B_right, B_down=f.B_down,
            patterns=patterns_arr,
        )

    @classmethod
    def load(cls, path) -> "PatternMemory":
        """Reconstruct a PatternMemory from a .npz produced by save()."""
        from pathlib import Path
        d = np.load(str(Path(path)))
        cfg = FieldConfig(
            rows=int(d["cfg_rows"]), cols=int(d["cfg_cols"]),
            diffusion_rate=float(d["cfg_diffusion_rate"]),
            decay=float(d["cfg_decay"]),
            n_steps=int(d["cfg_n_steps"]),
            nonlinear_alpha=float(d["cfg_nonlinear_alpha"]),
        )
        cov_baseline = float(d["cov_baseline"])
        if np.isnan(cov_baseline):
            cov_baseline = None
        mem = cls(
            cfg=cfg, seed=int(d["seed"]),
            write_steps=int(d["write_steps"]),
            write_lr=float(d["write_lr"]),
            write_decay=float(d["write_decay"]) if "write_decay" in d.files else 0.0,
            recall_steps=int(d["recall_steps"]),
            wta_k=int(d["wta_k"]),
            wta_suppression=float(d["wta_suppression"]),
            use_global_gain=bool(d["use_global_gain"]),
            target_mean=float(d["target_mean"]),
            rule=str(d["rule"]),
            cov_baseline=cov_baseline,
        )
        mem.field.C_right[:] = d["C_right"]
        mem.field.C_down[:] = d["C_down"]
        mem.field.B_right[:] = d["B_right"]
        mem.field.B_down[:] = d["B_down"]
        # Reconstruct patterns list from flat (idx, r, c) array
        pat_arr = d["patterns"]
        if pat_arr.size > 0:
            from collections import defaultdict
            by_idx = defaultdict(list)
            for row in pat_arr:
                by_idx[int(row[0])].append((int(row[1]), int(row[2])))
            mem.patterns = [by_idx[i] for i in sorted(by_idx)]
        else:
            mem.patterns = []
        return mem


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
