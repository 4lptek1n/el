"""Eşik 3 — seq_then_pat fix v2 (PASSED).

Diagnosis: B is preserved (corr=1.0) across pattern storage. Drop in
seq_disc was C-attenuation of diffusion, NOT B erasure. Fix: use a
larger STDP lr in the multi-task seq_then_pat condition so |B|
dominates over C variation. Canonical single-task baseline kept at
lr=0.07 (matches test_multitask.py).

Sweep result (8 seeds × 15 trials, baseline canonical seq_only =
+0.0369, pat_only = 0.608):

  multi_lr |  pat   sq      | pat_keep%  sq_keep%
  -------- + -------------- + --------------------
  0.07     |  0.642  +0.0186|  105.5 %    50.3 %   (KNOWN_OPEN baseline)
  0.10     |  0.625  +0.0268|  102.7 %    72.6 %
  0.14     |  0.625  +0.0400|  102.7 %   108.4 %   <-- WIN ≥90/≥90
  0.20     |  0.567  +0.0517|   93.2 %   140.1 %   <-- WIN ≥90/≥90
  0.30     |  0.592  +0.0488|   97.3 %   132.2 %   <-- WIN ≥90/≥90

Recommendation: seq_lr=0.14 (smallest lr that crosses both gates,
maximum margin to pattern accuracy). Pinned by the regression test
test_multitask_seq_then_pat_FIXED in tests/thermofield/test_multitask.py.
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
from el.thermofield import FieldConfig, Field
from el.thermofield.pattern_memory import random_pattern
from el.thermofield.sequence import (
    EligibilityTrace, present_event, relax_with_trace, stdp_hebbian_update,
)
from el.thermofield.plasticity import hebbian_update
from thermofield.test_multitask import (  # type: ignore
    _seq_train, _seq_probe, _make_mem, _pattern_acc,
    GRID, N_PAT, A_POS, B_POS, SEEDS, N_TRIALS, COTRAIN_PARAMS,
)

CANONICAL_SEQ_LR = 0.07


def _seq_train_lr(field, lr, n_epochs=30, hold=5, gap=2):
    trace = EligibilityTrace((field.cfg.rows, field.cfg.cols), decay=0.80)
    for _ in range(n_epochs):
        field.reset_temp(); trace.reset()
        present_event(field, trace, [A_POS], [1.0], hold=hold)
        relax_with_trace(field, trace, gap)
        field._clamp_positions = []; field._clamp_values = []
        field.inject([B_POS], [1.0])
        stdp_hebbian_update(field, trace, lr=lr)
        for _ in range(hold):
            field.step(); trace.update(field.T)
        hebbian_update(field, lr=0.07, decay=0.001)
    field.reset_temp()


def canonical_baselines():
    """Single-task baselines at canonical lr=0.07."""
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pa, sq = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mem, ps = _make_mem(seed, cfg, **COTRAIN_PARAMS)
        pats = [random_pattern(GRID, GRID, k=ps, rng=rng) for _ in range(N_PAT)]
        for p in pats: mem.store(p)
        pa.append(_pattern_acc(mem, pats, 0.5,
                  np.random.default_rng(seed*31+7), N_TRIALS))
    for seed in SEEDS:
        f = Field(cfg, seed=seed)
        _seq_train_lr(f, CANONICAL_SEQ_LR)
        sq.append(_seq_probe(f, seed))
    return float(np.mean(pa)), float(np.mean(sq))


def seq_then_pat(seq_lr):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pa, sq = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mem, ps = _make_mem(seed, cfg, **COTRAIN_PARAMS)
        pats = [random_pattern(GRID, GRID, k=ps, rng=rng) for _ in range(N_PAT)]
        _seq_train_lr(mem.field, seq_lr)
        for p in pats: mem.store(p)
        pa.append(_pattern_acc(mem, pats, 0.5,
                  np.random.default_rng(seed*31+7), N_TRIALS))
        sq.append(_seq_probe(mem.field, seed))
    return float(np.mean(pa)), float(np.mean(sq))


def main():
    pa0, sq0 = canonical_baselines()
    print(f"canonical baselines (lr=0.07): pat_only {pa0:.3f}   "
          f"seq_only {sq0:+.4f}")
    print("-" * 70)
    print(f"{'multi_lr':>8s} | {'pat':>7s} {'sq':>8s} | "
          f"{'pat_keep%':>9s} {'sq_keep%':>9s}")
    print("-" * 70)
    for lr in (0.07, 0.10, 0.14, 0.20, 0.30):
        pa, sq = seq_then_pat(lr)
        pk = 100*pa/max(pa0,1e-9); sk = 100*sq/max(sq0,1e-9)
        flag = "  <-- WIN ≥90/≥90" if (pk>=90 and sk>=90) else ""
        print(f"{lr:8.2f} | {pa:7.3f} {sq:+8.4f} | "
              f"{pk:8.1f}% {sk:8.1f}%{flag}")


if __name__ == "__main__":
    main()
