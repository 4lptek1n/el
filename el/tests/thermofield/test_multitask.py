"""Eşik 3 — multi-task substrate honest regression.

Same Field used by both PatternMemory (writes to C, symmetric Hebb) and
sequence STDP (writes to B, directional). What we measured:

  condition            | pattern_acc        | seq_discrim
  ---------------------|--------------------|-------------
  pattern_only         | 0.467 ± 0.128      | -0.005 ± 0.038 (no train)
  seq_only             |        —           | +0.037 ± 0.011  ← learned
  multi_pat_then_seq   | 0.517 ± 0.132      | -0.016 ± 0.014  ← seq broken
  multi_seq_then_pat   | 0.500 ± 0.147      | -0.020 ± 0.006  ← seq broken
  interleaved          | 0.533 ± 0.123      | -0.020 ± 0.006  ← seq broken

Interpretation:
  - Pattern memory survives co-training (no catastrophic forgetting on
    that side) — even improves slightly.
  - Sequence association is DESTROYED by pattern Hebb writes. Pattern
    Hebb is symmetric on C; this overwrites the directed B-bias signal
    that sequence STDP wrote, AND saturates C so generic A→B
    propagation becomes weaker than baseline.

This test pins down the asymmetric interference so we don't quietly
"forget" it later. It is NOT a green-light for multi-task — it
documents that Eşik 3 (multi-task substrate) is half-passed.
"""
from __future__ import annotations
import numpy as np
import pytest
from el.thermofield import FieldConfig, Field
from el.thermofield.pattern_memory import PatternMemory, random_pattern
from el.thermofield.sequence import (
    EligibilityTrace, present_event, relax_with_trace, stdp_hebbian_update,
)
from el.thermofield.plasticity import hebbian_update

GRID = 14
N_PAT = 8
A_POS = (12, 1)
B_POS = (12, 5)
SEEDS = list(range(6))   # smaller for CI speed
N_TRIALS = 10


def _noisy_cue(p, drop, rng, n_cells, cols):
    kn = max(1, int(round(len(p) * (1 - drop))))
    keep = [p[i] for i in sorted(rng.choice(len(p), kn, replace=False))]
    pset = set(p); ds = []
    while len(ds) < kn:
        idx = int(rng.integers(0, n_cells)); rc = (idx // cols, idx % cols)
        if rc not in pset and rc not in ds: ds.append(rc)
    return keep + ds


def _pattern_acc(mem, patterns, drop, rng, nt):
    cfg = mem.cfg; nc = cfg.rows * cfg.cols; correct = 0
    for _ in range(nt):
        i = int(rng.integers(0, len(patterns)))
        cue = _noisy_cue(patterns[i], drop, rng, nc, cfg.cols)
        b, _, _ = mem.recall(cue)
        if b == i: correct += 1
    return correct / nt


def _seq_train(field, n_epochs=30, hold=5, gap=2):
    trace = EligibilityTrace((field.cfg.rows, field.cfg.cols), decay=0.80)
    for _ in range(n_epochs):
        field.reset_temp(); trace.reset()
        present_event(field, trace, [A_POS], [1.0], hold=hold)
        relax_with_trace(field, trace, gap)
        field._clamp_positions = []; field._clamp_values = []
        field.inject([B_POS], [1.0])
        stdp_hebbian_update(field, trace, lr=0.07)
        for _ in range(hold):
            field.step(); trace.update(field.T)
        hebbian_update(field, lr=0.07, decay=0.001)
    field.reset_temp()


def _seq_probe(field, seed, hold=5, read_delay=6):
    field.reset_temp()
    tr = EligibilityTrace((field.cfg.rows, field.cfg.cols), decay=0.80)
    present_event(field, tr, [A_POS], [1.0], hold=hold)
    relax_with_trace(field, tr, read_delay)
    cue = float(field.T[B_POS])
    fresh = Field(field.cfg, seed=seed)
    ftr = EligibilityTrace((field.cfg.rows, field.cfg.cols), decay=0.80)
    present_event(fresh, ftr, [A_POS], [1.0], hold=hold)
    relax_with_trace(fresh, ftr, read_delay)
    return cue - float(fresh.T[B_POS])


def _make_mem(seed, cfg):
    pat_size = max(4, int(0.05 * GRID * GRID))
    wta_k = max(pat_size + 2, int(0.075 * GRID * GRID))
    return PatternMemory(
        cfg=cfg, seed=seed, write_steps=20, write_lr=0.30,
        wta_k=wta_k, wta_suppression=0.3, rule="hebb"), pat_size


def _run(condition):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pat_accs, discrims = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mem, pat_size = _make_mem(seed, cfg)
        patterns = [random_pattern(GRID, GRID, k=pat_size, rng=rng)
                    for _ in range(N_PAT)]
        field = mem.field
        if condition == "seq_only":
            _seq_train(field)
        elif condition == "pattern_only":
            for p in patterns: mem.store(p)
        elif condition == "multi_pat_then_seq":
            for p in patterns: mem.store(p)
            _seq_train(field)
        if mem.patterns:
            pat_accs.append(_pattern_acc(mem, patterns, 0.5,
                np.random.default_rng(seed*31+7), N_TRIALS))
        discrims.append(_seq_probe(field, seed))
    return pat_accs, discrims


def test_pattern_memory_survives_sequence_cotraining():
    """Pattern recall must not collapse when sequence training is added."""
    alone, _ = _run("pattern_only")
    multi, _ = _run("multi_pat_then_seq")
    chance = 1.0 / N_PAT
    assert np.mean(alone) > chance + 0.20, (
        f"pattern alone too weak: {np.mean(alone):.3f}")
    # Multi-task must keep pattern recall within 0.15 of alone (one-sided)
    assert np.mean(multi) > np.mean(alone) - 0.15, (
        f"pattern recall collapsed under co-training: "
        f"alone={np.mean(alone):.3f} multi={np.mean(multi):.3f}")


def test_sequence_learning_works_alone():
    """Sequence A→B discrimination must be clearly positive when trained."""
    _, discrims = _run("seq_only")
    m = float(np.mean(discrims))
    sd = float(np.std(discrims, ddof=1))
    se = sd / np.sqrt(len(discrims))
    assert m - 2 * se > 0.0, (
        f"seq learning failed alone: discrim={m:.4f} 2σ_low={m-2*se:.4f}; "
        f"raw={discrims}")


def test_sequence_destroyed_by_pattern_cotraining_KNOWN_BUG():
    """HONEST regression: pattern Hebb writes (symmetric C updates)
    OVERWRITE the directed B-bias signal that sequence STDP needs.

    After multi_pat_then_seq, A→B discrimination drops from clearly
    positive (≈+0.04) to ≈zero or negative. We pin this down so it
    can't be silently fixed (would also be silent breakage).

    Empirically (8 seeds × 15 trials):
       seq_only:           +0.037 ± 0.011
       multi_pat_then_seq: -0.016 ± 0.014  ← below baseline

    This test will START FAILING once we fix multi-task interference
    (e.g. by gating C-writes during seq training, or by separating
    the channels topologically). At that point, FLIP this assertion
    to assert positive discrim and remove the _KNOWN_BUG suffix.
    """
    _, multi_discrims = _run("multi_pat_then_seq")
    m = float(np.mean(multi_discrims))
    # Currently negative; the bug is "active" if mean ≤ +0.005
    assert m < 0.015, (
        f"multi-task seq discrim is no longer broken: mean={m:.4f}. "
        f"If you intentionally fixed this, FLIP the assertion to "
        f"assert m > 0.005 (clearly positive) and rename the test.")
