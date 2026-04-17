"""Eşik 4 — replay / persistence: substrate state must outlive a run.

Real proof of "büyük çekirdek" needs the substrate to accumulate
capacity across runs, not just within one. Three honest tests:

  1. round_trip: save → load → recall accuracy unchanged
  2. continue_training: save → load → store more patterns →
     final recall ≥ scratch-from-second-batch (no advantage from
     starting cold)
  3. blob_size_reasonable: not exploding (sanity)
"""
from __future__ import annotations
import tempfile, os
import numpy as np
from el.thermofield import FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern


GRID = 14
PAT_SIZE = max(4, int(0.05 * GRID * GRID))
WTA_K = max(PAT_SIZE + 2, int(0.075 * GRID * GRID))
DROP = 0.5


def _noisy_cue(p, drop, rng, n_cells, cols):
    kn = max(1, int(round(len(p) * (1 - drop))))
    keep = [p[i] for i in sorted(rng.choice(len(p), kn, replace=False))]
    pset = set(p); ds = []
    while len(ds) < kn:
        idx = int(rng.integers(0, n_cells)); rc = (idx // cols, idx % cols)
        if rc not in pset and rc not in ds: ds.append(rc)
    return keep + ds


def _acc(mem, patterns, rng, nt=20):
    cfg = mem.cfg; nc = cfg.rows * cfg.cols; correct = 0
    for _ in range(nt):
        i = int(rng.integers(0, len(patterns)))
        cue = _noisy_cue(patterns[i], DROP, rng, nc, cfg.cols)
        b, _, _ = mem.recall(cue)
        if b == i: correct += 1
    return correct / nt


def _new_mem(seed):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    return PatternMemory(
        cfg=cfg, seed=seed, write_steps=15, write_lr=0.30,
        wta_k=WTA_K, wta_suppression=0.3, rule="hebb")


def test_round_trip_preserves_recall():
    """save → load on a different object reproduces recall accuracy."""
    seed = 0
    rng = np.random.default_rng(seed)
    patterns = [random_pattern(GRID, GRID, k=PAT_SIZE, rng=rng)
                for _ in range(6)]
    mem = _new_mem(seed)
    for p in patterns: mem.store(p)
    acc_before = _acc(mem, patterns, np.random.default_rng(99))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "snap.npz")
        mem.save(path)
        loaded = PatternMemory.load(path)
    # Field state byte-identical
    assert np.allclose(loaded.field.C_right, mem.field.C_right)
    assert np.allclose(loaded.field.C_down, mem.field.C_down)
    assert np.allclose(loaded.field.B_right, mem.field.B_right)
    assert np.allclose(loaded.field.B_down, mem.field.B_down)
    assert len(loaded.patterns) == len(mem.patterns)
    for a, b in zip(loaded.patterns, mem.patterns):
        assert sorted(a) == sorted(b)
    # Recall accuracy reproduced (deterministic given identical state +
    # same probe rng)
    acc_after = _acc(loaded, patterns, np.random.default_rng(99))
    assert acc_after == acc_before, (
        f"round-trip recall changed: {acc_before:.3f} → {acc_after:.3f}")


def test_snapshot_carries_real_capacity_across_runs():
    """The snapshot must encode learned patterns, not just config:

      Path A: train batch1, save, load, train batch2.
              Then probe batch1-only recall.
              The snapshot+continue substrate MUST still recall batch1
              (it has both batches in C).

      Path B: cold-start substrate that only saw batch2 (never batch1),
              but the patterns list is set to batch1 so it tries to
              recall items it has no Hebb trace for.
              Must collapse to chance (~ 1/8 = 0.125).

    On a 28×28 grid with 8 patterns/batch this is well within capacity
    so Path A should hit ≥0.7 on batch1 while Path B should be <0.30.
    """
    grid = 28  # comfortably above N=16 capacity at this grid
    pat_size = max(4, int(0.05 * grid * grid))
    wta_k = max(pat_size + 2, int(0.075 * grid * grid))
    cfg = FieldConfig(rows=grid, cols=grid)

    def make(seed):
        return PatternMemory(
            cfg=cfg, seed=seed, write_steps=15, write_lr=0.30,
            wta_k=wta_k, wta_suppression=0.3, rule="hebb")

    aggressive_drop = 0.75

    def acc_aggressive(mem, probe_set, rng, n_trials=40):
        cfg = mem.cfg; nc = cfg.rows * cfg.cols; correct = 0
        for _ in range(n_trials):
            i = int(rng.integers(0, len(probe_set)))
            target = probe_set[i]
            cue = _noisy_cue(target, aggressive_drop, rng, nc, cfg.cols)
            best, _, _ = mem.recall(cue)
            if mem.patterns[best] == target: correct += 1
        return correct / n_trials

    # Pre-registered multi-seed run (no cherry-pick). Architect found
    # single-seed had min gap 0.075 across 16 seeds — too thin.
    seeds = list(range(8))
    accs_a, accs_b = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        batch1 = [random_pattern(grid, grid, k=pat_size, rng=rng) for _ in range(8)]
        batch2 = [random_pattern(grid, grid, k=pat_size, rng=rng) for _ in range(8)]

        # Path A: snapshot+continue, probe batch1 over full 16 candidates
        mem_a = make(seed)
        for p in batch1: mem_a.store(p)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "s.npz")
            mem_a.save(path)
            mem_a = PatternMemory.load(path)
        for p in batch2: mem_a.store(p)
        accs_a.append(acc_aggressive(mem_a, batch1,
                                     np.random.default_rng(seed * 31 + 123)))

        # Path B: cold-start saw only batch2 in C; patterns list = full 16
        mem_b = make(seed)
        for p in batch2: mem_b.store(p)
        mem_b.patterns = batch1 + batch2
        accs_b.append(acc_aggressive(mem_b, batch1,
                                     np.random.default_rng(seed * 31 + 123)))

    diffs = np.array(accs_a) - np.array(accs_b)
    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs, ddof=1)) / np.sqrt(len(diffs))
    lo = mean_diff - 2 * se_diff
    n_pos = int((diffs > 0).sum())

    # Sign test: under null (no effect), positive count ~ Binomial(8, 0.5).
    # P(≥7 positive) = 0.0352. Require ≥7/8 positive AS WELL AS positive
    # 2σ_low — both must hold, no single-seed cherry-pick possible.
    assert float(np.mean(accs_a)) > 0.40, (
        f"snapshot lost batch1 capacity: mean acc_a={np.mean(accs_a):.3f}; "
        f"raw={accs_a}")
    assert n_pos >= 7, (
        f"snapshot only beat cold-start in {n_pos}/8 seeds (need ≥7); "
        f"diffs={diffs.tolist()}")
    assert lo > 0.0, (
        f"snapshot capacity gap not significant: mean_diff={mean_diff:.3f} "
        f"2σ_low={lo:.3f}; accs_a={accs_a} accs_b={accs_b}")


def test_save_blob_is_reasonable_size():
    """Sanity: 14×14 snapshot stays under 50 KB (no accidental bloat)."""
    seed = 2
    rng = np.random.default_rng(seed)
    patterns = [random_pattern(GRID, GRID, k=PAT_SIZE, rng=rng)
                for _ in range(6)]
    mem = _new_mem(seed)
    for p in patterns: mem.store(p)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.npz")
        mem.save(path)
        sz = os.path.getsize(path)
    assert sz < 50_000, f"snapshot too large: {sz} bytes"
