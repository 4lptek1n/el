"""Eşik 4 — pattern memory survives serialize/deserialize and accumulates."""
from __future__ import annotations
import os, tempfile, numpy as np, pytest

from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern, corrupt


GRID = 20


def _gen(seed, n, k=12):
    rng = np.random.default_rng(seed)
    return [random_pattern(GRID, GRID, k, rng) for _ in range(n)]


def _acc(pm, patterns, seed, cues_per=4, drop=0.5):
    rng = np.random.default_rng(seed + 1000)
    correct = total = 0
    for i, p in enumerate(patterns):
        for _ in range(cues_per):
            cue = corrupt(p, drop, rng)
            pred, _, _ = pm.recall(cue)
            correct += int(pred == i); total += 1
    return correct / total


def _build(seed, patterns):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pm = PatternMemory(cfg=cfg, seed=seed)
    for p in patterns:
        pm.store(p)
    return pm


def test_save_load_roundtrip_preserves_recall():
    seed = 0
    patterns = _gen(seed, n=8)
    pm = _build(seed, patterns)
    acc_before = _acc(pm, patterns, seed)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.npz")
        pm.save(path)
        pm2 = PatternMemory.load(path)
        # field must be reconstructed
        assert pm2.field is not None
        # patterns list preserved
        assert len(pm2.patterns) == len(pm.patterns)
        for p_orig, p_load in zip(pm.patterns, pm2.patterns):
            assert sorted(p_orig) == sorted(p_load)
        # substrate weights preserved exactly
        np.testing.assert_array_equal(pm.field.C_right, pm2.field.C_right)
        np.testing.assert_array_equal(pm.field.B_right, pm2.field.B_right)
        # recall accuracy preserved within tight tolerance
        acc_after = _acc(pm2, patterns, seed)
        assert abs(acc_after - acc_before) <= 0.05, \
            f"recall drift {acc_before:.3f} -> {acc_after:.3f}"


def test_load_and_continue_beats_scratch_on_second_batch():
    """Train N=8, save. Load + train N=8 more. Compare to fresh-on-second-batch.

    The cumulative model must beat the from-scratch-on-second-batch model
    on the FULL 16-pattern set — that is the operational meaning of
    persistence: yesterday's training has value today.
    """
    seed = 1
    batch1 = _gen(seed, n=8)
    batch2 = _gen(seed + 100, n=8)
    full = batch1 + batch2

    # Trajectory A: train batch1, save, load, train batch2 (cumulative)
    pm_a = _build(seed, batch1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snap.npz")
        pm_a.save(path)
        pm_a_loaded = PatternMemory.load(path)
        for p in batch2:
            pm_a_loaded.store(p)
        cum_acc = _acc(pm_a_loaded, full, seed)

    # Trajectory B: fresh, train ONLY batch2
    pm_b = _build(seed, batch2)
    # but scored against full set (it never saw batch1, so by construction
    # batch1 patterns should be near-chance for it)
    scratch_acc = _acc(pm_b, full, seed)

    # Cumulative must be strictly better — it has seen both batches.
    assert cum_acc > scratch_acc + 0.10, \
        f"persistence failed: cum={cum_acc:.3f} scratch_b={scratch_acc:.3f}"


def test_save_load_empty_memory():
    """An untrained memory must round-trip."""
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pm = PatternMemory(cfg=cfg, seed=42)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "empty.npz")
        pm.save(path)
        pm2 = PatternMemory.load(path)
        assert pm2.patterns == []
        assert pm2.field is not None
