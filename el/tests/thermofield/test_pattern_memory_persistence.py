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


def test_continue_training_beats_cold_start():
    """save snapshot of N₁=8 patterns, load, store N₂=8 more →
    final recall on all 16 must be ≥ a fresh substrate that only
    saw the 2nd batch (the snapshot must carry real capacity)."""
    seed = 1
    rng = np.random.default_rng(seed)
    batch1 = [random_pattern(GRID, GRID, k=PAT_SIZE, rng=rng) for _ in range(8)]
    batch2 = [random_pattern(GRID, GRID, k=PAT_SIZE, rng=rng) for _ in range(8)]

    # Path A: train batch1, save, load, train batch2 → recall on all 16
    mem_a = _new_mem(seed)
    for p in batch1: mem_a.store(p)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.npz")
        mem_a.save(path)
        mem_a_loaded = PatternMemory.load(path)
    for p in batch2: mem_a_loaded.store(p)
    full = batch1 + batch2
    acc_continue = _acc(mem_a_loaded, full, np.random.default_rng(123),
                        nt=40)

    # Path B: cold-start substrate, only sees batch2; tested on full 16
    # (it cannot recall batch1 — it never saw them — so it must lose)
    mem_b = _new_mem(seed)
    for p in batch2: mem_b.store(p)
    # Inject batch1 patterns into mem_b's pattern list so the recall
    # function CAN match them — but the substrate has no Hebb trace
    # for them, so it will fail those queries
    mem_b.patterns = list(full)
    acc_cold = _acc(mem_b, full, np.random.default_rng(123), nt=40)

    assert acc_continue > acc_cold + 0.10, (
        f"continue-from-snapshot ({acc_continue:.3f}) failed to beat "
        f"cold-start ({acc_cold:.3f}) by ≥0.10 — snapshot didn't carry "
        f"real capacity")


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
