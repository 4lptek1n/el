"""Multi-modal substrate regression — pattern + chain + persistence on ONE Field."""
from __future__ import annotations
import os, tempfile, numpy as np
import pytest

from el.thermofield.field import FieldConfig
from el.thermofield.multi_substrate import MultiModalSubstrate
from el.thermofield.pattern_memory import random_pattern, corrupt
from el.thermofield.skip_bank import SkipBank


GRID = 14


def _patterns(seed, n=4, k=8):
    rng = np.random.default_rng(seed)
    return [random_pattern(GRID, GRID, k, rng) for _ in range(n)]


def _make_chain(seed, n=3, md=3):
    rng = np.random.default_rng(seed + 99)
    chosen = []
    for _ in range(400):
        if len(chosen) == n + 1:
            break
        c = (int(rng.integers(0, GRID)), int(rng.integers(0, GRID)))
        if all(abs(c[0] - x[0]) + abs(c[1] - x[1]) >= md for x in chosen):
            chosen.append(c)
    while len(chosen) < n + 1:
        chosen.append((int(rng.integers(0, GRID)), int(rng.integers(0, GRID))))
    return chosen


def _build(seed):
    return MultiModalSubstrate(cfg=FieldConfig(rows=GRID, cols=GRID),
                               seed=seed, K=4, md_min=3)


def test_pattern_memory_works_on_multi_substrate():
    """Pattern memory façade still recalls correctly on the unified API."""
    seed = 0
    sub = _build(seed)
    pats = _patterns(seed)
    for p in pats:
        sub.store_pattern(p)
    rng = np.random.default_rng(seed + 333)
    correct = total = 0
    for true_idx, p in enumerate(pats):
        for _ in range(4):
            cue = corrupt(p, 0.4, rng)
            pred, _, _ = sub.recall(cue)
            correct += int(pred == true_idx); total += 1
    acc = correct / total
    assert acc >= 0.5, f"pattern recall on multi-substrate: {acc:.3f}"


def test_chain_link_learning_works_on_multi_substrate():
    """Chain training on the unified substrate produces positive link discrim."""
    seed = 0
    sub = _build(seed)
    anchors = _make_chain(seed, n=3, md=3)
    sub.train_chain(anchors, n_epochs=120, lr=0.20)
    discrims = [sub.probe_chain_link(anchors[j], anchors[j + 1])
                for j in range(len(anchors) - 1)]
    assert all(d > 0 for d in discrims), \
        f"link discriminations not all positive: {discrims}"


def test_pattern_and_chain_coexist_on_one_substrate():
    """Both modalities on the SAME substrate keep ≥50% pattern recall AND
    positive chain discrim. This is the kızıl elma multi-modal claim."""
    seed = 1
    sub = _build(seed)
    pats = _patterns(seed)
    for p in pats:
        sub.store_pattern(p)
    anchors = _make_chain(seed, n=3, md=3)
    sub.train_chain(anchors, n_epochs=120, lr=0.20)

    # Pattern recall AFTER chain training (chain must not destroy pattern)
    rng = np.random.default_rng(seed + 333)
    correct = total = 0
    for true_idx, p in enumerate(pats):
        for _ in range(4):
            cue = corrupt(p, 0.4, rng)
            pred, _, _ = sub.recall(cue)
            correct += int(pred == true_idx); total += 1
    acc = correct / total
    assert acc >= 0.5, f"pattern recall after chain training: {acc:.3f}"

    # Chain links still positive
    discrims = [sub.probe_chain_link(anchors[j], anchors[j + 1])
                for j in range(len(anchors) - 1)]
    pos = sum(1 for d in discrims if d > 0)
    assert pos >= 2, f"chain pos links after coexistence: {pos}/3 ({discrims})"


def test_skipbank_save_load_roundtrip():
    bank = SkipBank(GRID, GRID, K=4, md_min=3, seed=7)
    rng = np.random.default_rng(7)
    bank.w[:] = rng.normal(0, 0.05, size=bank.w.shape).astype(np.float32)
    np.clip(bank.w, -bank.w_clip, bank.w_clip, out=bank.w)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bank.npz")
        bank.save(p)
        bank2 = SkipBank.load(p)
        np.testing.assert_array_equal(bank.src, bank2.src)
        np.testing.assert_array_equal(bank.dst, bank2.dst)
        np.testing.assert_array_equal(bank.w, bank2.w)
        assert bank.K == bank2.K
        assert bank.cfg.md_min == bank2.cfg.md_min


def test_multi_substrate_save_load_preserves_both_modalities():
    """Save → load → both pattern recall AND chain discrim survive."""
    seed = 2
    sub = _build(seed)
    pats = _patterns(seed)
    for p in pats:
        sub.store_pattern(p)
    anchors = _make_chain(seed, n=3, md=3)
    sub.train_chain(anchors, n_epochs=120, lr=0.20)

    # Score before save
    rng = np.random.default_rng(seed + 333)
    def score(s):
        rng_local = np.random.default_rng(seed + 333)
        correct = total = 0
        for ti, p in enumerate(pats):
            for _ in range(4):
                cue = corrupt(p, 0.4, rng_local)
                pred, _, _ = s.recall(cue)
                correct += int(pred == ti); total += 1
        return correct / total

    pat_before = score(sub)
    chain_before = [sub.probe_chain_link(anchors[j], anchors[j + 1])
                    for j in range(len(anchors) - 1)]

    with tempfile.TemporaryDirectory() as d:
        sub.save(d)
        sub2 = MultiModalSubstrate.load(d)
        pat_after = score(sub2)
        chain_after = [sub2.probe_chain_link(anchors[j], anchors[j + 1])
                       for j in range(len(anchors) - 1)]

    assert abs(pat_before - pat_after) <= 0.05, \
        f"pattern recall drift: {pat_before:.3f} -> {pat_after:.3f}"
    for db, da in zip(chain_before, chain_after):
        assert abs(db - da) <= 0.05, \
            f"chain link drift: {db:.4f} -> {da:.4f}"
