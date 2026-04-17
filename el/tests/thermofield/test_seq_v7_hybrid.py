"""Canonical regression for v7 hybrid sequence substrate.

Pins the WINS achieved with heat substrate (untouched) + sparse
temporal skip-edge bank. Tests are tight enough to catch any
regression but not so tight they break on minor tuning.

Acceptance pinned per /godsay v7-final spec:
  - n=3 md=2 -> 3/3 pos links
  - n=5 md=3 -> >=4/5 pos links (5/5 typical, 4/5 lower bound)
  - n=10 md=3 -> >=8/10 pos links (10/10 typical)
  - density < 5% (sparse identity preserved)
  - empty bank ablation: aggregate near zero
  - topology robust: 3 bank seeds, all pass

Pattern memory tests live in test_pattern_memory.py — that they
keep passing is the orthogonal guarantee that v7 doesn't touch core.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

import numpy as np
import pytest
import seq_chain_v7_hybrid as V7

SEEDS = list(range(8))


# ============================================================== Canonical
def test_canonical_n3_md2_perfect():
    """n=3, md=2: must achieve 3/3 positive links, density < 5%."""
    ovr, pos, lm, dens = V7.run_hybrid(
        n=3, md=2, lr=0.20, ep=100, K=4, md_min=3,
        skip_eta=0.05, grid=14, seeds=SEEDS,
    )
    assert pos == 3, f"expected 3/3 pos, got {pos}/3 (means={lm}, ovr={ovr:+.4f})"
    assert dens < 0.05, f"density {dens*100:.2f}% exceeds 5% sparse limit"
    assert ovr > 0.01, f"overall {ovr:+.4f} too weak"


def test_canonical_n5_md3():
    """n=5, md=3: must achieve >=4/5 pos (5/5 typical)."""
    ovr, pos, lm, dens = V7.run_hybrid(
        n=5, md=3, lr=0.20, ep=120, K=4, md_min=3,
        skip_eta=0.05, grid=14, seeds=SEEDS,
    )
    assert pos >= 4, f"expected >=4/5 pos, got {pos}/5 (means={lm})"
    assert dens < 0.05, f"density {dens*100:.2f}% exceeds 5%"


def test_canonical_n10_md3():
    """n=10, md=3: must achieve >=8/10 pos (10/10 typical).
    This is the headline result — long-range sequence learning solved.
    """
    ovr, pos, lm, dens = V7.run_hybrid(
        n=10, md=3, lr=0.20, ep=150, K=4, md_min=3,
        skip_eta=0.05, grid=14, seeds=SEEDS,
    )
    assert pos >= 8, f"expected >=8/10 pos, got {pos}/10 (means={lm})"
    assert dens < 0.05, f"density {dens*100:.2f}% exceeds 5%"
    assert ovr > 0.01, f"overall {ovr:+.4f} suggests collapse"


# ============================================================== Ablation
def test_empty_bank_ablation_near_zero():
    """If skip-bank weights stay zero (no training), discrimination ~ 0.
    This is the v4-equivalent ablation — heat alone cannot bridge md>=3.
    """
    from seq_chain_v7_hybrid import HybridTrainer, make_chain, probe_link
    discrim = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        anchors = make_chain(5, 14, rng, md=3)
        tr = HybridTrainer(rows=14, cols=14, seed=seed, K=4, md_min=3)
        # NO TRAINING — bank weights stay at 0
        for li in range(5):
            d = probe_link(seed, anchors[li], anchors[li+1], tr.bank,
                           K=4, md_min=3, grid=14, skip_eta=0.05)
            discrim.append(d)
    arr = np.asarray(discrim)
    assert abs(arr.mean()) < 0.005, (
        f"untrained bank should give ~0 discrim, got mean={arr.mean():+.4f}"
    )


# ============================================================== Topology robustness
@pytest.mark.parametrize("bank_offset", [0, 100, 200])
def test_topology_robust_n5(bank_offset):
    """v7 must not depend on a single lucky topology.
    Re-sample bank with different seed offsets — must still pass.
    """
    ovr, pos, lm, dens = V7.run_hybrid(
        n=5, md=3, lr=0.20, ep=120, K=4, md_min=3,
        skip_eta=0.05, grid=14, seeds=SEEDS,
        bank_seed_offset=bank_offset,
    )
    assert pos >= 4, (
        f"topology offset {bank_offset}: expected >=4/5, got {pos}/5 "
        f"(means={lm}). v7 is topology-fragile if this fails."
    )
    assert dens < 0.05


# ============================================================== Sparseness
def test_sparseness_K2_still_works():
    """Even at K=2 (1% density), v7 should beat baseline at n=5."""
    ovr, pos, lm, dens = V7.run_hybrid(
        n=5, md=3, lr=0.20, ep=150, K=2, md_min=3,
        skip_eta=0.08, grid=14, seeds=SEEDS,
    )
    assert pos >= 2, f"K=2 too sparse: pos={pos}/5 (means={lm})"
    assert dens < 0.02, f"K=2 density should be ~1%, got {dens*100:.2f}%"


# ============================================================== Long-range
def test_long_range_md7():
    """Anchors at min_dist=7 — true long-range. Heat alone cannot bridge."""
    ovr, pos, lm, dens = V7.run_hybrid(
        n=5, md=7, lr=0.20, ep=200, K=4, md_min=7,
        skip_eta=0.05, grid=14, seeds=SEEDS,
    )
    assert pos >= 3, f"long-range md=7: pos={pos}/5 (means={lm})"
