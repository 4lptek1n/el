"""Pattern memory tests — pre-registered, paired, causally-controlled.

Three claims are encoded as separate tests:

  A. Without inhibition, Hebbian training on the substrate ACTIVELY
     HURTS recall under hard noisy cues (mean lift strongly negative).

  B. k-WTA inhibition causally rescues this — same patterns, same
     cues, same write protocol, only `wta_k` toggled. Adding WTA
     produces a large positive paired lift in nearly every seed.

  C. With k-WTA on, training no longer hurts on average, and is
     non-negative in a clear majority of seeds.

Seeds are pre-registered as `range(0, 12)` to defuse cherry-picking.
Configs are IDENTICAL across the WTA-on / WTA-off arms except for
the single toggled hyper-parameter.

Auxiliary tests cover the kwta_step exact-k semantics and the
global_gain_step homeostatic primitive.
"""
from __future__ import annotations

import numpy as np

from el.thermofield import FieldConfig
from el.thermofield.inhibition import global_gain_step, kwta_step
from el.thermofield.pattern_memory import (
    PatternMemory,
    corrupt,
    random_pattern,
)


# ---------------------------------------------------------------------------
# Shared helpers — IDENTICAL between WTA-on and WTA-off arms
# ---------------------------------------------------------------------------
SEEDS = list(range(12))     # pre-registered, NOT cherry-picked
WRITE_STEPS = 20
WRITE_LR = 0.30
WTA_K = 15
WTA_SUPP = 0.3
N_PATTERNS = 3
PATTERN_SIZE = 10
DROP_FRAC = 0.5
N_TRIALS = 20
ROWS = COLS = 14


def _noisy_cue(pattern, drop_frac, rng, n_cells, cols):
    """Cue = (1-drop_frac) of pattern + same many distractor cells."""
    keep_n = max(1, int(round(len(pattern) * (1.0 - drop_frac))))
    keep = [pattern[i] for i in
            sorted(rng.choice(len(pattern), keep_n, replace=False))]
    pattern_set = set(pattern)
    distractors = []
    while len(distractors) < keep_n:
        idx = int(rng.integers(0, n_cells))
        rc = (idx // cols, idx % cols)
        if rc not in pattern_set and rc not in distractors:
            distractors.append(rc)
    return keep + distractors


def _accuracy(mem, patterns, drop, rng, n_trials):
    cfg = mem.cfg
    n_cells = cfg.rows * cfg.cols
    correct = 0
    for _ in range(n_trials):
        i = int(rng.integers(0, len(patterns)))
        cue = _noisy_cue(patterns[i], drop, rng, n_cells, cfg.cols)
        best_i, _, _ = mem.recall(cue)
        if best_i == i:
            correct += 1
    return correct / n_trials


def _trial(seed, *, wta_k, write_steps, write_lr):
    """Return (trained_acc, untrained_acc, lift). Deterministic for a seed."""
    cfg = FieldConfig(rows=ROWS, cols=COLS)
    rng = np.random.default_rng(seed)
    patterns = [random_pattern(cfg.rows, cfg.cols, k=PATTERN_SIZE, rng=rng)
                for _ in range(N_PATTERNS)]
    trained = PatternMemory(
        cfg=cfg, seed=seed,
        write_steps=write_steps, write_lr=write_lr,
        wta_k=wta_k, wta_suppression=WTA_SUPP,
    )
    untrained = PatternMemory(
        cfg=cfg, seed=seed,
        write_steps=0, write_lr=0.0,
        wta_k=wta_k, wta_suppression=WTA_SUPP,
    )
    for p in patterns:
        trained.store(p)
        untrained.store(p)
    rng_t = np.random.default_rng(seed * 31 + 7)
    rng_u = np.random.default_rng(seed * 31 + 7)   # IDENTICAL stream
    a_t = _accuracy(trained, patterns, DROP_FRAC, rng_t, N_TRIALS)
    a_u = _accuracy(untrained, patterns, DROP_FRAC, rng_u, N_TRIALS)
    return a_t, a_u, a_t - a_u


# ---------------------------------------------------------------------------
# Pipeline smoke test
# ---------------------------------------------------------------------------
def test_pipeline_runs_end_to_end() -> None:
    rng = np.random.default_rng(0)
    cfg = FieldConfig(rows=12, cols=12)
    mem = PatternMemory(cfg=cfg, seed=0)
    p = random_pattern(cfg.rows, cfg.cols, k=8, rng=rng)
    mem.store(p)
    cue = corrupt(p, drop_frac=0.5, rng=rng)
    best_i, score, hot = mem.recall(cue)
    assert best_i == 0
    assert 0.0 <= score <= 1.0
    assert hot.ndim == 2 and hot.shape[1] == 2


# ---------------------------------------------------------------------------
# Claim A: without inhibition, training hurts
# ---------------------------------------------------------------------------
def test_no_inhibition_training_hurts_recall() -> None:
    """With wta_k=0 (no inhibition), the trained substrate is strictly
    worse than the untrained one in nearly every seed of a pre-registered
    12-seed run. Establishes the architectural problem the WTA fix solves.
    """
    lifts = [
        _trial(seed, wta_k=0, write_steps=WRITE_STEPS, write_lr=WRITE_LR)[2]
        for seed in SEEDS
    ]
    mean_lift = float(np.mean(lifts))
    n_negative = sum(1 for l in lifts if l < 0)
    # Population probe: mean ~-0.36, ALL 12 seeds non-positive.
    assert mean_lift < -0.10, (
        f"Without WTA, expected strongly negative mean lift but got "
        f"{mean_lift:+.3f}: {lifts}"
    )
    assert n_negative >= 9, (
        f"Without WTA, expected >=9/12 seeds with negative lift but got "
        f"{n_negative}/12: {lifts}"
    )


# ---------------------------------------------------------------------------
# Claim B: WTA causally rescues training (paired test)
# ---------------------------------------------------------------------------
def test_kwta_causally_improves_training_lift_paired() -> None:
    """For each seed, run both WTA-on and WTA-off with otherwise IDENTICAL
    config. Adding WTA must produce a large positive paired difference
    (mean delta-lift >> 0) and improve nearly every seed individually.
    This is the strong causal evidence that WTA is the active ingredient.
    """
    deltas = []
    for seed in SEEDS:
        _, _, lift_off = _trial(seed, wta_k=0,
                                write_steps=WRITE_STEPS, write_lr=WRITE_LR)
        _, _, lift_on = _trial(seed, wta_k=WTA_K,
                               write_steps=WRITE_STEPS, write_lr=WRITE_LR)
        deltas.append(lift_on - lift_off)

    mean_delta = float(np.mean(deltas))
    n_positive = sum(1 for d in deltas if d > 0)
    # Probe: paired mean ~+0.40, all 12 seeds positive.
    assert mean_delta > 0.20, (
        f"WTA did not causally improve training lift (paired): "
        f"mean delta={mean_delta:+.3f}, deltas={deltas}"
    )
    assert n_positive >= 11, (
        f"WTA improved <11/12 seeds (paired): {n_positive}/12 deltas={deltas}"
    )


# ---------------------------------------------------------------------------
# Claim C: with WTA on, training no longer hurts on average
# ---------------------------------------------------------------------------
def test_kwta_on_training_no_longer_hurts() -> None:
    """With WTA on (and the same hard noisy-cue protocol as the negative
    test), trained recall is at least neutral on average and non-negative
    in a clear majority of seeds. Conservative claim, defensible at the
    measured ~+0.04 mean lift / 8/12 non-negative population statistic.
    """
    lifts = [
        _trial(seed, wta_k=WTA_K, write_steps=WRITE_STEPS, write_lr=WRITE_LR)[2]
        for seed in SEEDS
    ]
    mean_lift = float(np.mean(lifts))
    n_nonneg = sum(1 for l in lifts if l >= 0)
    assert mean_lift > 0.0, (
        f"With WTA, expected non-harmful training (mean lift > 0) but got "
        f"{mean_lift:+.3f}: {lifts}"
    )
    assert n_nonneg >= 7, (
        f"With WTA, expected >=7/12 non-negative seeds, got {n_nonneg}/12: "
        f"{lifts}"
    )


# ---------------------------------------------------------------------------
# Auxiliary unit tests
# ---------------------------------------------------------------------------
def test_kwta_exact_k_with_ties() -> None:
    """Exact-k contract: kwta_step must spare EXACTLY k cells even when
    there are ties at the boundary value."""
    T = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    kwta_step(T, k=2, suppression=0.0)
    n_winners = int((T > 0).sum())
    assert n_winners == 2, f"expected exactly 2 winners with ties, got {n_winners}"


def test_kwta_no_op_on_edge_k() -> None:
    """k <= 0 or k >= n should be a no-op (no inhibition applicable)."""
    T = np.array([0.1, 0.5, 0.3, 0.8], dtype=np.float32)
    original = T.copy()
    kwta_step(T, k=0, suppression=0.0)
    assert np.array_equal(T, original)
    kwta_step(T, k=4, suppression=0.0)
    assert np.array_equal(T, original)


def test_global_gain_step_pulls_mean_down() -> None:
    """global_gain_step subtracts a uniform offset when mean(T) is above
    the target, leaving all cells non-negative."""
    T = np.array([0.6, 0.7, 0.8, 0.9], dtype=np.float32)  # mean=0.75
    global_gain_step(T, target_mean=0.30, rate=1.0)
    assert float(T.mean()) < 0.75
    assert T.min() >= 0.0


def test_global_gain_step_no_op_when_mean_already_low() -> None:
    """When mean is at or below target, do nothing (no upward push)."""
    T = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    original = T.copy()
    global_gain_step(T, target_mean=0.10, rate=1.0)
    assert np.array_equal(T, original)


# ---------------------------------------------------------------------------
# Covariance rule (Hebb + anti-Hebb) on the full pattern memory pipeline
# ---------------------------------------------------------------------------
def test_covariance_rule_runs_and_is_non_harmful():
    """Honest scope test for the covariance rule on the full pipeline.

    A 32-seed paired probe (`scripts/probe_cov32.py` style) found NO
    statistically significant advantage of the covariance rule over
    plain Hebb on this benchmark: paired mean delta = -0.005,
    sign-test p ≈ 0.85, 95 % bootstrap CI [-0.045, +0.030]. So we do
    NOT assert covariance is better than Hebb.

    What we DO assert: the covariance rule, when used as the
    plasticity rule under WTA, still gives a non-harmful (>= 0 mean,
    majority non-negative) lift over the no-training baseline — i.e.
    it is a *valid* alternative plasticity primitive, just not a
    measurable improvement on this task. This guards against silent
    breakage of the rule itself.
    """
    cov_lifts = []
    for seed in SEEDS:
        cfg = FieldConfig(rows=ROWS, cols=COLS)
        rng = np.random.default_rng(seed)
        patterns = [random_pattern(cfg.rows, cfg.cols, k=PATTERN_SIZE, rng=rng)
                    for _ in range(N_PATTERNS)]
        t = PatternMemory(cfg=cfg, seed=seed,
                          write_steps=WRITE_STEPS, write_lr=WRITE_LR,
                          wta_k=WTA_K, wta_suppression=WTA_SUPP, rule="covariance")
        u = PatternMemory(cfg=cfg, seed=seed,
                          write_steps=0, write_lr=0.0,
                          wta_k=WTA_K, wta_suppression=WTA_SUPP, rule="covariance")
        for p in patterns:
            t.store(p); u.store(p)
        a_t = _accuracy(t, patterns, DROP_FRAC,
                        np.random.default_rng(seed*31+7), N_TRIALS)
        a_u = _accuracy(u, patterns, DROP_FRAC,
                        np.random.default_rng(seed*31+7), N_TRIALS)
        cov_lifts.append(a_t - a_u)

    cov_mean = float(np.mean(cov_lifts))
    cov_nonneg = sum(1 for l in cov_lifts if l >= 0)
    assert cov_mean > 0.0, (
        f"covariance rule should give non-harmful (positive) mean lift: "
        f"{cov_mean:+.3f} {cov_lifts}")
    assert cov_nonneg >= 7, (
        f"covariance rule should give >=7/12 non-negative seeds: "
        f"{cov_nonneg}/12 lifts={cov_lifts}")


# ---------------------------------------------------------------------------
# Capacity sweep — how many patterns can the substrate hold?
# ---------------------------------------------------------------------------
def test_capacity_curve_substrate_clearly_above_chance():
    """Capacity probe with stronger statistics than the original draft.

    Runs 8 pre-registered seeds × 30 trials per N, computes mean
    accuracy and a basic 95 % t-style CI lower bound. Asserts that the
    LOWER bound of the CI is clearly above chance for each N — not
    just the point estimate. This is a real inferential gate, not a
    near-noise threshold.

    Empirically (full 8-seed × 30-trial probe):
      N=2: mean ≈ 0.70 (chance 0.50)
      N=4: mean ≈ 0.45 (chance 0.25)
      N=6: mean ≈ 0.30 (chance 0.167)

    So we set thresholds as `chance + clear_margin`, where the margin
    is several CI-standard-errors above chance.
    """
    cfg = FieldConfig(rows=ROWS, cols=COLS)
    seeds = list(range(8))
    target = {2: (0.50, 0.10), 4: (0.25, 0.10), 6: (0.167, 0.07)}

    for n_pat, (chance, margin) in target.items():
        accs = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            patterns = [random_pattern(cfg.rows, cfg.cols, k=PATTERN_SIZE, rng=rng)
                        for _ in range(n_pat)]
            mem = PatternMemory(
                cfg=cfg, seed=seed,
                write_steps=WRITE_STEPS, write_lr=WRITE_LR,
                wta_k=WTA_K, wta_suppression=WTA_SUPP,
                rule="covariance",
            )
            for p in patterns:
                mem.store(p)
            a = _accuracy(mem, patterns, DROP_FRAC,
                          np.random.default_rng(seed*31+7), n_trials=30)
            accs.append(a)
        mean = float(np.mean(accs))
        # Two-sigma lower bound (rough 95 % CI, normal approx)
        sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        se = sd / np.sqrt(len(accs))
        lo = mean - 2 * se
        assert lo > chance + margin, (
            f"N={n_pat}: mean={mean:.3f} 2σ-low={lo:.3f} not clearly > "
            f"chance({chance:.3f}) + margin({margin}); accs={accs}")


# ---------------------------------------------------------------------------
# Kızıl elma capacity threshold — 32 seeds × N=64 patterns @ 56×56 grid
# ---------------------------------------------------------------------------
def test_kizil_elma_capacity_threshold_56_grid_64_patterns():
    """User's frontier criterion ("kızıl elma"):
    >= 32 seeds × 16-64+ patterns × clear positive recall.

    Empirically (full 32-seed × 15-trial × 56×56 probe):
      N=16: mean=1.000  (chance=0.062, ~16× over)
      N=32: mean=0.990  (chance=0.031, ~32× over)
      N=64: mean=0.988  (chance=0.016, ~63× over)
      N=128: mean=0.975 (chance=0.008, ~124× over)

    For CI speed this regression test runs 8 seeds × 10 trials at
    56×56 × N=64 — much smaller but still asserts the substrate
    clears the kızıl elma capacity threshold by a wide margin
    (mean ≥ 0.85 vs chance 0.016).
    """
    GRID, N_PAT = 56, 64
    pat_size = max(4, int(0.05 * GRID * GRID))
    wta_k = max(pat_size + 2, int(0.075 * GRID * GRID))
    cfg = FieldConfig(rows=GRID, cols=GRID)
    accs = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        patterns = [random_pattern(GRID, GRID, k=pat_size, rng=rng)
                    for _ in range(N_PAT)]
        mem = PatternMemory(
            cfg=cfg, seed=seed, write_steps=20, write_lr=0.30,
            wta_k=wta_k, wta_suppression=0.3, rule="hebb")
        for p in patterns:
            mem.store(p)
        accs.append(_accuracy(mem, patterns, DROP_FRAC,
                              np.random.default_rng(seed*31+7), n_trials=10))
    mean = float(np.mean(accs))
    sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    se = sd / np.sqrt(len(accs))
    lo = mean - 2 * se
    chance = 1.0 / N_PAT
    assert mean >= 0.85, (
        f"capacity at 56×56 N=64 collapsed: mean={mean:.3f} "
        f"(should be ≥ 0.85, was 0.988 in full probe); accs={accs}")
    assert lo > chance + 0.5, (
        f"capacity 2σ lower bound not clearly above chance: "
        f"lo={lo:.3f} chance={chance:.4f}; accs={accs}")
