"""Tests for local plasticity rules."""
from __future__ import annotations

import numpy as np

from el.thermofield import Field, FieldConfig, hebbian_update, supervised_nudge


def test_hebbian_strengthens_co_active_pairs():
    f = Field(FieldConfig(rows=4, cols=4), seed=0)
    # Make two adjacent cells both hot
    f.T[1, 1] = 0.9
    f.T[1, 2] = 0.9
    c_before = float(f.C_right[1, 1])
    hebbian_update(f, lr=0.1, decay=0.0)
    c_after = float(f.C_right[1, 1])
    assert c_after > c_before


def test_hebbian_decays_inactive_pairs():
    f = Field(FieldConfig(rows=4, cols=4), seed=0)
    f.T[:] = 0.0  # all cold
    c_before = f.C_right.copy()
    hebbian_update(f, lr=0.0, decay=0.1)
    assert float((c_before - f.C_right).mean()) > 0.0


def test_supervised_nudge_returns_error():
    f = Field(FieldConfig(rows=4, cols=4))
    f.T[3, 2] = 0.1
    err = supervised_nudge(f, [(3, 2)], [1.0], lr=0.0)
    assert abs(err - 0.9) < 1e-5


def test_conductivity_stays_bounded():
    f = Field(FieldConfig(rows=4, cols=4), seed=0)
    f.T[:] = 1.0
    for _ in range(50):
        hebbian_update(f, lr=0.5, decay=0.0)
    assert float(f.C_right.max()) <= 1.0
    assert float(f.C_right.min()) >= 0.05


# ---------------------------------------------------------------------------
# Covariance rule (Hebb + anti-Hebb)
# ---------------------------------------------------------------------------
def test_covariance_rule_strengthens_coactive_weakens_decorrelated():
    """The covariance rule should INCREASE C between two cells that are
    both above the mean, and DECREASE C between a cell above the mean
    and a cell below the mean. Pure Hebbian cannot do the latter.
    """
    from el.thermofield.field import Field, FieldConfig
    from el.thermofield.plasticity import covariance_update
    cfg = FieldConfig(rows=1, cols=4)
    f = Field(cfg, seed=0)
    # Make C uniform so we can read changes cleanly
    f.C_right[:] = 0.5
    # Pattern: T = [0.9, 0.9, 0.0, 0.0]
    # Mean = 0.45 → both cells 0,1 above; both 2,3 below; edge 1<->2 crosses
    f.T[0, :] = [0.9, 0.9, 0.0, 0.0]
    c_before = f.C_right[0].copy()
    covariance_update(f, lr=0.5, decay=0.0)
    c_after = f.C_right[0]
    # Edge 0<->1 (both above mean): C should grow
    assert c_after[0] > c_before[0], (
        f"coactive edge should strengthen: {c_before[0]} -> {c_after[0]}")
    # Edge 1<->2 (one above, one below mean): C should shrink
    assert c_after[1] < c_before[1], (
        f"decorrelated edge should weaken: {c_before[1]} -> {c_after[1]}")
    # Edge 2<->3 (both below mean by symmetric amount): also positive
    # covariance, so should grow (or stay) — never shrink
    assert c_after[2] >= c_before[2] - 1e-6


def test_covariance_rule_clips_to_valid_range():
    """C must stay within [0.05, 1.0] regardless of input magnitudes."""
    from el.thermofield.field import Field, FieldConfig
    from el.thermofield.plasticity import covariance_update
    cfg = FieldConfig(rows=4, cols=4)
    f = Field(cfg, seed=0)
    f.T[:] = np.random.default_rng(0).uniform(0, 1, f.T.shape).astype(np.float32)
    for _ in range(50):
        covariance_update(f, lr=2.0, decay=0.0)  # extreme LR
    assert f.C_right.min() >= 0.05 - 1e-6
    assert f.C_right.max() <= 1.0 + 1e-6
    assert f.C_down.min() >= 0.05 - 1e-6
    assert f.C_down.max() <= 1.0 + 1e-6
