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
