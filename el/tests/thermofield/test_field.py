"""Tests for the thermodynamic field substrate."""
from __future__ import annotations

import numpy as np

from el.thermofield import Field, FieldConfig


def test_field_initial_state_is_cold():
    f = Field(FieldConfig(rows=8, cols=8))
    assert f.T.shape == (8, 8)
    assert float(f.T.sum()) == 0.0


def test_inject_sets_temperature():
    f = Field(FieldConfig(rows=6, cols=6))
    f.inject([(0, 0), (5, 5)], [1.0, 0.5])
    assert f.T[0, 0] == 1.0
    assert f.T[5, 5] == 0.5


def test_diffusion_spreads_heat_outward():
    # Use low diffusion + zero nonlinearity for clean conservation behavior.
    cfg = FieldConfig(rows=11, cols=11, n_steps=20, diffusion_rate=0.1, nonlinear_alpha=0.0)
    f = Field(cfg, seed=42)
    f.inject([(5, 5)], [1.0], clamp=False)  # one-shot pulse
    energy_before = f.total_energy()
    f.relax()
    # Heat spread to neighbors (center cooled, neighbors warmed)
    assert f.T[5, 5] < 1.0
    assert f.T[4, 5] > 0.0 or f.T[6, 5] > 0.0
    # Total energy decreased due to decay (linear regime is conservative)
    assert 0.0 < f.total_energy() < energy_before


def test_clamped_input_persists_across_steps():
    f = Field(FieldConfig(rows=6, cols=6, n_steps=20), seed=0)
    f.inject([(0, 0)], [1.0], clamp=True)
    f.relax()
    assert abs(float(f.T[0, 0]) - 1.0) < 1e-5


def test_field_is_bounded():
    f = Field(FieldConfig(rows=6, cols=6))
    f.inject([(2, 2)], [10.0])  # over-inject — should be clipped to 1.0
    f.relax(n_steps=30)
    assert float(f.T.max()) <= 1.0 + 1e-6
    assert float(f.T.min()) >= 0.0 - 1e-6


def test_conductivity_arrays_have_correct_shape():
    cfg = FieldConfig(rows=7, cols=9)
    f = Field(cfg)
    assert f.C_right.shape == (7, 8)
    assert f.C_down.shape == (6, 9)


def test_isolated_cells_dont_share_heat_when_conductivity_zero():
    f = Field(FieldConfig(rows=5, cols=5, n_steps=10))
    f.C_right[:] = 0.0
    f.C_down[:] = 0.0
    f.inject([(0, 0)], [1.0])
    f.relax()
    # With zero conductivity, only decay reduces (0,0); rest stay zero
    assert f.T[0, 0] > 0.5
    rest = f.T.copy()
    rest[0, 0] = 0.0
    assert float(rest.sum()) < 1e-6
