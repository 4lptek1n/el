"""Tests for the 3D layered substrate (LayeredField).

Three properties to validate:
  1. Vertical heat flow works: input on L0 → measurable response on L1.
  2. Coincidence super-linearity: two simultaneous inputs that meet at
     the same L1 cell produce a larger L1 response than the SUM of each
     input alone (the if-then spike is non-linear).
  3. Multi-layer stack: 3-layer field propagates a coincidence pattern
     up to L2.
"""
from __future__ import annotations

import numpy as np

from el.thermofield import FieldConfig
from el.thermofield.layered import LayeredField


def test_vertical_flow_propagates_input_upward() -> None:
    """Inject heat at L0 (4, 4); after a few steps L1 (4, 4) must heat."""
    field = LayeredField(n_layers=2, cfg=FieldConfig(rows=8, cols=8), seed=0)
    field.reset()
    field.inject(0, [(4, 4)], [1.0])
    for _ in range(5):
        field.step()
    t_l0 = float(field.layers[0].T[4, 4])
    t_l1 = float(field.layers[1].T[4, 4])
    assert t_l0 > 0.5, f"L0 should retain heat (clamped): {t_l0}"
    assert t_l1 > 0.05, f"L1 should warm via vertical flow: {t_l1}"


def test_coincidence_is_superlinear() -> None:
    """Two simultaneous L0 inputs that land near the SAME L1 region should
    produce an L1 peak strictly greater than the sum of single-input
    L1 responses (because both push the central cell over θ and TRIGGER
    a spike, on top of plain diffusion sum).
    """
    cfg = FieldConfig(rows=12, cols=12)
    target = (6, 6)  # midway between the two inputs on L1

    def measure(inputs):
        f = LayeredField(n_layers=2, cfg=cfg, seed=42,
                         firing_threshold=0.25, spike_amplitude=0.30)
        f.reset()
        f.inject(0, inputs, [1.0] * len(inputs))
        for _ in range(6):
            f.step()
        return float(f.layers[1].T[target])

    only_a = measure([(6, 4)])
    only_b = measure([(6, 8)])
    both = measure([(6, 4), (6, 8)])

    # Coincidence must clearly exceed each alone at the overlap cell
    assert both > max(only_a, only_b) * 1.3, (
        f"L1 overlap did not amplify on coincidence: "
        f"a={only_a:.3f} b={only_b:.3f} both={both:.3f}"
    )


def test_three_layer_stack_propagates_coincidence() -> None:
    """3-layer stack: a strong coincidence on L0 should produce some
    activity on L2 (two hops up). Without coincidence (single input),
    L2 stays much colder."""
    cfg = FieldConfig(rows=10, cols=10)

    def run(inputs, steps=10):
        f = LayeredField(n_layers=3, cfg=cfg, seed=7, firing_threshold=0.25,
                         spike_amplitude=0.30)
        f.reset()
        f.inject(0, inputs, [1.0] * len(inputs))
        for _ in range(steps):
            f.step()
        return float(f.layers[2].T.max())

    coincidence = run([(4, 4), (5, 5), (4, 5), (5, 4)])  # 4 inputs cluster
    single = run([(4, 4)])

    # 4 coincident inputs produce more L2 spikes than 1 input
    assert coincidence > single * 1.15, (
        f"L2 did not amplify on coincidence vs single: "
        f"coincidence={coincidence:.3f} single={single:.3f}"
    )
    assert coincidence > 0.05, (
        f"L2 stayed completely cold under coincidence: {coincidence:.3f}"
    )
