"""Tests for the inhibitory interneuron population and the two-population
XOR architecture."""
from __future__ import annotations

import numpy as np

from el.thermofield import (
    Field,
    FieldConfig,
    InterneuronConfig,
    Interneurons,
    train_xor_with_interneurons,
)


def test_interneuron_activate_thresholded() -> None:
    cfg = FieldConfig()
    inh = Interneurons(InterneuronConfig(n=1), (cfg.rows, cfg.cols), seed=0)
    inh.w_in[:] = 0.0
    inh.w_in[0, 1, 1] = 0.7
    inh.w_in[0, 1, 5] = 0.7
    inh.theta[0] = 1.0

    T = np.zeros((cfg.rows, cfg.cols), dtype=np.float32)
    assert inh.activate(T)[0] == 0.0  # nothing on

    T[1, 1] = 1.0
    assert inh.activate(T)[0] == 0.0  # one input is below threshold

    T[1, 5] = 1.0
    assert inh.activate(T)[0] > 0.0  # both inputs cross threshold


def test_inhibition_subtracts_at_output() -> None:
    cfg = FieldConfig()
    inh = Interneurons(InterneuronConfig(n=1), (cfg.rows, cfg.cols), seed=0)
    inh.w_in[:] = 0.0
    inh.w_in[0, 1, 1] = 0.7
    inh.w_in[0, 1, 5] = 0.7
    inh.theta[0] = 1.0
    inh.w_out[0] = 5.0

    T = np.zeros((cfg.rows, cfg.cols), dtype=np.float32)
    T[1, 1] = 1.0
    T[1, 5] = 1.0
    inhibition = inh.inhibition(T)
    assert inhibition > 0.5  # strong suppression on coincident input


def test_xor_with_interneurons_solves_4_of_4_across_seeds() -> None:
    """Two-population architecture: thermal field + inhibitory interneuron
    reaches 4/4 XOR accuracy on every seed tested."""
    seeds_perfect = 0
    # Use a larger non-consecutive seed set to guard against luck on
    # neighboring seeds. Architect-suggested check: should be robust.
    seed_set = [0, 1, 2, 3, 7, 11, 17, 23, 31, 42, 67, 99, 137, 200, 314,
                421, 555, 777, 999, 1234]
    for s in seed_set:
        result = train_xor_with_interneurons(epochs=150, seed=s)
        if result.accuracy == 1.0:
            seeds_perfect += 1
    assert seeds_perfect == len(seed_set), (
        f"Expected 4/4 on all {len(seed_set)} seeds, got {seeds_perfect}"
    )


def test_xor_inhibition_only_fires_on_coincident_input() -> None:
    result = train_xor_with_interneurons(epochs=150, seed=0)
    by_input = {tuple(d["input"]): d for d in result.details}
    # Single-input cases should have ~zero inhibition
    assert by_input[(0.0, 1.0)]["inhibition"] < 0.01
    assert by_input[(1.0, 0.0)]["inhibition"] < 0.01
    # Coincident input should have substantial inhibition
    assert by_input[(1.0, 1.0)]["inhibition"] > 0.1


def test_interneuron_receptive_field_is_learned_at_input_cells() -> None:
    """The interneuron must DISCOVER which cells to listen to via local
    Hebbian (no hand-set positions). After training, its receptive-field
    mass should concentrate at the two input cells, with little leakage."""
    result = train_xor_with_interneurons(epochs=150, seed=0)
    w_in = result.interneurons.w_in[0]
    rows, cols = w_in.shape
    in_positions = [(1, 1), (1, cols - 2)]
    rf_at_inputs = sum(float(w_in[r, c]) for r, c in in_positions)
    rf_total = float(w_in.sum())
    # At least 95% of the receptive field mass should be at the input cells.
    assert rf_total > 0.5, f"Receptive field collapsed to ~zero ({rf_total})"
    assert rf_at_inputs / rf_total > 0.95, (
        f"Receptive field not concentrated at inputs: "
        f"{rf_at_inputs:.3f} / {rf_total:.3f}"
    )
