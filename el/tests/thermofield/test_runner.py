"""Tests for thermofield training behavior.

Honest finding: the state-dependent conductivity gives the field a
naturally nonlinear, XOR-shaped response WITHOUT any training. The current
local plasticity rules (Hebbian + supervised nudge) reliably execute and
keep the field stable, but do not yet improve XOR accuracy beyond chance
when judged with a fixed 0.5 threshold. This is a real research finding
worth documenting rather than hiding.
"""
from __future__ import annotations

from el.thermofield import (
    Field,
    FieldConfig,
    evaluate,
    make_xor_dataset,
    train_xor,
)


def test_natural_xor_shape_without_training():
    """The untrained field's response is monotonic and nontrivial across XOR
    inputs. The (1,1) case is suppressed relative to (0,1) and (1,0) due to
    short-circuit between two hot inputs — the signature of XOR-shape."""
    cfg = FieldConfig(rows=7, cols=7, n_steps=80)
    f = Field(cfg, seed=0)
    in_pos = [(1, 1), (1, 5)]
    out_pos = [(3, 3)]

    outs = {}
    for inp in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]:
        f.reset_temp()
        f.inject(in_pos, list(inp))
        f.relax()
        outs[inp] = float(f.read(out_pos)[0])

    # Cold input pair gives essentially zero output.
    assert outs[(0.0, 0.0)] < 0.05
    # Single-hot inputs give clear positive output.
    assert outs[(0.0, 1.0)] > 0.2
    assert outs[(1.0, 0.0)] > 0.2
    # Both-hot suppressed below single-hot — the XOR signature.
    assert outs[(1.0, 1.0)] < outs[(0.0, 1.0)]
    assert outs[(1.0, 1.0)] < outs[(1.0, 0.0)]


def test_xor_training_runs_and_stays_bounded():
    """Training executes without numerical issues."""
    result = train_xor(epochs=100, seed=0)
    assert all(0.0 <= h < 2.0 for h in result.history)
    # Outputs always within [0, 1]
    for r in evaluate(result.field, make_xor_dataset):
        assert 0.0 <= r["output"] <= 1.0


def test_xor_training_does_not_collapse_field():
    """Field doesn't lose all dynamics after training (a common failure mode
    of poorly designed plasticity rules)."""
    result = train_xor(epochs=200, seed=0)
    stats = result.field.stats()
    # Conductivity has nonzero variance — the field hasn't collapsed to uniform.
    assert stats["C_right_std"] > 0.01 or stats["C_down_std"] > 0.01
