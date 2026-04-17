"""Local plasticity rules for the thermodynamic field.

Two purely local rules — no backprop, no global gradients:

1. hebbian_update: conductivity between adjacent cells increases when both
   are hot (correlated activation) and decays passively otherwise.

2. supervised_nudge: at output cells only, push neighboring conductivities
   toward the target temperature using one-cell-deep error. This is the
   minimal feedback signal — equivalent to a local error spike, not a
   backpropagated gradient.
"""
from __future__ import annotations

import numpy as np

from .field import Field


def hebbian_update(field: Field, lr: float = 0.01, decay: float = 0.001) -> None:
    """Strengthen conductivity where both endpoints are hot together."""
    T = field.T
    co_h = T[:, :-1] * T[:, 1:]
    co_v = T[:-1, :] * T[1:, :]
    field.C_right += lr * co_h - decay * field.C_right
    field.C_down += lr * co_v - decay * field.C_down
    np.clip(field.C_right, 0.05, 1.0, out=field.C_right)
    np.clip(field.C_down, 0.05, 1.0, out=field.C_down)


def gated_hebbian_update(
    field: Field,
    output_positions,
    *,
    target: float | None = None,
    threshold: float = 0.5,
    lr_pos: float = 0.01,
    lr_inh: float = 0.04,
    decay: float = 0.001,
) -> None:
    """User's proposed homeostatic rule:

        Δw = η · (pre · post) − η_inh · (pre · post · [output > threshold])

    When the output cell exceeds the threshold (and we want it low), the
    inhibitory term turns on and weakens the same co-active synapses that
    Hebbian learning would strengthen. This breaks the symmetry between
    the four XOR examples that share the same paths.

    If `target` is provided, the inhibition only triggers when output
    overshoots the target (output > target + small margin), so the rule
    naturally handles both "should be high" and "should be low" cases.
    """
    T = field.T
    co_h = T[:, :-1] * T[:, 1:]
    co_v = T[:-1, :] * T[1:, :]

    out_avg = float(np.mean([T[r, c] for r, c in output_positions]))
    if target is not None:
        # Inhibit only when output is too high relative to target.
        gate = 1.0 if out_avg > (target + 0.1) else 0.0
        # Positive Hebbian only when output is too low relative to target.
        pos_gate = 1.0 if out_avg < (target - 0.1) else 0.5
    else:
        gate = 1.0 if out_avg > threshold else 0.0
        pos_gate = 1.0

    eff_lr = pos_gate * lr_pos - gate * lr_inh

    field.C_right += eff_lr * co_h - decay * field.C_right
    field.C_down += eff_lr * co_v - decay * field.C_down
    np.clip(field.C_right, 0.05, 1.0, out=field.C_right)
    np.clip(field.C_down, 0.05, 1.0, out=field.C_down)


def supervised_nudge(
    field: Field,
    output_positions,
    target_temps,
    lr: float = 0.05,
) -> float:
    """Local error correction at output cells only.

    Returns mean absolute error before the update.
    """
    total_err = 0.0
    for (r, c), target in zip(output_positions, target_temps):
        actual = float(field.T[r, c])
        error = float(target) - actual
        total_err += abs(error)
        # Adjust the four incoming conductivities, scaled by neighbor temperature.
        # If error > 0 and a neighbor is hot, boost the conductivity feeding from
        # that neighbor (more heat flows in). If error < 0, weaken those paths.
        if c > 0:
            field.C_right[r, c - 1] += lr * error * float(field.T[r, c - 1])
        if c < field.cfg.cols - 1:
            field.C_right[r, c] += lr * error * float(field.T[r, c + 1])
        if r > 0:
            field.C_down[r - 1, c] += lr * error * float(field.T[r - 1, c])
        if r < field.cfg.rows - 1:
            field.C_down[r, c] += lr * error * float(field.T[r + 1, c])
    np.clip(field.C_right, 0.05, 1.0, out=field.C_right)
    np.clip(field.C_down, 0.05, 1.0, out=field.C_down)
    return total_err / max(len(output_positions), 1)
