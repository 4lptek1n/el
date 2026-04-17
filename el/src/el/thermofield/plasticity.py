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


def covariance_update(
    field: Field,
    lr: float = 0.01,
    decay: float = 0.001,
    baseline: float | None = None,
) -> None:
    """Hebb + anti-Hebb in a single covariance rule (BCM-flavoured).

        Δw = lr · (T_a − μ) · (T_b − μ) − decay · w

    where μ is either a fixed `baseline` or the per-step mean of T.
    Both endpoints above μ → strengthen (Hebbian).
    One above, one below → weaken (anti-Hebbian, decorrelating).
    Both below → tiny strengthening of "cold-together" edges
    (mathematically the rule allows it; in practice T≥0 keeps the
    magnitude small).

    This is the principled fix for pattern-memory smear: pure Hebb
    saturates every edge that the diffusion ever touches, while the
    covariance rule actively *separates* coactive from non-coactive
    pairs. Combine with k-WTA on T for sparse-coding regime.
    """
    T = field.T
    mu = float(T.mean()) if baseline is None else float(baseline)
    dev = T - mu
    cov_h = dev[:, :-1] * dev[:, 1:]
    cov_v = dev[:-1, :] * dev[1:, :]
    field.C_right += lr * cov_h - decay * field.C_right
    field.C_down += lr * cov_v - decay * field.C_down
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


def lateral_inhibition_update(
    field: Field,
    input_positions,
    *,
    target: float,
    target_low_thresh: float = 0.3,
    co_active_thresh: float = 0.5,
    lr: float = 0.08,
) -> None:
    """Mutual-suppression rule between simultaneously-active inputs.

    Biologically inspired by lateral inhibitory interneurons: when several
    input cells fire together, weaken their downward (toward-output) paths
    so the combined signal does not reach the output. Only triggers when
    the target says the output should stay low — this carves a (1,1)-
    specific shunt without disturbing single-input cases.

    Mechanism: if the minimum input temperature exceeds `co_active_thresh`
    (i.e. ALL inputs are active) and target is below `target_low_thresh`,
    decrease C_down at each input cell proportionally to co-activation.
    """
    if target > target_low_thresh:
        return
    T = field.T
    acts = np.array([T[r, c] for r, c in input_positions])
    co_active = float(np.min(acts))  # AND-like across inputs
    if co_active < co_active_thresh:
        return
    # Weaken C_down across the entire band spanned by inputs (the row
    # between them), so heat that diffuses laterally cannot find an
    # alternate downward path. This is the destructive-interference path.
    rows = sorted({r for r, _ in input_positions})
    cols = [c for _, c in input_positions]
    c_min, c_max = min(cols), max(cols)
    for r in rows:
        if r < field.C_down.shape[0]:
            field.C_down[r, c_min:c_max + 1] -= lr * co_active
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
