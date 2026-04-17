"""Sequence learning via eligibility traces and STDP-like plasticity.

The static field can solve XOR but has no memory of *order*. To learn
temporal sequences we add two ingredients on top of the existing field:

1. **Eligibility trace E[r,c]**: a per-cell low-pass of recent activity.
   `E_{t+1} = decay * E_t + (1 - decay) * T_t`. The trace remembers
   "this cell was hot recently" even after T decays back to zero.

2. **STDP-like Hebbian on conductivity**: instead of `pre_now * post_now`,
   use `pre_trace * post_now`. This strengthens edges where the pre-cell
   fired *before* the post-cell — a causal, temporally-asymmetric rule.

Combined with the field's existing state-dependent (nonlinear) diffusion,
this gives the system a working memory window of ~1/(1 - decay) steps
and lets it learn order-sensitive mappings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import Field, FieldConfig
from .plasticity import hebbian_update, supervised_nudge


class EligibilityTrace:
    """Per-cell low-pass filter over recent temperature."""

    def __init__(self, shape: tuple[int, int], decay: float = 0.85):
        self.E = np.zeros(shape, dtype=np.float32)
        self.decay = float(decay)

    def reset(self) -> None:
        self.E[:] = 0.0

    def update(self, T: np.ndarray) -> None:
        self.E = self.decay * self.E + (1.0 - self.decay) * T


def stdp_hebbian_update(
    field: Field,
    trace: EligibilityTrace,
    *,
    lr: float = 0.07,
    decay: float = 0.001,
    bias_clip: float = 0.6,
) -> None:
    """Anti-symmetric STDP on the directional bias channel B.

    Update rule (per edge):
        ΔB = lr · (E_pre · T_post  −  E_post · T_pre)  −  decay · B

    For the right-edge between cells (i,j) and (i,j+1):
        forward = E[i,j] · T[i,j+1]   (left led right)
        backward = E[i,j+1] · T[i,j]  (right led left)
        ΔB_right[i,j] = lr · (forward − backward) − decay · B_right[i,j]

    Positive B favors left→right flow at runtime, negative B favors
    right→left. Decay pulls B back toward 0 when there is no consistent
    temporal lead. The symmetric C channel is NOT touched by this rule —
    that channel is reserved for the standard Hebbian co-activation
    rules in `plasticity.py`. The two channels can therefore be applied
    independently in the same training loop.

    `bias_clip` keeps |B| < bias_clip so that the effective forward and
    backward conductances stay within the valid (0.05, 1.0) range when
    combined with the typical C ≈ 0.3–0.7 range.
    """
    E = trace.E
    T = field.T

    fwd_h = E[:, :-1] * T[:, 1:]   # left led right
    bwd_h = E[:, 1:] * T[:, :-1]   # right led left
    fwd_v = E[:-1, :] * T[1:, :]   # up led down
    bwd_v = E[1:, :] * T[:-1, :]   # down led up

    field.B_right += lr * (fwd_h - bwd_h) - decay * field.B_right
    field.B_down += lr * (fwd_v - bwd_v) - decay * field.B_down
    np.clip(field.B_right, -bias_clip, bias_clip, out=field.B_right)
    np.clip(field.B_down, -bias_clip, bias_clip, out=field.B_down)


def _release_clamps(field: Field) -> None:
    field._clamp_positions = []
    field._clamp_values = []


def present_event(
    field: Field,
    trace: EligibilityTrace,
    positions,
    values,
    *,
    hold: int = 10,
    release_after: bool = True,
) -> None:
    """Inject + clamp `positions=values`, run `hold` diffusion steps,
    update the trace each step, then release the clamps."""
    _release_clamps(field)
    field.inject(positions, values)
    for _ in range(hold):
        field.step()
        trace.update(field.T)
    if release_after:
        _release_clamps(field)


def relax_with_trace(
    field: Field, trace: EligibilityTrace, n_steps: int
) -> None:
    """Free relaxation (no clamps, just diffusion + trace update)."""
    _release_clamps(field)
    for _ in range(n_steps):
        field.step()
        trace.update(field.T)


@dataclass
class SequenceResult:
    field: Field
    trace: EligibilityTrace
    accuracy: float
    details: list[dict]


def _seq_io(cfg: FieldConfig) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return (cue_A, cue_B, output_cell) positions on the grid."""
    a = (1, 1)
    b = (1, cfg.cols - 2)
    out = (cfg.rows - 2, cfg.cols // 2)
    return a, b, out


@dataclass
class PredictNextResult:
    field: Field
    trace: EligibilityTrace
    cue_response: float        # B's temperature after presenting only A
    control_response: float    # B's temperature after presenting only an unrelated cue X
    untrained_response: float  # B's temperature after presenting only A on a fresh field
    discrimination: float      # cue_response - max(control, untrained)


def train_predict_next(
    *,
    epochs: int = 30,
    seed: int = 0,
    cfg: FieldConfig | None = None,
    hold: int = 5,
    gap: int = 2,
    trace_decay: float = 0.80,
    stdp_lr: float = 0.07,
    read_delay: int = 6,
) -> PredictNextResult:
    """Learn that two events A and B are *temporally associated* (no
    supervised target). After training, presenting A alone evokes a
    HIGHER response at B than the same field probed without training.

    HONEST SCOPE: this is symmetric coactivity learning, NOT direction
    learning. After training on A→B, presenting B alone also evokes
    elevated activity at A — the two are bound together in both
    directions because the underlying conductivity is symmetric.
    See `stdp_hebbian_update` docstring for details and the planned
    fix (directed conductivities).

    The cleanest control here is the NAIVE baseline: same field, same
    probe, but never trained. cue_response − untrained_response is
    the real measure of association strength. The control_response
    (probing with an unrelated, geometrically distant cue X) is
    confounded by distance and should not be the primary metric.

    Training (each epoch):
      1. Reset field + trace
      2. Present A clamped for `hold` steps (the cue)
      3. Free relax for `gap` steps (A's trace decays slowly)
      4. Present B clamped for `hold` steps (the consequence)
      5. Apply STDP: edges where pre-trace × post-now is high get
         strengthened. The freshly-present B activity multiplied by A's
         lingering trace strengthens the path A→B.

    Test (presenting A alone, no B, no supervision):
      1. Reset field + trace
      2. Present A clamped for `hold` steps
      3. Free relax for `read_delay` steps
      4. Read T at B's cell — should be elevated relative to controls.

    This is the cleanest possible demonstration that the substrate has
    learned a temporal association: a cue alone evokes the consequence.
    """
    cfg = cfg or FieldConfig()
    field = Field(cfg, seed=seed)
    trace = EligibilityTrace((cfg.rows, cfg.cols), decay=trace_decay)
    A, B, _ = _seq_io(cfg)
    # An "unrelated" cue position used only for the control, far from B.
    X = (cfg.rows - 2, 1)

    for _ in range(epochs):
        field.reset_temp()
        trace.reset()
        present_event(field, trace, [A], [1.0], hold=hold)
        relax_with_trace(field, trace, gap)
        # EVENT-BOUNDARY STDP: at the moment B is first presented, E[A] is
        # high (recently fired, decaying) but E[B] is still ~0. This is
        # the clean "A leads B" snapshot. Setting B's temperature directly
        # before any step and applying STDP once captures the directional
        # signal cleanly. If we let B's clamp run for several steps before
        # STDP, E[B] grows huge and dominates the rule symmetrically.
        _release_clamps(field)
        field.inject([B], [1.0])
        stdp_hebbian_update(field, trace, lr=stdp_lr)
        # Now actually run B's hold so the symmetric Hebbian sees the
        # full coactivity state (heat propagating between A's residual and
        # B's clamp).
        for _ in range(hold):
            field.step()
            trace.update(field.T)
        hebbian_update(field, lr=stdp_lr, decay=0.001)

    def probe(cue_pos) -> float:
        field.reset_temp()
        trace.reset()
        present_event(field, trace, [cue_pos], [1.0], hold=hold)
        relax_with_trace(field, trace, read_delay)
        return float(field.T[B])

    cue_response = probe(A)
    control_response = probe(X)

    # Untrained baseline: a brand-new field with the same seed
    fresh = Field(cfg, seed=seed)
    fresh_trace = EligibilityTrace((cfg.rows, cfg.cols), decay=trace_decay)
    fresh.reset_temp()
    fresh_trace.reset()
    present_event(fresh, fresh_trace, [A], [1.0], hold=hold)
    relax_with_trace(fresh, fresh_trace, read_delay)
    untrained_response = float(fresh.T[B])

    return PredictNextResult(
        field=field,
        trace=trace,
        cue_response=cue_response,
        control_response=control_response,
        untrained_response=untrained_response,
        discrimination=cue_response - max(control_response, untrained_response),
    )
