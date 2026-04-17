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
from .plasticity import supervised_nudge


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
    lr: float = 0.01,
    decay: float = 0.001,
) -> None:
    """Trace-gated Hebbian: strengthen edges where pre-cell trace × post-cell
    current is high.

    LIMITATION (important): conductivity in this model is a single scalar
    per edge — it represents flow in BOTH directions. We therefore sum
    the forward and backward credit terms, which makes this rule a
    *coactivity-with-temporal-window* rule, not a directional STDP. After
    training on A→B the edge A↔B is strengthened, but at inference time
    presenting A activates B and presenting B activates A roughly
    equally. True directional learning requires directed edges (separate
    C_AB and C_BA) — this is a planned refactor, not yet implemented.

    The trace decay (~0.8) sets the temporal pairing window; events
    farther apart in time get less credit. The rule does learn that
    two events are temporally associated (above a no-pairing baseline),
    just not which one came first.
    """
    E = trace.E
    T = field.T

    co_h_fwd = E[:, :-1] * T[:, 1:]
    co_h_bwd = E[:, 1:] * T[:, :-1]
    co_v_fwd = E[:-1, :] * T[1:, :]
    co_v_bwd = E[1:, :] * T[:-1, :]

    field.C_right += lr * (co_h_fwd + co_h_bwd) - decay * field.C_right
    field.C_down += lr * (co_v_fwd + co_v_bwd) - decay * field.C_down
    np.clip(field.C_right, 0.05, 1.0, out=field.C_right)
    np.clip(field.C_down, 0.05, 1.0, out=field.C_down)


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
    stdp_lr: float = 0.01,
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
        present_event(field, trace, [B], [1.0], hold=hold)
        stdp_hebbian_update(field, trace, lr=stdp_lr)

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
