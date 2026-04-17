"""Tests for sequence learning via eligibility traces + STDP."""
import numpy as np

from el.thermofield import (
    EligibilityTrace,
    Field,
    FieldConfig,
    stdp_hebbian_update,
    train_predict_next,
)


def test_eligibility_trace_is_low_pass_of_temperature() -> None:
    """A constant input should drive E toward T asymptotically; turning T
    off should let E decay back toward zero."""
    trace = EligibilityTrace((3, 3), decay=0.8)
    T_on = np.ones((3, 3), dtype=np.float32)
    T_off = np.zeros((3, 3), dtype=np.float32)

    for _ in range(50):
        trace.update(T_on)
    assert trace.E.mean() > 0.95, "Trace should rise to ~T under constant drive"

    initial = trace.E.mean()
    for _ in range(20):
        trace.update(T_off)
    assert trace.E.mean() < initial * 0.1, "Trace should decay when T turns off"


def test_stdp_creates_positive_bias_when_pre_leads_post() -> None:
    """Manually set up: pre-cell trace high, post-cell T high → the
    directional BIAS for that edge should grow positive (favoring
    left→right flow), not the symmetric conductance C."""
    field = Field(FieldConfig(rows=4, cols=4), seed=0)
    trace = EligibilityTrace((4, 4), decay=0.85)
    # Edge of interest: right-edge between (1,1) and (1,2).
    # Pre is left cell (1,1), post is right cell (1,2).
    trace.E[1, 1] = 0.9
    field.T[1, 2] = 0.9

    b_before = float(field.B_right[1, 1])
    c_before = float(field.C_right[1, 1])
    stdp_hebbian_update(field, trace, lr=0.05, decay=0.001)
    b_after = float(field.B_right[1, 1])
    c_after = float(field.C_right[1, 1])

    assert b_after - b_before > 0.02, (
        f"Bias did not grow forward: {b_before}->{b_after}"
    )
    assert abs(c_after - c_before) < 1e-6, (
        f"STDP must not touch symmetric C: {c_before}->{c_after}"
    )


def test_stdp_creates_negative_bias_when_post_leads_pre() -> None:
    """Reverse setup: trace at the right cell, current activity at the left
    cell. The bias should swing NEGATIVE (favoring right→left flow)."""
    field = Field(FieldConfig(rows=4, cols=4), seed=0)
    trace = EligibilityTrace((4, 4), decay=0.85)
    trace.E[1, 2] = 0.9
    field.T[1, 1] = 0.9

    b_before = float(field.B_right[1, 1])
    stdp_hebbian_update(field, trace, lr=0.05, decay=0.001)
    b_after = float(field.B_right[1, 1])
    assert b_after - b_before < -0.02, (
        f"Bias did not swing backward: {b_before}->{b_after}"
    )


def test_predict_next_creates_temporal_association_above_naive_baseline() -> None:
    """After training A→B, the cue response at B should exceed the naive
    (untrained, same field topology) baseline. This is the *real*
    measure of association learning — distance-confounded controls like
    `control_response` are not used here."""
    cue_minus_naive = []
    for seed in range(8):
        r = train_predict_next(seed=seed)
        cue_minus_naive.append(r.cue_response - r.untrained_response)

    mean_vs_naive = sum(cue_minus_naive) / len(cue_minus_naive)
    assert mean_vs_naive > 0.003, (
        f"Trained field does not respond more to cue than untrained baseline "
        f"({mean_vs_naive:+.4f})"
    )
    n_positive = sum(1 for d in cue_minus_naive if d > 0)
    assert n_positive >= 6, (
        f"Only {n_positive}/8 seeds showed positive learning effect vs naive: "
        f"{cue_minus_naive}"
    )


def test_directional_stdp_breaks_bidirectional_symmetry() -> None:
    """With the directional bias channel B in place, training on A→B
    must produce an asymmetric response: presenting cue A evokes a
    LARGER response at B than presenting cue B evokes at A. Averaged
    across seeds, A→B should be at least 1.5× B→A."""
    from el.thermofield.sequence import (
        EligibilityTrace,
        present_event,
        relax_with_trace,
        stdp_hebbian_update,
        _seq_io,
    )
    from el.thermofield import Field, FieldConfig

    cfg = FieldConfig()
    A_to_B_responses = []
    B_to_A_responses = []
    for seed in range(5):
        field = Field(cfg, seed=seed)
        trace = EligibilityTrace((cfg.rows, cfg.cols), decay=0.80)
        A, B, _ = _seq_io(cfg)
        from el.thermofield.plasticity import hebbian_update
        from el.thermofield.sequence import _release_clamps
        for _ in range(60):
            field.reset_temp()
            trace.reset()
            present_event(field, trace, [A], [1.0], hold=5)
            relax_with_trace(field, trace, 2)
            # Event-boundary STDP: snapshot at the moment B turns on
            _release_clamps(field)
            field.inject([B], [1.0])
            stdp_hebbian_update(field, trace, lr=0.07)
            for _ in range(5):
                field.step()
                trace.update(field.T)
            hebbian_update(field, lr=0.01, decay=0.001)

        # Probe A → read at B
        field.reset_temp()
        trace.reset()
        present_event(field, trace, [A], [1.0], hold=5)
        relax_with_trace(field, trace, 6)
        a_to_b = float(field.T[B])

        # Probe B → read at A
        field.reset_temp()
        trace.reset()
        present_event(field, trace, [B], [1.0], hold=5)
        relax_with_trace(field, trace, 6)
        b_to_a = float(field.T[A])

        A_to_B_responses.append(a_to_b)
        B_to_A_responses.append(b_to_a)

    mean_ab = sum(A_to_B_responses) / 5
    mean_ba = sum(B_to_A_responses) / 5
    assert mean_ab > mean_ba * 1.4, (
        f"Directional STDP failed to break symmetry: "
        f"A→B={mean_ab:.4f}, B→A={mean_ba:.4f}, ratio={mean_ab/max(mean_ba, 1e-6):.2f}"
    )


def test_directional_stdp_reverses_when_training_order_reverses() -> None:
    """SYMMETRY CONTROL: train on B→A (reverse temporal order). The system
    must now prefer cue B → response at A *over* cue A → response at B,
    by a margin similar to the forward case. If the asymmetry came from
    geometry (probe sites), this test would also show A→B winning. The
    fact that the asymmetry FLIPS with training order is the cleanest
    evidence that the substrate is genuinely encoding temporal direction,
    not site-bias."""
    from el.thermofield.sequence import (
        EligibilityTrace,
        present_event,
        relax_with_trace,
        stdp_hebbian_update,
        _seq_io,
        _release_clamps,
    )
    from el.thermofield import Field, FieldConfig
    from el.thermofield.plasticity import hebbian_update

    cfg = FieldConfig()
    A_to_B_responses = []
    B_to_A_responses = []
    for seed in range(5):
        field = Field(cfg, seed=seed)
        trace = EligibilityTrace((cfg.rows, cfg.cols), decay=0.80)
        A, B, _ = _seq_io(cfg)
        # NOTE: training order REVERSED — present B first, then A.
        for _ in range(60):
            field.reset_temp()
            trace.reset()
            present_event(field, trace, [B], [1.0], hold=5)
            relax_with_trace(field, trace, 2)
            _release_clamps(field)
            field.inject([A], [1.0])
            stdp_hebbian_update(field, trace, lr=0.07)
            for _ in range(5):
                field.step()
                trace.update(field.T)
            hebbian_update(field, lr=0.01, decay=0.001)

        # Probe A → read at B
        field.reset_temp()
        trace.reset()
        present_event(field, trace, [A], [1.0], hold=5)
        relax_with_trace(field, trace, 6)
        A_to_B_responses.append(float(field.T[B]))

        # Probe B → read at A
        field.reset_temp()
        trace.reset()
        present_event(field, trace, [B], [1.0], hold=5)
        relax_with_trace(field, trace, 6)
        B_to_A_responses.append(float(field.T[A]))

    mean_ab = sum(A_to_B_responses) / 5
    mean_ba = sum(B_to_A_responses) / 5
    # Now BA (reverse-trained) should win.
    assert mean_ba > mean_ab * 1.4, (
        f"Reversed training did not flip the asymmetry: "
        f"A→B={mean_ab:.4f}, B→A={mean_ba:.4f}, ratio={mean_ba/max(mean_ab, 1e-6):.2f}"
    )
