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


def test_stdp_strengthens_edge_when_pre_leads_post() -> None:
    """Manually set up: pre-cell trace high, post-cell T high → that
    edge's conductivity should grow more than an uninvolved edge."""
    field = Field(FieldConfig(rows=4, cols=4), seed=0)
    trace = EligibilityTrace((4, 4), decay=0.85)
    # Edge of interest: (1,1)<->(1,2). Set pre-trace at (1,1), post-T at (1,2).
    trace.E[1, 1] = 0.9
    field.T[1, 2] = 0.9

    c_before = float(field.C_right[1, 1])
    other_before = float(field.C_right[2, 2])
    stdp_hebbian_update(field, trace, lr=0.05, decay=0.001)

    c_after = float(field.C_right[1, 1])
    other_after = float(field.C_right[2, 2])
    # Targeted edge grows; uninvolved edge only suffers passive decay.
    assert c_after - c_before > 0.02, f"Targeted edge did not grow: {c_before}->{c_after}"
    assert other_after - other_before < 0.0


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


def test_known_limitation_association_is_bidirectional_not_directional() -> None:
    """Documenting a known LIMITATION of the current STDP rule: because
    conductivity is symmetric per edge, training on A→B produces an
    association that works in BOTH directions. After training,
    presenting B alone should evoke an elevated response at A that is
    comparable to (within 50% of) the response at B from cue A.

    This test exists to keep the limitation visible. When directed
    conductivities are added, this test should be REPLACED with one
    that asserts the opposite (cue A → B >> cue B → A)."""
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
        for _ in range(30):
            field.reset_temp()
            trace.reset()
            present_event(field, trace, [A], [1.0], hold=5)
            relax_with_trace(field, trace, 2)
            present_event(field, trace, [B], [1.0], hold=5)
            stdp_hebbian_update(field, trace, lr=0.01)

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
    # Both directions show similar response — symmetric edge cannot
    # represent direction.
    ratio = min(mean_ab, mean_ba) / max(mean_ab, mean_ba)
    assert ratio > 0.5, (
        f"Symmetric edge should give bidirectional association: "
        f"A→B={mean_ab:.4f}, B→A={mean_ba:.4f}, ratio={ratio:.2f}"
    )
