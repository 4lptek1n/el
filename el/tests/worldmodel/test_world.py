"""WorldModel: prediction sanity, persistence, eviction."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from el.worldmodel.hdc import HDC, cosine_sim
from el.worldmodel.world import WorldModel


def _make_triple(hdc: HDC, verb: str, obj: str, scope: str, prim: str, reward: float):
    state = hdc.encode_intent_atoms(verb, obj, scope)
    action = hdc.encode_action(prim, ())
    outcome = hdc.encode_outcome(ok=reward > 0.5, reward=reward)
    return state, action, outcome, reward


def test_predict_with_no_data_returns_zero_confidence():
    wm = WorldModel(HDC())
    state = wm.hdc.encode_intent_atoms("list", "file", "")
    action = wm.hdc.encode_action("file_list", ())
    pred = wm.predict(state, action)
    assert pred.confidence == 0.0
    assert pred.n_supporting == 0


def test_observe_then_predict_recovers_outcome():
    hdc = HDC()
    wm = WorldModel(hdc)
    state, action, outcome, reward = _make_triple(hdc, "list", "file", "", "file_list", 0.9)
    wm.observe(state, action, outcome, reward, ok=True)
    pred = wm.predict(state, action)
    assert pred.n_supporting == 1
    # Predicted outcome must be highly similar to the stored one.
    assert cosine_sim(pred.predicted_outcome_hv, outcome) > 0.9
    assert pred.expected_reward > 0.7


def test_predict_generalizes_to_similar_state():
    hdc = HDC()
    wm = WorldModel(hdc)
    s1, a, o, r = _make_triple(hdc, "list", "file", "this_folder", "file_list", 0.9)
    wm.observe(s1, a, o, r, ok=True)
    # Query a similar but not identical state (different scope)
    s2 = hdc.encode_intent_atoms("list", "file", "downloads")
    pred = wm.predict(s2, a)
    # Should still find the related experience at moderate confidence
    assert pred.n_supporting >= 1
    assert pred.confidence > 0.3


def test_observe_merges_duplicate_experiences():
    hdc = HDC()
    wm = WorldModel(hdc)
    s, a, o, r = _make_triple(hdc, "git_status", "", "", "git_status", 0.95)
    e1 = wm.observe(s, a, o, r, ok=True)
    e2 = wm.observe(s, a, o, r, ok=True)
    assert e1 is e2
    assert e1.confirms == 2
    assert wm.size == 1


def test_capacity_eviction():
    hdc = HDC(dim=2000)  # smaller for speed
    wm = WorldModel(hdc, capacity=10)
    for i in range(25):
        s, a, o, r = _make_triple(hdc, f"verb{i}", "", "", f"prim{i}", 0.5)
        wm.observe(s, a, o, r, ok=True)
    assert wm.size == 10


def test_novel_action_truly_unsupported():
    """Regression: high-D bipolar noise must not register as support.

    Before the min_support_sim floor, ~50% of unrelated actions
    showed n_supporting>0, breaking cold-start novelty detection.
    """
    hdc = HDC()
    wm = WorldModel(hdc)
    s, a, o, r = _make_triple(hdc, "list", "file", "", "file_list", 0.9)
    wm.observe(s, a, o, r, ok=True)
    misses = 0
    trials = 50
    for i in range(trials):
        novel_state = hdc.encode_intent_atoms(f"verb_x{i}", f"obj_y{i}", f"sc_z{i}")
        novel_action = hdc.encode_action(f"prim_z{i}", ())
        pred = wm.predict(novel_state, novel_action)
        if pred.n_supporting == 0:
            misses += 1
    assert misses >= int(trials * 0.9), f"only {misses}/{trials} novel queries detected as novel"


def test_observe_merges_across_multiple_near_keys():
    """Regression: merge scan must look past the first key match.

    A near-key existing experience with contradicting outcome must not
    prevent merging into a later near-key experience whose outcome agrees.
    """
    hdc = HDC()
    wm = WorldModel(hdc)
    s, a, _, _ = _make_triple(hdc, "list", "file", "", "file_list", 0.9)
    bad = hdc.encode_outcome(ok=False, reward=0.0)
    good = hdc.encode_outcome(ok=True, reward=0.95)
    e_bad = wm.observe(s, a, bad, 0.0, ok=False)
    e_good = wm.observe(s, a, good, 0.95, ok=True)
    assert e_bad is not e_good
    assert wm.size == 2
    # Now a third "good" observation should merge into e_good, not append.
    e_good2 = wm.observe(s, a, good, 0.95, ok=True)
    assert e_good2 is e_good
    assert e_good.confirms == 2
    assert wm.size == 2


def test_save_and_load_roundtrip():
    hdc = HDC(dim=2000)
    wm = WorldModel(hdc)
    triples = [
        _make_triple(hdc, "list", "file", "", "file_list", 0.9),
        _make_triple(hdc, "git_status", "", "", "git_status", 0.85),
        _make_triple(hdc, "find", "", "", "grep", 0.7),
    ]
    for s, a, o, r in triples:
        wm.observe(s, a, o, r, ok=True)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "wm.npz"
        wm.save(path)

        wm2 = WorldModel(HDC(dim=2000))
        wm2.load(path)
        assert wm2.size == 3
        # And prediction still works
        s, a, o, _ = triples[0]
        pred = wm2.predict(s, a)
        assert cosine_sim(pred.predicted_outcome_hv, o) > 0.9
