from el.intent import Intent
from el.primitives import PrimResult
from el.rewards import score_outcome


def _ok(name="file_write", stdout="", data=None):
    return PrimResult(name=name, ok=True, duration_ms=10, stdout=stdout, data=data)


def _bad(name="sh"):
    return PrimResult(name=name, ok=False, duration_ms=10, error="boom")


def test_perfect_summarize_gets_high_reward():
    intent = Intent(verb="summarize", obj="folder", scope="this_folder")
    results = [_ok("file_list", data=[{"path": "a"}]), _ok("file_write", data={"path": "s.md"})]
    assert score_outcome(results, intent) >= 0.85


def test_all_failure_zero_reward():
    intent = Intent(verb="list")
    assert score_outcome([_bad()], intent) < 0.3


def test_empty_results_zero():
    assert score_outcome([], Intent(verb="list")) == 0.0
