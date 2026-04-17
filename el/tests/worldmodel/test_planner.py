"""ActiveInferencePlanner: EFE-based ranking sanity."""
from __future__ import annotations

from el.intent import Intent
from el.primitives import Action
from el.worldmodel.hdc import HDC, bundle, permute
from el.worldmodel.planner import ActiveInferencePlanner
from el.worldmodel.world import WorldModel


def _act(name: str, **kw) -> Action:
    return Action(name=name, kwargs=tuple(kw.items()))


def _seed(world: WorldModel, intent: Intent, actions: tuple[Action, ...], reward: float):
    hdc = world.hdc
    state = hdc.encode_intent_atoms(intent.verb, intent.obj, intent.scope)
    a_hvs = [hdc.encode_action(a.name, a.kwargs) for a in actions]
    composed = bundle([permute(hv, k=i) for i, hv in enumerate(a_hvs)])
    outcome = hdc.encode_outcome(ok=reward > 0.5, reward=reward)
    world.observe(state, composed, outcome, reward, ok=reward > 0.5)


def test_high_reward_plan_wins_when_both_known():
    hdc = HDC()
    world = WorldModel(hdc)
    intent = Intent(verb="list", obj="file", scope="")
    good = (_act("file_list", path="."),)
    bad = (_act("noop"),)
    # Seed each plan multiple times so confidence is high for both.
    for _ in range(3):
        _seed(world, intent, good, reward=0.9)
        _seed(world, intent, bad, reward=0.1)
    planner = ActiveInferencePlanner(world)
    ranked = planner.rank([(intent, good), (intent, bad)])
    assert ranked[0].actions == good
    assert ranked[0].efe < ranked[1].efe
    assert ranked[0].pragmatic_value > ranked[1].pragmatic_value


def test_unknown_plan_gets_novelty_bonus():
    hdc = HDC()
    world = WorldModel(hdc)
    intent = Intent(verb="list", obj="file", scope="")
    known = (_act("file_list", path="."),)
    novel = (_act("totally_new_primitive_no_one_has_seen"),)
    # Known plan with mediocre outcome
    for _ in range(3):
        _seed(world, intent, known, reward=0.4)
    planner = ActiveInferencePlanner(world)
    ranked = planner.rank([(intent, known), (intent, novel)])
    novel_score = next(s for s in ranked if s.actions == novel)
    known_score = next(s for s in ranked if s.actions == known)
    # Novel plan must carry epistemic value = 1.0 (full uncertainty)
    assert novel_score.epistemic_value >= 0.9
    # And the novelty bonus pushes its pragmatic above the mediocre known
    # value (0.4) because cold-start reward = bonus only.
    # We check the EFE ordering is such that novel is preferred.
    assert novel_score.efe < known_score.efe


def test_score_dict_serializable():
    hdc = HDC()
    world = WorldModel(hdc)
    planner = ActiveInferencePlanner(world)
    intent = Intent(verb="git_status", obj="", scope="")
    score = planner.score(intent, (_act("git_status"),))
    d = score.to_dict()
    assert "efe" in d and "pragmatic_value" in d and "epistemic_value" in d
