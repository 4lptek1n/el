"""Active Inference planner — Expected Free Energy minimization.

The Free Energy Principle (Friston 2010) reframes choice as minimizing
**expected free energy** across candidate actions:

    EFE(a | s) = -E[ pragmatic_value(a, s) ]  -  E[ epistemic_value(a, s) ]

Where:

* ``pragmatic_value`` is goal-attainment (here: predicted reward).
* ``epistemic_value`` is information gain (here: 1 - confidence). The
  agent is rewarded for trying actions whose outcome it doesn't yet
  know — principled exploration, not epsilon-greedy hacks.

The planner takes a list of candidate (intent, actions) plans and
returns them ranked by EFE (lower = better).

This is a small but real implementation of an idea LLMs do not have:
LLMs choose by next-token probability, which conflates "I'm sure" with
"this maximizes outcome." Active Inference splits the two and rewards
useful uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..intent import Intent
from ..primitives import Action
from .hdc import HDC
from .world import WorldModel


@dataclass
class EFEScore:
    """One scored plan."""

    intent: Intent
    actions: tuple[Action, ...]
    pragmatic_value: float  # expected reward (higher = better)
    epistemic_value: float  # info gain (higher = better)
    confidence: float       # in [0, 1]
    n_supporting: int       # how many past experiences voted
    efe: float              # = -(w_p * pragmatic + w_e * epistemic), lower = better

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "pragmatic_value": round(self.pragmatic_value, 4),
            "epistemic_value": round(self.epistemic_value, 4),
            "confidence": round(self.confidence, 4),
            "n_supporting": self.n_supporting,
            "efe": round(self.efe, 4),
        }


class ActiveInferencePlanner:
    """Score and rank candidate plans by Expected Free Energy.

    Hyper-parameters:

    * ``w_pragmatic``: weight on predicted reward (default 1.0).
    * ``w_epistemic``: weight on uncertainty/info gain (default 0.4).
    * ``novelty_bonus``: extra value when the world model has zero
      experience for a (state, action) pair — true cold-start
      exploration.
    """

    def __init__(
        self,
        world: WorldModel,
        hdc: HDC | None = None,
        *,
        w_pragmatic: float = 1.0,
        w_epistemic: float = 0.4,
        novelty_bonus: float = 0.2,
    ):
        self.world = world
        self.hdc = hdc if hdc is not None else world.hdc
        self.w_pragmatic = w_pragmatic
        self.w_epistemic = w_epistemic
        self.novelty_bonus = novelty_bonus

    def score(self, intent: Intent, actions: tuple[Action, ...]) -> EFEScore:
        state_hv = self.hdc.encode_intent_atoms(intent.verb, intent.obj, intent.scope)
        if not actions:
            # No actions = pure no-op. Pragmatic 0, epistemic 0.
            return EFEScore(
                intent=intent,
                actions=(),
                pragmatic_value=0.0,
                epistemic_value=0.0,
                confidence=0.0,
                n_supporting=0,
                efe=0.0,
            )

        # Bundle all action hypervectors into one composite action HV.
        action_hvs = [
            self.hdc.encode_action(a.name, a.kwargs) for a in actions
        ]
        # Sequential composition: bind sequential actions with permute order.
        from .hdc import bundle, permute

        composed_action_hv = bundle(
            [permute(hv, k=i) for i, hv in enumerate(action_hvs)]
        )

        pred = self.world.predict(state_hv, composed_action_hv)
        pragmatic = pred.expected_reward
        # Epistemic value = how much the world model is unsure. When
        # nothing supports the prediction, full novelty bonus applies.
        epistemic = pred.uncertainty
        if pred.n_supporting == 0:
            epistemic = max(epistemic, 1.0)
            pragmatic += self.novelty_bonus  # encourage cold-start trial

        efe = -(self.w_pragmatic * pragmatic + self.w_epistemic * epistemic)
        return EFEScore(
            intent=intent,
            actions=actions,
            pragmatic_value=pragmatic,
            epistemic_value=epistemic,
            confidence=pred.confidence,
            n_supporting=pred.n_supporting,
            efe=efe,
        )

    def rank(
        self,
        candidates: list[tuple[Intent, tuple[Action, ...]]],
    ) -> list[EFEScore]:
        scored = [self.score(intent, actions) for intent, actions in candidates]
        scored.sort(key=lambda s: s.efe)
        return scored
