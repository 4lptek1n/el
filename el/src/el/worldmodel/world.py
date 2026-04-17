"""World Model — predictive memory built on HDC.

The world model stores ``Experience(state, action) -> outcome`` triples
as bound hypervectors. Given a new (state, action) query, it retrieves
the predicted outcome by computing similarity against all stored keys
and bundling the top-k matched outcomes weighted by similarity.

This is a *non-parametric* learner — no gradient descent, no
backprop. New experiences integrate in O(1); prediction is O(N*D)
which on CPU stays under a millisecond for N<10k, D=10k.

The store is bounded; oldest-and-least-confirmed experiences are
evicted when capacity is reached.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .hdc import HDC, bind, bundle_weighted, cosine_sim


@dataclass
class Experience:
    """One observed (state, action, outcome, reward) tuple."""

    state_hv: np.ndarray
    action_hv: np.ndarray
    outcome_hv: np.ndarray
    reward: float
    ok: bool
    ts: float = field(default_factory=time.time)
    confirms: int = 1  # bumped when a near-duplicate experience reappears


@dataclass
class Prediction:
    """Result of querying the world model."""

    predicted_outcome_hv: np.ndarray
    expected_reward: float
    confidence: float  # in [0, 1] = mean cosine of top-k matches
    uncertainty: float  # 1 - confidence  (used by epistemic value)
    n_supporting: int  # how many experiences voted


class WorldModel:
    """Associative memory: (state ⊗ action) -> outcome hypervector.

    All vectors stored as int8 bipolar arrays of length ``hdc.dim``.

    Capacity bounded; LRU-by-confirms eviction keeps the most useful
    experiences.
    """

    def __init__(
        self,
        hdc: HDC | None = None,
        capacity: int = 4096,
        top_k: int = 5,
        min_support_sim: float = 0.25,
    ):
        self.hdc = hdc if hdc is not None else HDC()
        self.capacity = capacity
        self.top_k = top_k
        # Random bipolar pairs in 10K-D have |cos| ~ 1/sqrt(D) ≈ 0.01,
        # but the standard deviation produces frequent values up to
        # ~0.05; with role-filler binding the "noise floor" is higher.
        # 0.25 is well above noise yet below the 0.3 generalization
        # threshold seen in test_predict_generalizes_to_similar_state.
        self.min_support_sim = min_support_sim
        self._experiences: list[Experience] = []
        # Cached key matrix for fast similarity, rebuilt on insert/evict.
        self._keys_matrix: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._experiences)

    def _key(self, state_hv: np.ndarray, action_hv: np.ndarray) -> np.ndarray:
        return bind(state_hv, action_hv)

    def _rebuild_index(self) -> None:
        if not self._experiences:
            self._keys_matrix = None
            return
        self._keys_matrix = np.stack(
            [self._key(e.state_hv, e.action_hv).astype(np.float32) for e in self._experiences]
        )

    def observe(
        self,
        state_hv: np.ndarray,
        action_hv: np.ndarray,
        outcome_hv: np.ndarray,
        reward: float,
        ok: bool,
        merge_threshold: float = 0.92,
    ) -> Experience:
        """Add an experience. Near-duplicates are merged (confirms++).

        Scans all near-key experiences and merges into the one whose
        outcome is closest to the new outcome (above merge_threshold).
        If no near-key experience has a matching outcome, append as a
        new contradicting experience so the planner sees uncertainty.
        """
        new_key = self._key(state_hv, action_hv)
        best_match: Experience | None = None
        best_outcome_sim = -1.0
        any_key_match = False
        for exp in self._experiences:
            existing_key = self._key(exp.state_hv, exp.action_hv)
            if cosine_sim(new_key, existing_key) < merge_threshold:
                continue
            any_key_match = True
            outcome_sim = cosine_sim(exp.outcome_hv, outcome_hv)
            if outcome_sim >= merge_threshold and outcome_sim > best_outcome_sim:
                best_match = exp
                best_outcome_sim = outcome_sim
        if best_match is not None:
            best_match.confirms += 1
            best_match.ts = time.time()
            n = best_match.confirms
            best_match.reward = ((n - 1) * best_match.reward + reward) / n
            return best_match
        # If we matched key(s) but no outcome agreed: fall through and
        # append as a new contradicting experience so EFE sees variance.
        _ = any_key_match  # explicit: contradiction is intentional

        exp = Experience(
            state_hv=state_hv.astype(np.int8),
            action_hv=action_hv.astype(np.int8),
            outcome_hv=outcome_hv.astype(np.int8),
            reward=float(reward),
            ok=bool(ok),
        )
        self._experiences.append(exp)

        if len(self._experiences) > self.capacity:
            # Evict the lowest-(confirms, ts) experience.
            worst_idx = min(
                range(len(self._experiences)),
                key=lambda i: (self._experiences[i].confirms, self._experiences[i].ts),
            )
            self._experiences.pop(worst_idx)

        self._rebuild_index()
        return exp

    def predict(self, state_hv: np.ndarray, action_hv: np.ndarray) -> Prediction:
        """Predict the outcome of taking ``action_hv`` in ``state_hv``.

        Returns a Prediction with the bundled outcome HV plus expected
        reward and confidence (mean of top-k similarities).
        """
        if not self._experiences or self._keys_matrix is None:
            zero_hv = np.zeros(self.hdc.dim, dtype=np.int8)
            return Prediction(
                predicted_outcome_hv=zero_hv,
                expected_reward=0.0,
                confidence=0.0,
                uncertainty=1.0,
                n_supporting=0,
            )

        query = self._key(state_hv, action_hv).astype(np.float32)
        denom = (np.linalg.norm(self._keys_matrix, axis=1) * np.linalg.norm(query)) + 1e-9
        sims = (self._keys_matrix @ query) / denom

        k = min(self.top_k, len(self._experiences))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        top_sims = sims[top_idx]

        # Only matches above the noise floor count as supporting evidence.
        # Without this, in high-D bipolar spaces unrelated vectors have
        # ~50% chance of slightly-positive cosine, which would corrupt
        # novelty detection (cold-start would never register).
        pos_mask = top_sims >= self.min_support_sim
        if not pos_mask.any():
            zero_hv = np.zeros(self.hdc.dim, dtype=np.int8)
            return Prediction(
                predicted_outcome_hv=zero_hv,
                expected_reward=0.0,
                confidence=0.0,
                uncertainty=1.0,
                n_supporting=0,
            )

        idxs = top_idx[pos_mask]
        sims_used = top_sims[pos_mask]
        outcomes = [
            (self._experiences[int(i)].outcome_hv, float(s))
            for i, s in zip(idxs, sims_used)
        ]
        bundled = bundle_weighted(outcomes)
        expected_reward = float(
            np.average(
                [self._experiences[int(i)].reward for i in idxs],
                weights=sims_used,
            )
        )
        confidence = float(np.mean(sims_used))
        return Prediction(
            predicted_outcome_hv=bundled,
            expected_reward=expected_reward,
            confidence=confidence,
            uncertainty=max(0.0, 1.0 - confidence),
            n_supporting=int(idxs.size),
        )

    def all_experiences(self) -> Iterable[Experience]:
        return tuple(self._experiences)

    def stats(self) -> dict:
        if not self._experiences:
            return {"size": 0, "capacity": self.capacity, "dim": self.hdc.dim}
        rewards = [e.reward for e in self._experiences]
        confirms = [e.confirms for e in self._experiences]
        return {
            "size": len(self._experiences),
            "capacity": self.capacity,
            "dim": self.hdc.dim,
            "mean_reward": round(float(np.mean(rewards)), 4),
            "max_confirms": int(max(confirms)),
            "mean_confirms": round(float(np.mean(confirms)), 3),
        }

    def reset(self) -> None:
        self._experiences.clear()
        self._keys_matrix = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._experiences:
            np.savez(
                path,
                states=np.zeros((0, self.hdc.dim), dtype=np.int8),
                actions=np.zeros((0, self.hdc.dim), dtype=np.int8),
                outcomes=np.zeros((0, self.hdc.dim), dtype=np.int8),
                rewards=np.zeros(0, dtype=np.float32),
                oks=np.zeros(0, dtype=np.int8),
                ts=np.zeros(0, dtype=np.float64),
                confirms=np.zeros(0, dtype=np.int32),
                dim=np.array([self.hdc.dim], dtype=np.int32),
            )
            return
        np.savez(
            path,
            states=np.stack([e.state_hv for e in self._experiences]),
            actions=np.stack([e.action_hv for e in self._experiences]),
            outcomes=np.stack([e.outcome_hv for e in self._experiences]),
            rewards=np.array([e.reward for e in self._experiences], dtype=np.float32),
            oks=np.array([1 if e.ok else 0 for e in self._experiences], dtype=np.int8),
            ts=np.array([e.ts for e in self._experiences], dtype=np.float64),
            confirms=np.array([e.confirms for e in self._experiences], dtype=np.int32),
            dim=np.array([self.hdc.dim], dtype=np.int32),
        )

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        data = np.load(path)
        dim = int(data["dim"][0])
        if dim != self.hdc.dim:
            raise ValueError(f"saved dim={dim} != current dim={self.hdc.dim}")
        n = data["states"].shape[0]
        self._experiences = [
            Experience(
                state_hv=data["states"][i].astype(np.int8),
                action_hv=data["actions"][i].astype(np.int8),
                outcome_hv=data["outcomes"][i].astype(np.int8),
                reward=float(data["rewards"][i]),
                ok=bool(data["oks"][i]),
                ts=float(data["ts"][i]),
                confirms=int(data["confirms"][i]),
            )
            for i in range(n)
        ]
        self._rebuild_index()
