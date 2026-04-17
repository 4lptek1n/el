"""Hyperdimensional Computing (HDC) primitives.

A hypervector here is a 1-D ``np.int8`` array of bipolar values
(``-1`` / ``+1``) of length ``D`` (default 10000). Operations:

* ``bind(a, b)``      = elementwise multiplication   (binding / "and")
* ``bundle(*xs)``     = elementwise sum + sign       (superposition / "or")
* ``permute(x, k)``   = cyclic rotation by ``k``     (sequence ordering)
* ``cosine_sim(a, b)`` in [-1, 1]                    (similarity)

These satisfy:

* ``bind`` is its own inverse:  ``bind(bind(a, b), b) == a``
* ``bundle`` is approximately information-preserving for moderate counts:
  the bundled vector remains close (cos > 0.3) to each ingredient when
  the bundle size is small relative to ``sqrt(D)``.

The combination of bind + bundle + permute is Turing-complete for
symbolic representation (Kanerva 2009, Plate 1995).

All randomness is seeded by content (``hash(name) -> seed``) so the
same string always maps to the same hypervector across processes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_DIM = 10_000


def _seed_from_name(name: str) -> int:
    """Deterministic 64-bit seed from a string."""
    h = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False) & 0xFFFFFFFF


def random_hv(name: str | None = None, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Random bipolar hypervector. If ``name`` is given, deterministic."""
    if name is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(_seed_from_name(name))
    v = rng.integers(0, 2, size=dim, dtype=np.int8)
    v = (v * 2 - 1).astype(np.int8)
    return v


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise multiplication; XOR-equivalent for bipolar.

    Self-inverse:  bind(bind(a, b), b) == a   (since b * b == 1)
    """
    return (a.astype(np.int16) * b.astype(np.int16)).astype(np.int8)


def bundle(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Majority bundling: elementwise sum, then sign.

    Ties are broken to +1 deterministically. The bundled vector is
    similar (high cosine) to each component when the number of
    components is small.
    """
    vs = list(vectors)
    if not vs:
        raise ValueError("bundle requires at least one vector")
    s = np.zeros(vs[0].shape, dtype=np.int32)
    for v in vs:
        s += v.astype(np.int32)
    out = np.where(s >= 0, np.int8(1), np.int8(-1))
    return out


def bundle_weighted(
    vectors: Iterable[tuple[np.ndarray, float]],
) -> np.ndarray:
    """Weighted bundling with continuous weights, then sign-clipped."""
    vs = list(vectors)
    if not vs:
        raise ValueError("bundle_weighted requires at least one vector")
    s = np.zeros(vs[0][0].shape, dtype=np.float32)
    for v, w in vs:
        s += v.astype(np.float32) * float(w)
    out = np.where(s >= 0, np.int8(1), np.int8(-1))
    return out


def permute(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Cyclic rotation by ``k`` positions. Encodes order."""
    return np.roll(x, k)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Bipolar => norms are sqrt(D)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    na = float(np.linalg.norm(a.astype(np.float32)))
    nb = float(np.linalg.norm(b.astype(np.float32)))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)) / (na * nb))


@dataclass
class HDC:
    """A vocabulary that maps atoms (strings) to hypervectors.

    Atoms are interned: same string -> same hypervector across calls.
    Roles (verb-role, object-role, ...) are also atoms, just with a
    distinguishing prefix.
    """

    dim: int = DEFAULT_DIM
    _vocab: dict[str, np.ndarray] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._vocab is None:
            self._vocab = {}

    def atom(self, name: str) -> np.ndarray:
        v = self._vocab.get(name)
        if v is None:
            v = random_hv(name=name, dim=self.dim)
            self._vocab[name] = v
        return v

    def role(self, name: str) -> np.ndarray:
        return self.atom(f"__role__:{name}")

    def encode_intent_atoms(self, verb: str, obj: str, scope: str) -> np.ndarray:
        """Encode an Intent's symbolic skeleton as a single hypervector.

        Uses role-filler binding: VERB_ROLE * verb + OBJ_ROLE * obj +
        SCOPE_ROLE * scope, then bundled. The result is unique to the
        triple but compositional: changing only the object yields a
        vector still ~50% similar to the original (shared verb+scope).
        """
        return bundle([
            bind(self.role("verb"), self.atom(f"verb:{verb}")),
            bind(self.role("obj"), self.atom(f"obj:{obj or '_'}")),
            bind(self.role("scope"), self.atom(f"scope:{scope or '_'}")),
        ])

    def encode_action(self, action_name: str, kwargs: tuple[tuple[str, object], ...]) -> np.ndarray:
        """Encode a primitive action (name + key=value args) as one hypervector."""
        parts = [bind(self.role("prim"), self.atom(f"prim:{action_name}"))]
        for k, v in kwargs:
            key_hv = self.atom(f"argk:{k}")
            val_hv = self.atom(f"argv:{str(v)[:64]}")
            parts.append(bind(key_hv, val_hv))
        return bundle(parts)

    def encode_outcome(self, ok: bool, reward: float) -> np.ndarray:
        """Quantize reward into 5 bins, encode with success/fail flag."""
        bin_idx = max(0, min(4, int(round(reward * 4))))
        return bundle([
            bind(self.role("status"), self.atom("ok" if ok else "fail")),
            bind(self.role("reward"), self.atom(f"r{bin_idx}")),
        ])
