"""HDM-EL: Hyperdimensional + Active Inference world model.

A third memory complementing el's skill registry (epizodic) and Action
Transformer (statistical). The world model is *predictive*: given
(state, action) it predicts the outcome's hypervector, which the
Active Inference planner uses to score candidate plans by Expected
Free Energy.

Pure NumPy. No GPU. ~1 MB per ~1000 stored experiences.
"""
from __future__ import annotations

from .hdc import (
    DEFAULT_DIM,
    HDC,
    bind,
    bundle,
    cosine_sim,
    permute,
    random_hv,
)
from .planner import ActiveInferencePlanner, EFEScore
from .store import WorldModelStore
from .world import Experience, WorldModel

__all__ = [
    "ActiveInferencePlanner",
    "DEFAULT_DIM",
    "EFEScore",
    "Experience",
    "HDC",
    "WorldModel",
    "WorldModelStore",
    "bind",
    "bundle",
    "cosine_sim",
    "permute",
    "random_hv",
]
