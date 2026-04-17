"""Continual / class-incremental learning helpers on top of the
thermofield substrate.

Two reusable pieces:

  1. `feature_snapshot(field, pattern, snap_steps)` — frozen substrate
     used as a featurizer. Inject a binary pattern, relax, snapshot
     the field temperature at multiple time steps, return the
     concatenation as a dense feature vector. The substrate stays
     frozen across tasks (no plasticity during featurization).

  2. `PerClassReadout(dim, n_classes)` — per-class linear head trained
     1-vs-rest. Each class gets its own weight vector that is trained
     ONLY on samples of that class plus a negative pool, then frozen.
     Subsequent class introductions never touch it. This is what
     prevents the catastrophic forgetting that a shared softmax
     readout exhibits in class-incremental MNIST.

Empirical: on class-incremental MNIST (5 binary tasks, 10 classes
total), a frozen 28×28 substrate feature extractor + this readout
reaches 83 % final accuracy with NO replay buffer, vs 17 % for the
same features with a shared-softmax readout, vs 87 % for an MLP that
gets a full replay buffer. See `el/scripts/bench/continual_per_class_head.py`.
"""
from __future__ import annotations
from typing import Iterable, Sequence

import numpy as np

from .field import Field

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def feature_snapshot(
    field: Field,
    active_cells: Sequence[tuple[int, int]],
    snap_steps: tuple[int, ...] = (2, 5, 8),
) -> np.ndarray:
    """Inject `active_cells` (do NOT clamp), relax `max(snap_steps)`
    steps, return T at each requested step concatenated as a 1-D
    feature vector. The substrate is NOT modified (no plasticity).

    Returns a vector of size `field.cfg.rows * field.cfg.cols * len(snap_steps)`.
    """
    field.reset_temp()
    if not active_cells:
        return np.zeros(field.cfg.rows * field.cfg.cols * len(snap_steps),
                        dtype=np.float32)
    field.inject(list(active_cells), [1.0] * len(active_cells), clamp=False)
    out: list[np.ndarray] = []
    mx = max(snap_steps)
    for t in range(mx + 1):
        if t in snap_steps:
            out.append(field.T.flatten().copy())
        if t < mx:
            field.step()
    return np.concatenate(out)


class PerClassReadout:
    """Per-class 1-vs-rest linear head on top of frozen features.

    Each class is trained once, when first introduced. After that its
    weights are frozen — the catastrophic forgetting of a shared
    softmax readout cannot occur because the per-class weight vectors
    are *physically separate parameters*.

    Predictions are argmax over class scores `w_c · phi(x) + b_c`,
    restricted to classes that have been trained so far.
    """

    def __init__(self, dim: int, n_classes: int = 10):
        if not _HAS_TORCH:
            raise ImportError(
                "PerClassReadout requires torch. Install with: pip install torch")
        self.dim = dim
        self.n_classes = n_classes
        self.w = np.zeros((n_classes, dim), dtype=np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.trained = [False] * n_classes

    def train_class(
        self,
        cls: int,
        F_pos: np.ndarray,
        F_neg: np.ndarray,
        lr: float = 1e-2,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        """Logistic regression for one class. Other classes' weights
        are not touched."""
        if cls < 0 or cls >= self.n_classes:
            raise ValueError(f"cls {cls} out of range [0, {self.n_classes})")
        if F_pos.shape[1] != self.dim or F_neg.shape[1] != self.dim:
            raise ValueError(
                f"feature dim mismatch: expected {self.dim}, got "
                f"pos={F_pos.shape[1]}, neg={F_neg.shape[1]}")
        X = np.concatenate([F_pos, F_neg], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(len(F_pos)), np.zeros(len(F_neg))]).astype(np.float32)
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y)
        w = torch.zeros(self.dim, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            logits = Xt @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.w[cls] = w.detach().numpy()
        self.b[cls] = float(b.detach().numpy().item())
        self.trained[cls] = True

    def predict(self, F_input: np.ndarray) -> np.ndarray:
        """Argmax over trained classes only. Untrained classes get
        score -inf so they are never selected."""
        if F_input.shape[1] != self.dim:
            raise ValueError(
                f"feature dim mismatch: expected {self.dim}, got {F_input.shape[1]}")
        scores = F_input.astype(np.float32) @ self.w.T + self.b
        mask = np.array([0.0 if t else -1e9 for t in self.trained],
                        dtype=np.float32)
        return (scores + mask).argmax(axis=1)
