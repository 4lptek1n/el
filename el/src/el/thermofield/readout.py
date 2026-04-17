"""Streaming linear readout for the FrozenSubstrate.

Designed for terabyte-scale corpora: never holds the full feature
matrix in memory. Instead it accumulates the two sufficient statistics
of ridge regression on the fly:

    A = sum_i  x_i x_iᵀ   (D × D)
    B = sum_i  x_i y_iᵀ   (D × K)   for one-hot y_i

After the stream is finished, the closed-form solution is

    W = (A + λI)^{-1} B          shape (D, K)

and inference is a single matmul `feat @ W`.

Memory cost: O(D² + D·K), independent of corpus size N. For D=512
features and K=1000 classes this is ~2 MB — trivially TB-scalable.

Why ridge (and not gradient descent)? Closed-form, deterministic,
seed-stable, and an exact incremental update is available. Equivalent
to fully-converged GD on the squared-error loss with L2 regularization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class StreamingRidge:
    """Online accumulator for ridge regression sufficient statistics.

    Multi-class via one-vs-rest squared-error encoding (one-hot targets).
    Predict by argmax over the K column scores. This is the canonical
    "least-squares classifier" — fast, convex, and surprisingly strong
    when features are non-trivial (which is exactly what a relaxed
    thermofield gives us).
    """

    n_features: int
    n_classes: int
    ridge_lambda: float = 1.0
    _A: np.ndarray = field(init=False, repr=False)
    _B: np.ndarray = field(init=False, repr=False)
    _n_seen: int = field(init=False, default=0)
    _W: np.ndarray | None = field(init=False, default=None, repr=False)
    _bias: np.ndarray | None = field(init=False, default=None, repr=False)
    _feat_mean: np.ndarray = field(init=False, repr=False)
    _feat_M2: np.ndarray = field(init=False, repr=False)  # Welford running variance
    _class_count: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        D, K = self.n_features, self.n_classes
        self._A = np.zeros((D, D), dtype=np.float64)
        self._B = np.zeros((D, K), dtype=np.float64)
        self._feat_mean = np.zeros(D, dtype=np.float64)
        self._feat_M2 = np.zeros(D, dtype=np.float64)
        self._class_count = np.zeros(K, dtype=np.int64)

    # ------------------------------------------------------------------
    # Streaming update
    # ------------------------------------------------------------------
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Accumulate sufficient statistics from a mini-batch.

        X: (n, D) float
        y: (n,)   int class labels in [0, n_classes)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.shape[1] != self.n_features:
            raise ValueError(f"feature dim mismatch: got {X.shape[1]}, "
                             f"expected {self.n_features}")
        # update running mean / variance (Welford, vectorized batch)
        for x in X:
            self._n_seen += 1
            delta = x - self._feat_mean
            self._feat_mean += delta / self._n_seen
            self._feat_M2 += delta * (x - self._feat_mean)
        # one-hot Y
        Y = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)
        Y[np.arange(X.shape[0]), y] = 1.0
        # accumulate
        self._A += X.T @ X
        self._B += X.T @ Y
        # per-class counts (for intercept solve)
        for k in range(self.n_classes):
            self._class_count[k] += int((y == k).sum())
        # invalidate solved weights
        self._W = None

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self) -> None:
        """Solve W = (A + λI)^{-1} B once, cache it.

        Uses Cholesky for stability — A + λI is SPD by construction.
        """
        D = self.n_features
        reg = self._A + self.ridge_lambda * np.eye(D, dtype=np.float64)
        try:
            L = np.linalg.cholesky(reg)
            z = np.linalg.solve(L, self._B)
            W = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            # Numerical fallback: pseudo-inverse
            W = np.linalg.pinv(reg) @ self._B
        self._W = W.astype(np.float32, copy=False)
        # Per-class intercept solved analytically:
        #   pred_k(x) = (x - x̄)·W_k + p_k ,  where p_k = N_k / N
        # so equivalently  pred_k(x) = x·W_k + b_k  with  b_k = p_k - x̄·W_k.
        N = max(self._n_seen, 1)
        class_freq = self._class_count.astype(np.float64) / N
        self._bias = (class_freq - self._feat_mean @ W).astype(np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._W is None:
            self.solve()
        scores = np.asarray(X, dtype=np.float32) @ self._W  # (n, K)
        scores = scores + self._bias  # broadcast
        return scores.argmax(axis=1)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self._W is None:
            self.solve()
        return np.asarray(X, dtype=np.float32) @ self._W + self._bias

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path) -> None:
        if self._W is None:
            self.solve()
        np.savez(
            str(Path(path)),
            n_features=self.n_features, n_classes=self.n_classes,
            ridge_lambda=self.ridge_lambda,
            A=self._A, B=self._B, n_seen=self._n_seen,
            W=self._W, bias=self._bias,
            feat_mean=self._feat_mean, feat_M2=self._feat_M2,
            class_count=self._class_count,
        )

    @classmethod
    def load(cls, path) -> "StreamingRidge":
        d = np.load(str(Path(path)))
        r = cls(
            n_features=int(d["n_features"]),
            n_classes=int(d["n_classes"]),
            ridge_lambda=float(d["ridge_lambda"]),
        )
        r._A = d["A"].copy()
        r._B = d["B"].copy()
        r._n_seen = int(d["n_seen"])
        r._W = d["W"].copy() if d["W"].size else None
        r._bias = d["bias"].copy() if d["bias"].size else None
        r._feat_mean = d["feat_mean"].copy()
        r._feat_M2 = d["feat_M2"].copy()
        if "class_count" in d.files:
            r._class_count = d["class_count"].copy()
        return r

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def n_seen(self) -> int:
        return self._n_seen

    def feature_stats(self) -> dict:
        var = self._feat_M2 / max(self._n_seen - 1, 1)
        return {
            "mean_abs_mean": float(np.abs(self._feat_mean).mean()),
            "mean_std":      float(np.sqrt(var).mean()),
            "active_frac":   float((self._feat_mean > 1e-4).mean()),
        }
