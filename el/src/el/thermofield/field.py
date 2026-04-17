"""Thermodynamic field: 2D grid of cells with temperature and conductivity.

Each cell has a scalar temperature T in [0, 1]. Adjacent cells share a
conductivity C in [0, 1]. Heat flows along the conductivity gradient.

The field is intentionally physics-shaped: diffusion is the only dynamic.
Nonlinearity comes from temperature-dependent conductivity (hot regions
conduct better — like a metal heating up) which makes the system capable
of nontrivial input/output mappings such as XOR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FieldConfig:
    rows: int = 7
    cols: int = 7
    diffusion_rate: float = 0.25
    decay: float = 0.002
    n_steps: int = 80
    nonlinear_alpha: float = 2.5  # state-dependent conductivity gain


class Field:
    """A 2D thermodynamic substrate.

    Conductivity arrays:
      C_right[i, j]  conductivity between cell (i, j) and (i, j+1)
      C_down[i, j]   conductivity between cell (i, j) and (i+1, j)
    """

    def __init__(self, cfg: FieldConfig | None = None, seed: int = 0):
        self.cfg = cfg or FieldConfig()
        rng = np.random.default_rng(seed)
        self.T = np.zeros((self.cfg.rows, self.cfg.cols), dtype=np.float32)
        self.C_right = rng.uniform(0.3, 0.6, (self.cfg.rows, self.cfg.cols - 1)).astype(np.float32)
        self.C_down = rng.uniform(0.3, 0.6, (self.cfg.rows - 1, self.cfg.cols)).astype(np.float32)

    def reset_temp(self) -> None:
        self.T[:] = 0.0
        self._clamp_positions: list[tuple[int, int]] = []
        self._clamp_values: list[float] = []

    def inject(self, positions, values, *, clamp: bool = True) -> None:
        """Set cell temperatures.

        When clamp=True (default), the cells are held at their value across
        every diffusion step (Dirichlet boundary). This makes inputs behave
        like constant heat sources rather than one-shot pulses that drain
        away. Pass clamp=False for a single-shot impulse.
        """
        if not hasattr(self, "_clamp_positions"):
            self._clamp_positions = []
            self._clamp_values = []
        for (r, c), v in zip(positions, values):
            v_clipped = float(np.clip(v, 0.0, 1.0))
            self.T[r, c] = v_clipped
            if clamp:
                self._clamp_positions.append((r, c))
                self._clamp_values.append(v_clipped)

    def _apply_clamps(self) -> None:
        for (r, c), v in zip(self._clamp_positions, self._clamp_values):
            self.T[r, c] = v

    def step(self) -> None:
        T = self.T
        a = self.cfg.nonlinear_alpha
        # State-dependent conductivity: hotter pair => more flow (mild nonlinearity)
        avg_h = 0.5 * (T[:, :-1] + T[:, 1:])
        avg_v = 0.5 * (T[:-1, :] + T[1:, :])
        c_h = self.C_right * (1.0 + a * avg_h)
        c_v = self.C_down * (1.0 + a * avg_v)

        flux_h = c_h * (T[:, 1:] - T[:, :-1])
        flux_v = c_v * (T[1:, :] - T[:-1, :])

        new_T = T.copy()
        new_T[:, :-1] += self.cfg.diffusion_rate * flux_h
        new_T[:, 1:] -= self.cfg.diffusion_rate * flux_h
        new_T[:-1, :] += self.cfg.diffusion_rate * flux_v
        new_T[1:, :] -= self.cfg.diffusion_rate * flux_v
        new_T *= (1.0 - self.cfg.decay)
        np.clip(new_T, 0.0, 1.0, out=new_T)
        self.T = new_T
        if hasattr(self, "_clamp_positions"):
            self._apply_clamps()

    def relax(self, n_steps: int | None = None) -> None:
        n = n_steps if n_steps is not None else self.cfg.n_steps
        for _ in range(n):
            self.step()

    def read(self, positions) -> np.ndarray:
        return np.array([self.T[r, c] for r, c in positions], dtype=np.float32)

    def total_energy(self) -> float:
        return float(self.T.sum())

    def stats(self) -> dict:
        return {
            "rows": self.cfg.rows,
            "cols": self.cfg.cols,
            "n_cells": int(self.cfg.rows * self.cfg.cols),
            "total_energy": self.total_energy(),
            "C_right_mean": float(self.C_right.mean()),
            "C_down_mean": float(self.C_down.mean()),
            "C_right_std": float(self.C_right.std()),
            "C_down_std": float(self.C_down.std()),
        }
