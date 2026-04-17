"""Inhibitory interneuron population for the thermofield.

Biological motivation: a single excitatory layer with shared paths cannot
solve XOR — strengthening (0,1)/(1,0) paths also lights up (1,1). Real
brains add a separate inhibitory population that fires only on coincident
input and subtracts from the output.

This module adds a small bank of interneurons that read from the field's
temperature grid and produce a scalar inhibition value subtracted from
the output reading. Each interneuron learns:

  w_in[k, r, c]  -- receptive field over the excitatory grid
  w_out[k]       -- gain on the output (how strongly to inhibit)
  theta[k]       -- activation threshold

Activation: I_k = ReLU( sum(w_in[k] * T) - theta[k] )
Inhibition contribution at the output: sum_k w_out[k] * I_k

Learning is local:
  * Hebbian on w_in: when interneuron fires AND target says output should
    be low, strengthen connections to currently-active cells.
  * Supervised on w_out: when target=low and net_output is too high,
    grow inhibition gain; when target=high and net_output is too low,
    shrink it.
  * Threshold homeostasis: theta drifts up if the interneuron fires too
    often, down if too rarely (stays selective).

No backprop, all rules use only locally available signals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import Field


@dataclass
class InterneuronConfig:
    n: int = 2                  # number of interneurons
    init_theta: float = 0.8
    target_fire_rate: float = 0.25
    homeo_rate: float = 0.005


class Interneurons:
    def __init__(
        self,
        cfg: InterneuronConfig,
        field_shape: tuple[int, int],
        seed: int = 0,
    ):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        rows, cols = field_shape
        # Small positive random initial receptive field
        self.w_in = rng.uniform(0.05, 0.15, (cfg.n, rows, cols)).astype(np.float32)
        self.w_out = np.zeros(cfg.n, dtype=np.float32)
        self.theta = np.full(cfg.n, cfg.init_theta, dtype=np.float32)
        self._fire_history: list[np.ndarray] = []

    def activate(self, T: np.ndarray) -> np.ndarray:
        """Return I_k for each interneuron given the field temperature grid."""
        drives = (self.w_in * T[None, :, :]).sum(axis=(1, 2))
        I = np.maximum(0.0, drives - self.theta)
        return I.astype(np.float32)

    def inhibition(self, T: np.ndarray) -> float:
        """Total inhibitory current applied to the output reading."""
        I = self.activate(T)
        return float((self.w_out * I).sum())

    def update(
        self,
        T: np.ndarray,
        target: float,
        net_output: float,
        *,
        lr_in: float = 0.02,
        lr_out: float = 0.10,
        decay_in: float = 0.001,
    ) -> None:
        I = self.activate(T)
        fired = (I > 0).astype(np.float32)
        self._fire_history.append(fired)
        if len(self._fire_history) > 200:
            self._fire_history.pop(0)

        target_low = target < 0.3
        target_high = target > 0.7
        err = net_output - target

        for k in range(self.cfg.n):
            if I[k] <= 0:
                # Decay receptive field slowly when never firing
                self.w_in[k] *= (1.0 - decay_in)
                continue
            if target_low and err > 0.1:
                # Output too high but should be low -- strengthen receptive
                # field toward currently-active cells, grow inhibitory gain.
                self.w_in[k] += lr_in * I[k] * T
                self.w_out[k] += lr_out * I[k] * err
            elif target_high and err < -0.1:
                # Output too low but should be high -- this interneuron is
                # over-suppressing. Shrink its outgoing gain.
                self.w_out[k] -= lr_out * I[k] * (-err)
            # Always lightly decay w_in to prevent runaway growth
            self.w_in[k] *= (1.0 - decay_in)

        # Clip
        np.clip(self.w_in, 0.0, 1.0, out=self.w_in)
        np.clip(self.w_out, 0.0, 5.0, out=self.w_out)

        # Threshold homeostasis: keep firing rate near target
        if len(self._fire_history) >= 20:
            recent = np.mean(self._fire_history[-50:], axis=0)
            for k in range(self.cfg.n):
                err_rate = recent[k] - self.cfg.target_fire_rate
                self.theta[k] += self.cfg.homeo_rate * err_rate
                self.theta[k] = float(np.clip(self.theta[k], 0.1, 5.0))

    def learn_receptive_field_from_pattern(
        self,
        T_pattern: np.ndarray,
        *,
        lr: float = 0.10,
        decay: float = 0.02,
    ) -> None:
        """Hebbian update on the *sharp* (pre-relax) input pattern.

        Critical insight: after diffusion relaxes the field, heat spreads
        across most of the grid and the difference between (1,1) and
        single-input cases becomes blurred. But at the *injection* moment
        the pattern is sharp — only the clamped input cells are non-zero.
        Running Hebbian on this sharp snapshot lets the interneuron
        discover input-cell-specific receptive fields without supervision.

        Each call: strengthen w_in toward currently-active cells, decay
        elsewhere, then L1-normalize so weights stay bounded and
        competitive (some cells must lose mass for others to gain).
        """
        for k in range(self.cfg.n):
            self.w_in[k] += lr * T_pattern
            self.w_in[k] *= (1.0 - decay)
            np.clip(self.w_in[k], 0.0, 1.0, out=self.w_in[k])
            # L1 normalize so total receptive-field mass is bounded.
            total = self.w_in[k].sum()
            if total > 1e-6:
                self.w_in[k] *= (1.5 / total)

    def calibrate_threshold_and_gain(
        self,
        positive_drives: list[float],
        coincidence_drives: list[float],
        *,
        margin: float = 0.05,
        gain: float = 5.0,
    ) -> None:
        """Set theta between max(positive) and min(coincidence) drive.

        After the receptive field is learned, set the firing threshold
        so the interneuron is silent on single-input cases but active on
        coincident cases. Sets w_out to a fixed gain (could itself be
        learned via a supervised rule, left for later).
        """
        if not coincidence_drives:
            return
        pos_max = max(positive_drives) if positive_drives else 0.0
        coinc_min = min(coincidence_drives)
        if coinc_min <= pos_max:
            # Receptive field is too coarse to separate cases; fall back
            # to a threshold that at least suppresses on coincidence.
            self.theta[:] = max(0.0, coinc_min - margin)
        else:
            self.theta[:] = (pos_max + coinc_min) / 2.0
        self.w_out[:] = gain

    def stats(self) -> dict:
        return {
            "n": self.cfg.n,
            "w_in_mean": float(self.w_in.mean()),
            "w_out": [float(x) for x in self.w_out],
            "theta": [float(x) for x in self.theta],
        }
