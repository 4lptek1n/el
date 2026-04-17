"""Spike binarization layer — bridge from continuous T to SNN protocol.

Real spiking-neuromorphic chips (TrueNorth, Loihi 2, Tianjic, Darwin-3)
do not transmit analog values between cores; they transmit binary
spike events with addresses (AER protocol). To map our continuous
thermofield substrate onto such a chip, we need a deterministic
threshold-and-reset transducer:

  - When a cell's T crosses a fixed firing threshold θ from below,
    emit a single binary spike for that cell at this tick and apply a
    fixed reset (subtract `reset_drop` from T, clipped to 0).
  - During an absolute refractory window the cell cannot fire again,
    even if T re-crosses θ.

Spikes are returned as a 1-D array of flat indices, exactly the format
that real AER routers consume (just ints, no analog payload). The
substrate's continuous dynamics keep running underneath unchanged —
this layer is *only* the binarising transducer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpikeConfig:
    threshold: float = 0.40
    reset_drop: float = 0.50          # how much T drops on a spike
    refractory_steps: int = 2          # ticks during which cell cannot re-spike


class SpikeEncoder:
    """Stateful threshold-and-reset spike encoder over a flat T array."""

    def __init__(self, n_cells: int, cfg: SpikeConfig | None = None):
        self.cfg = cfg or SpikeConfig()
        self.n_cells = int(n_cells)
        self._prev_T = np.zeros(self.n_cells, dtype=np.float32)
        self._refractory_remaining = np.zeros(self.n_cells, dtype=np.int32)
        self.total_spikes = 0

    def reset(self) -> None:
        self._prev_T[:] = 0.0
        self._refractory_remaining[:] = 0
        self.total_spikes = 0

    def step(self, T_flat: np.ndarray) -> np.ndarray:
        """Read current T, emit spikes for newly-rising super-θ cells,
        apply reset and refractory bookkeeping. Modifies T_flat in place
        for the reset, returns flat indices of cells that fired this tick.
        """
        if T_flat.shape[0] != self.n_cells:
            raise ValueError(
                f"T_flat shape {T_flat.shape} != n_cells {self.n_cells}"
            )

        eligible = self._refractory_remaining == 0
        rising = (T_flat > self.cfg.threshold) & (
            self._prev_T <= self.cfg.threshold
        )
        spiked_mask = rising & eligible
        spiked_idx = np.nonzero(spiked_mask)[0]

        if spiked_idx.size > 0:
            T_flat[spiked_idx] -= self.cfg.reset_drop
            np.clip(T_flat, 0.0, 1.0, out=T_flat)
            self.total_spikes += int(spiked_idx.size)

        # Decay refractory counter for cells that did NOT just spike,
        # then load fresh refractory window for cells that DID spike.
        # This guarantees `refractory_steps` future ticks are blocked
        # (the previous version off-by-oned to refractory_steps - 1).
        not_spiked = ~spiked_mask
        self._refractory_remaining[not_spiked] = np.maximum(
            self._refractory_remaining[not_spiked] - 1, 0
        )
        if spiked_idx.size > 0:
            self._refractory_remaining[spiked_idx] = self.cfg.refractory_steps

        self._prev_T = T_flat.copy()
        return spiked_idx
