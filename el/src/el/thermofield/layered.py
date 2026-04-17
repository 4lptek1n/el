"""3D layered thermofield substrate — a stack of 2D Fields with vertical
conductances and coincidence-gated propagation.

Design:
- N layers, each is a standard `Field` (rows × cols).
- Between adjacent layers, every (r, c) cell has a vertical edge with
  symmetric conductance `C_v` and directional bias `B_v` (positive favors
  upward flow, negative favors downward) — exactly the same C+B scheme
  as horizontal/vertical edges within a layer.
- "If-then collision" = explicit coincidence gating: when a cell in layer
  l crosses the firing threshold θ from below (rising edge), it INJECTS
  a fixed packet of heat into the corresponding cell of layer l+1. This
  is on top of normal vertical diffusion. Two inputs that arrive at the
  same cell simultaneously will both push it through θ → produces a
  much larger upper-layer response than either alone (super-linear
  coincidence amplification).

Why three layers, why if-then:
- L0 = sensory substrate (raw inputs heat L0 cells)
- L1 = coincidence layer (fires only when multiple L0 patterns align)
- L2 = abstraction layer (fires only when multiple L1 patterns align)

This recreates the cortical column motif (sensors → coincidence →
invariant features) inside the same physical substrate, with no
backprop and no global control.
"""
from __future__ import annotations

import numpy as np

from .field import Field, FieldConfig


class LayeredField:
    """Stack of `n_layers` Fields with vertical C+B edges and coincidence
    gating between layers."""

    def __init__(
        self,
        n_layers: int = 3,
        cfg: FieldConfig | None = None,
        seed: int = 0,
        vertical_rate: float = 0.25,
        firing_threshold: float = 0.30,
        spike_amplitude: float = 0.20,
    ):
        cfg = cfg or FieldConfig()
        self.cfg = cfg
        self.n_layers = n_layers
        self.vertical_rate = vertical_rate
        self.firing_threshold = firing_threshold
        self.spike_amplitude = spike_amplitude

        rng = np.random.default_rng(seed)
        # Each layer gets its own Field with its own seed
        self.layers: list[Field] = [
            Field(cfg, seed=seed * 1000 + l) for l in range(n_layers)
        ]
        # Vertical conductance + bias between adjacent layers
        self.C_v: list[np.ndarray] = [
            rng.uniform(0.3, 0.6, (cfg.rows, cfg.cols)).astype(np.float32)
            for _ in range(n_layers - 1)
        ]
        self.B_v: list[np.ndarray] = [
            np.zeros((cfg.rows, cfg.cols), dtype=np.float32)
            for _ in range(n_layers - 1)
        ]
        # Track previous T for rising-edge detection
        self._prev_T: list[np.ndarray] = [layer.T.copy() for layer in self.layers]

    # ------------------------------------------------------------------
    # I/O on individual layers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        for layer in self.layers:
            layer.reset_temp()
        self._prev_T = [layer.T.copy() for layer in self.layers]

    def inject(self, layer: int, positions, values) -> None:
        self.layers[layer].inject(positions, values)

    def read(self, layer: int, positions) -> np.ndarray:
        return self.layers[layer].read(positions)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self) -> dict:
        """One global step. Returns a dict with diagnostic counts.

        Order of operations:
          1. Each layer takes its own intra-layer diffusion step.
          2. Vertical diffusion between adjacent layers (uses C_v + B_v).
          3. Coincidence gating: cells that just crossed θ inject a spike
             packet into the layer above.
        """
        # 1. Intra-layer dynamics
        for layer in self.layers:
            layer.step()

        # 2. Vertical (inter-layer) diffusion
        for l in range(self.n_layers - 1):
            T_lo = self.layers[l].T
            T_hi = self.layers[l + 1].T
            # Effective conductances: B_v positive = easier upward flow
            c_up = np.maximum(self.C_v[l] + self.B_v[l], 0.0)
            c_dn = np.maximum(self.C_v[l] - self.B_v[l], 0.0)
            diff = T_hi - T_lo  # >0: upper hotter, flow goes down (use c_dn)
            c_eff = np.where(diff < 0, c_up, c_dn)
            flux = self.vertical_rate * c_eff * diff
            # diff > 0: flux > 0, heat moves from upper (loses) to lower (gains)
            T_lo += flux
            T_hi -= flux
            np.clip(T_lo, 0.0, 1.0, out=T_lo)
            np.clip(T_hi, 0.0, 1.0, out=T_hi)

        # 3. Coincidence gating (if-then rising edge → spike up)
        spike_count = 0
        for l in range(self.n_layers - 1):
            cur = self.layers[l].T
            prev = self._prev_T[l]
            spiked = (cur > self.firing_threshold) & (prev <= self.firing_threshold)
            n = int(spiked.sum())
            spike_count += n
            if n > 0:
                self.layers[l + 1].T[spiked] += self.spike_amplitude
                np.clip(self.layers[l + 1].T, 0.0, 1.0, out=self.layers[l + 1].T)

        # 4. Re-apply clamps and update prev_T snapshot
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "_clamp_positions"):
                layer._apply_clamps()
            self._prev_T[i] = layer.T.copy()

        return {"spikes": spike_count}

    def relax(self, n_steps: int) -> int:
        total_spikes = 0
        for _ in range(n_steps):
            total_spikes += self.step()["spikes"]
        return total_spikes

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "rows": self.cfg.rows,
            "cols": self.cfg.cols,
            "energy_per_layer": [float(layer.T.sum()) for layer in self.layers],
            "C_v_mean": [float(c.mean()) for c in self.C_v],
            "B_v_absmean": [float(np.abs(b).mean()) for b in self.B_v],
        }
