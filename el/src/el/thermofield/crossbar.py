"""Sparse non-local crossbar — RRAM-style long-range connections.

The diffusion grid in `Field` is fundamentally local: heat travels at
roughly O(sqrt(t)) cells per step, so two distant points can never
"see" each other within any reasonable time horizon. Real neuromorphic
chips (Loihi, Tianjic, Darwin-3) and analog RRAM crossbars overcome
this by giving each cell a small set of arbitrary, possibly very
distant, partners — a sparse global router on top of the local grid.

This module implements exactly that as an overlay on top of an existing
Field (or any flat T array):

  - Each cell has K outgoing edges to K random other cells.
  - Each edge carries a symmetric conductance C and a directional
    bias B (same C+B scheme as the in-grid edges, so STDP rules can
    be reused later).
  - A `step(T_flat)` call applies one round of heat exchange across
    all edges, in place, using vectorised numpy ops.

The crossbar is direction-aware (B != 0 makes flow easier in one
direction than the other), so the same event-boundary STDP we use
in `sequence.py` will produce learned, asymmetric long-range
connections — the software analogue of a programmed RRAM crossbar.
"""
from __future__ import annotations

import numpy as np


class SparseCrossbar:
    """Sparse, learnable non-local edge set over `n_cells` cells.

    Edges are stored as flat arrays of length `n_cells * k`:
        src[i] -> dst[i]   conductance C[i] + bias B[i]
    """

    def __init__(
        self,
        n_cells: int,
        k: int = 8,
        seed: int = 0,
        c_init_lo: float = 0.30,
        c_init_hi: float = 0.60,
        flux_rate: float = 0.10,
    ):
        if k <= 0:
            raise ValueError("k must be > 0")
        rng = np.random.default_rng(seed)
        self.n_cells = int(n_cells)
        self.k = int(k)
        self.flux_rate = float(flux_rate)

        # K outgoing edges per cell to random other cells
        self.src = np.repeat(np.arange(n_cells, dtype=np.int64), k)
        dst = rng.integers(0, n_cells, size=n_cells * k).astype(np.int64)
        # Eliminate self-loops by shifting them to a neighbour
        self_loops = dst == self.src
        dst[self_loops] = (dst[self_loops] + 1) % n_cells
        self.dst = dst

        self.C = rng.uniform(c_init_lo, c_init_hi, n_cells * k).astype(np.float32)
        self.B = np.zeros(n_cells * k, dtype=np.float32)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def n_edges(self) -> int:
        return self.src.shape[0]

    def add_edge(self, src: int, dst: int, c: float = 0.5, b: float = 0.0) -> None:
        """Append a single hand-specified edge (useful for tests / demos)."""
        self.src = np.append(self.src, np.int64(src))
        self.dst = np.append(self.dst, np.int64(dst))
        self.C = np.append(self.C, np.float32(c))
        self.B = np.append(self.B, np.float32(b))

    def step(self, T_flat: np.ndarray) -> None:
        """One round of non-local heat exchange, in place on T_flat.

        For each edge (s, d):
          diff = T[d] - T[s]
          conductance is C+B in the "src->dst" direction (when src is
          hotter and heat flows forward, i.e. diff < 0) and C-B in the
          reverse direction. Both sides clipped at 0 so the bias cannot
          create negative resistance.
        """
        if T_flat.dtype != np.float32:
            T_flat = T_flat.astype(np.float32, copy=False)

        diff = T_flat[self.dst] - T_flat[self.src]   # >0 -> dst hotter

        c_fwd = np.maximum(self.C + self.B, 0.0)     # easier when src hot -> heat dst
        c_bwd = np.maximum(self.C - self.B, 0.0)     # easier when dst hot -> heat src
        c_eff = np.where(diff < 0, c_fwd, c_bwd)

        flux = self.flux_rate * c_eff * diff
        # diff > 0 -> dst hotter -> flux > 0 -> src gains, dst loses
        np.add.at(T_flat, self.src, flux)
        np.add.at(T_flat, self.dst, -flux)
        np.clip(T_flat, 0.0, 1.0, out=T_flat)


def crossbar_for_field(field, k: int = 8, seed: int = 0) -> SparseCrossbar:
    """Convenience: build a SparseCrossbar matching a Field's flat shape."""
    n_cells = int(field.cfg.rows * field.cfg.cols)
    return SparseCrossbar(n_cells=n_cells, k=k, seed=seed)
