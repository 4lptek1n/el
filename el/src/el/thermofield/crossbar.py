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

        # K outgoing edges per cell to random other cells.
        # We use a dynamic-array layout: arrays may have spare CAPACITY
        # past `_n` so add_edge is amortised O(1) instead of O(n) per
        # call. All public APIs slice through `[: self._n]`.
        n0 = n_cells * k
        capacity = max(n0, 16)
        self._n = n0
        self._cap = capacity

        src_full = np.zeros(capacity, dtype=np.int64)
        dst_full = np.zeros(capacity, dtype=np.int64)
        C_full = np.zeros(capacity, dtype=np.float32)
        B_full = np.zeros(capacity, dtype=np.float32)

        src_full[:n0] = np.repeat(np.arange(n_cells, dtype=np.int64), k)
        dst_init = rng.integers(0, n_cells, size=n0).astype(np.int64)
        # Eliminate self-loops by shifting them to a neighbour
        self_loops = dst_init == src_full[:n0]
        dst_init[self_loops] = (dst_init[self_loops] + 1) % n_cells
        dst_full[:n0] = dst_init
        C_full[:n0] = rng.uniform(c_init_lo, c_init_hi, n0).astype(np.float32)
        # B stays zero by default

        self._src_buf = src_full
        self._dst_buf = dst_full
        self._C_buf = C_full
        self._B_buf = B_full

    # ------------------------------------------------------------------
    # Views (these are real numpy slices — writes go through to buffers)
    # ------------------------------------------------------------------
    @property
    def src(self) -> np.ndarray:
        return self._src_buf[: self._n]

    @src.setter
    def src(self, value: np.ndarray) -> None:
        v = np.asarray(value, dtype=np.int64)
        m = v.shape[0]
        if m != self._n:
            # Length-changing replacement — must also be done for dst,
            # C, B in matching length, OR call replace_edges() instead.
            # Resize all four buffers to keep them coherent.
            self._src_buf = v.copy()
            # Pad/truncate the others to keep buffer lengths in sync.
            self._dst_buf = self._fit_buffer(self._dst_buf, m, np.int64)
            self._C_buf = self._fit_buffer(self._C_buf, m, np.float32)
            self._B_buf = self._fit_buffer(self._B_buf, m, np.float32)
            self._n = m
            self._cap = m
        else:
            self._src_buf[: self._n] = v

    @property
    def dst(self) -> np.ndarray:
        return self._dst_buf[: self._n]

    @dst.setter
    def dst(self, value: np.ndarray) -> None:
        v = np.asarray(value, dtype=np.int64)
        m = v.shape[0]
        if m != self._n:
            self._dst_buf = v.copy()
            self._src_buf = self._fit_buffer(self._src_buf, m, np.int64)
            self._C_buf = self._fit_buffer(self._C_buf, m, np.float32)
            self._B_buf = self._fit_buffer(self._B_buf, m, np.float32)
            self._n = m
            self._cap = m
        else:
            self._dst_buf[: self._n] = v

    @property
    def C(self) -> np.ndarray:
        return self._C_buf[: self._n]

    @C.setter
    def C(self, value: np.ndarray) -> None:
        v = np.asarray(value, dtype=np.float32)
        if v.shape[0] != self._n:
            raise ValueError(
                f"C length {v.shape[0]} must equal current edge count {self._n}; "
                f"use replace_edges(src, dst, C, B) for length changes")
        self._C_buf[: self._n] = v

    @property
    def B(self) -> np.ndarray:
        return self._B_buf[: self._n]

    @B.setter
    def B(self, value: np.ndarray) -> None:
        v = np.asarray(value, dtype=np.float32)
        if v.shape[0] != self._n:
            raise ValueError(
                f"B length {v.shape[0]} must equal current edge count {self._n}; "
                f"use replace_edges(src, dst, C, B) for length changes")
        self._B_buf[: self._n] = v

    @staticmethod
    def _fit_buffer(buf: np.ndarray, m: int, dtype) -> np.ndarray:
        """Return a length-m buffer holding the first min(len(buf), m)
        entries of `buf`, padded with zeros. Internal helper kept simple
        because the only callers are the src/dst length-changing setters
        for legacy test ergonomics."""
        out = np.zeros(m, dtype=dtype)
        n = min(buf.shape[0], m)
        out[:n] = buf[:n]
        return out

    def replace_edges(self, src, dst, C=None, B=None) -> None:
        """Atomic length-changing replacement of all four edge arrays
        with strict length-matching. Preferred path over four separate
        setters when the new edge set differs in size."""
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        m = src.shape[0]
        if dst.shape[0] != m:
            raise ValueError(
                f"src and dst must be the same length: {src.shape[0]} vs {dst.shape[0]}")
        C_arr = (np.asarray(C, dtype=np.float32) if C is not None
                 else np.full(m, 0.5, dtype=np.float32))
        B_arr = (np.asarray(B, dtype=np.float32) if B is not None
                 else np.zeros(m, dtype=np.float32))
        if C_arr.shape[0] != m or B_arr.shape[0] != m:
            raise ValueError(
                f"C ({C_arr.shape[0]}) and B ({B_arr.shape[0]}) must match src/dst length {m}")
        self._src_buf = src.copy()
        self._dst_buf = dst.copy()
        self._C_buf = C_arr.copy()
        self._B_buf = B_arr.copy()
        self._n = m
        self._cap = m

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def n_edges(self) -> int:
        return int(self._n)

    def _grow(self, needed: int) -> None:
        """Double capacity (or more) until at least `needed` edges fit."""
        new_cap = self._cap
        while new_cap < needed:
            new_cap *= 2
        if new_cap == self._cap:
            return
        def _resize(buf, dtype):
            new = np.zeros(new_cap, dtype=dtype)
            new[: self._n] = buf[: self._n]
            return new
        self._src_buf = _resize(self._src_buf, np.int64)
        self._dst_buf = _resize(self._dst_buf, np.int64)
        self._C_buf = _resize(self._C_buf, np.float32)
        self._B_buf = _resize(self._B_buf, np.float32)
        self._cap = new_cap

    def add_edge(self, src: int, dst: int, c: float = 0.5, b: float = 0.0) -> None:
        """Append a single hand-specified edge (amortised O(1))."""
        if self._n + 1 > self._cap:
            self._grow(self._n + 1)
        i = self._n
        self._src_buf[i] = src
        self._dst_buf[i] = dst
        self._C_buf[i] = c
        self._B_buf[i] = b
        self._n += 1

    def add_edges(self, src_arr, dst_arr, c_arr=None, b_arr=None) -> None:
        """Bulk add — vectorised, much faster for large batches."""
        src_arr = np.asarray(src_arr, dtype=np.int64)
        dst_arr = np.asarray(dst_arr, dtype=np.int64)
        m = src_arr.shape[0]
        assert dst_arr.shape[0] == m
        if self._n + m > self._cap:
            self._grow(self._n + m)
        i, j = self._n, self._n + m
        self._src_buf[i:j] = src_arr
        self._dst_buf[i:j] = dst_arr
        self._C_buf[i:j] = (np.asarray(c_arr, dtype=np.float32)
                            if c_arr is not None else 0.5)
        self._B_buf[i:j] = (np.asarray(b_arr, dtype=np.float32)
                            if b_arr is not None else 0.0)
        self._n += m

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
        # Snapshot view-arrays once per call; they are O(1) slices.
        src = self.src
        dst = self.dst
        C = self.C
        B = self.B

        diff = T_flat[dst] - T_flat[src]              # >0 -> dst hotter

        c_fwd = np.maximum(C + B, 0.0)                # easier when src hot -> heat dst
        c_bwd = np.maximum(C - B, 0.0)                # easier when dst hot -> heat src
        c_eff = np.where(diff < 0, c_fwd, c_bwd)

        flux = self.flux_rate * c_eff * diff
        # diff > 0 -> dst hotter -> flux > 0 -> src gains, dst loses
        np.add.at(T_flat, src, flux)
        np.add.at(T_flat, dst, -flux)
        np.clip(T_flat, 0.0, 1.0, out=T_flat)


def crossbar_for_field(field, k: int = 8, seed: int = 0) -> SparseCrossbar:
    """Convenience: build a SparseCrossbar matching a Field's flat shape."""
    n_cells = int(field.cfg.rows * field.cfg.cols)
    return SparseCrossbar(n_cells=n_cells, k=k, seed=seed)
