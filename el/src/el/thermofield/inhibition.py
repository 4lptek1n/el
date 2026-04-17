"""Lateral inhibition and k-Winner-Take-All for the thermofield substrate.

The pattern-memory probe revealed that pure positive-only diffusion
plus Hebbian potentiation cannot form attractors: every Hebb write
strengthens edges everywhere it touches, and without a mechanism
that *suppresses* non-pattern cells, recall heat just smears.

Real cortex solves this with inhibitory interneurons that enforce
sparse coding — at any moment only a small fraction of pyramidal
cells are firing, the rest are actively pushed down. We approximate
this with two cheap operators that can be composed with `field.step()`:

  - `kwta_step(T_flat, k, suppression)`:
        keep the k hottest cells, multiply all others by `suppression`
        (default 0.5 → halve them every tick). This produces sparse
        survivors and is the substrate analogue of competitive WTA.

  - `global_gain_step(T_flat, target_mean)`:
        soft homeostasis: subtract a fraction of (mean - target) from
        every cell so total activity is bounded but no cell is
        annihilated. Useful as a gentler companion to k-WTA.

Both operate in place on a flat array — works for any Field by
passing `field.T.reshape(-1)`.
"""
from __future__ import annotations

import numpy as np


def kwta_step(T_flat: np.ndarray, k: int, suppression: float = 0.5) -> None:
    """Exact-k Winner-Take-All in place: keep EXACTLY k untouched
    (highest values, ties broken by index), suppress the remaining
    n-k cells.

    Args:
        T_flat: 1-D array of cell temperatures (modified in place).
        k: number of "winning" cells to spare.
        suppression: multiplier applied to non-winners (0.0 = hard kill,
            1.0 = no inhibition). Default 0.5 = halve.

    Implementation note: we use `argpartition` and a boolean index mask
    rather than a value boundary, so ties at the boundary value do NOT
    inflate the winner set. This was an architect-flagged correctness
    issue with the previous value-threshold version.
    """
    n = T_flat.shape[0]
    if k <= 0 or k >= n:
        return
    winner_idx = np.argpartition(T_flat, n - k)[n - k:]   # exactly k indices
    mask = np.ones(n, dtype=bool)
    mask[winner_idx] = False                              # losers = ~mask
    T_flat[mask] *= suppression


def global_gain_step(
    T_flat: np.ndarray,
    target_mean: float = 0.05,
    rate: float = 0.50,
) -> None:
    """Soft global gain control: nudge mean(T) toward `target_mean`.

    If activity is too high we subtract a uniform offset (clipped to
    keep T >= 0); if too low we let it grow naturally (no upward push).
    Bounded, biologically motivated, cheap.
    """
    m = float(T_flat.mean())
    if m <= target_mean:
        return
    excess = (m - target_mean) * rate
    T_flat -= excess
    np.clip(T_flat, 0.0, 1.0, out=T_flat)
