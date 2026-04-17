"""SkipBank — sparse long-range temporal edges over a 2D Field.

Graduated from `scripts/bench/seq_chain_v7_hybrid.py` into core. This is
the breakthrough that lets the substrate learn N>1 sequence chains: a
sparse bank of K random long-range edges per cell, learned via trace-
modulated Hebbian updates, providing nonlocal heat propagation in
addition to the standard local diffusion.

Storage: COO (src_flat, dst_flat, weight) of length M = K * R * C.
Density typically 1-3% of all possible cell pairs.

Save/load: NPZ format with src/dst/w arrays + topology config.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class SkipBankConfig:
    rows: int
    cols: int
    K: int = 4
    md_min: int = 3
    w_clip: float = 0.5


class SkipBank:
    """Sparse temporal skip-edge bank over a 2D grid.

    Each cell has K outgoing edges to randomly chosen cells at
    Manhattan distance >= md_min, sampled at construction time.
    Weights start at 0 and grow under trace-modulated Hebbian rule.
    """

    def __init__(self, rows: int, cols: int, *, K: int = 4,
                 md_min: int = 3, w_clip: float = 0.5, seed: int = 0):
        self.cfg = SkipBankConfig(rows=rows, cols=cols, K=K,
                                  md_min=md_min, w_clip=w_clip)
        self.R, self.C, self.K = rows, cols, K
        self.w_clip = w_clip
        self.seed = seed
        rng = np.random.default_rng(seed)
        rs, cs = np.indices((rows, cols))
        rs_flat = rs.reshape(-1); cs_flat = cs.reshape(-1)
        N = rows * cols
        src_list = []; dst_list = []
        for src_idx in range(N):
            r0, c0 = rs_flat[src_idx], cs_flat[src_idx]
            md = np.abs(rs_flat - r0) + np.abs(cs_flat - c0)
            cands = np.where(md >= md_min)[0]
            if len(cands) < K:
                continue
            picked = rng.choice(cands, size=K, replace=False)
            src_list.extend([src_idx] * K)
            dst_list.extend(picked.tolist())
        self.src = np.asarray(src_list, dtype=np.int32)
        self.dst = np.asarray(dst_list, dtype=np.int32)
        self.w = np.zeros(len(self.src), dtype=np.float32)

    @property
    def n_edges(self) -> int:
        return len(self.w)

    def density(self) -> float:
        N = self.R * self.C
        return self.n_edges / (N * (N - 1)) if N > 1 else 0.0

    def propagate(self, T_flat: np.ndarray, *, eta: float = 0.05) -> np.ndarray:
        """Skip-mediated injection contribution: dst += eta * w * src_T."""
        contrib = np.zeros_like(T_flat)
        np.add.at(contrib, self.dst, eta * self.w * T_flat[self.src])
        return contrib

    def stdp_update(self, T_flat: np.ndarray, E_long_flat: np.ndarray,
                    *, lr: float) -> None:
        """w[i->j] += lr · T[j] · E_long[i] (post-pre Hebbian, clipped)."""
        delta = lr * T_flat[self.dst] * E_long_flat[self.src]
        self.w += delta
        np.clip(self.w, -self.w_clip, self.w_clip, out=self.w)

    def reset_weights(self) -> None:
        self.w[:] = 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path) -> None:
        np.savez(str(Path(path)),
                 rows=self.R, cols=self.C, K=self.K,
                 md_min=self.cfg.md_min, w_clip=self.w_clip, seed=self.seed,
                 src=self.src, dst=self.dst, w=self.w)

    @classmethod
    def load(cls, path) -> "SkipBank":
        d = np.load(str(Path(path)))
        bank = cls.__new__(cls)
        bank.R = int(d["rows"]); bank.C = int(d["cols"]); bank.K = int(d["K"])
        bank.w_clip = float(d["w_clip"])
        bank.seed = int(d["seed"])
        bank.cfg = SkipBankConfig(rows=bank.R, cols=bank.C, K=bank.K,
                                  md_min=int(d["md_min"]),
                                  w_clip=bank.w_clip)
        bank.src = d["src"].astype(np.int32)
        bank.dst = d["dst"].astype(np.int32)
        bank.w = d["w"].astype(np.float32)
        return bank
