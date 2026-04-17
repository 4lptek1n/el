"""FrozenSubstrate — read-only thermofield used as a deterministic
feature extractor for downstream supervised learning.

After the substrate has been written into (Hebbian/covariance imprinting
of representative patterns), C and B can be FROZEN. From that point the
substrate is a fixed nonlinear dynamical system: any input (a set of
clamped/injected cells) is mapped to a stationary temperature pattern
after `relax_steps` of diffusion. We then read the temperature at a
fixed set of "electrode" positions to produce a dense float feature
vector.

Key design points for terabyte-scale streaming:

  * The substrate's parameter footprint is fixed: O(grid²) for C and B,
    independent of corpus size. A 192×192 grid uses ~600 KB total.
  * The feature vector size is controlled separately by `n_readout`
    (default 512) so the readout matrix stays small no matter how big
    the substrate is.
  * `encode()` is stateless w.r.t. learning — it never modifies C, B, or
    the stored pattern list. Calling it a million times in a row leaves
    the substrate identical.
  * No backprop, no torch dependency.

Intended use:

    from el.thermofield.pattern_memory import PatternMemory
    from el.thermofield.frozen import FrozenSubstrate

    pm = PatternMemory(cfg=...)
    for p in seed_patterns:
        pm.store(p)              # Hebbian imprint phase
    frozen = FrozenSubstrate.from_pattern_memory(pm, n_readout=512, seed=0)
    feat = frozen.encode(cue)    # numpy float32 vector of length n_readout
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .field import Field, FieldConfig

Pattern = list[tuple[int, int]]


@dataclass
class FrozenSubstrate:
    """A frozen thermofield + fixed read electrodes.

    The substrate state (C, B, FieldConfig) is treated as immutable.
    `encode()` runs the field forward only — no Hebbian update, no
    pattern list bookkeeping.
    """

    cfg: FieldConfig
    C_right: np.ndarray
    C_down: np.ndarray
    B_right: np.ndarray
    B_down: np.ndarray
    read_positions: np.ndarray  # int32, shape (n_readout, 2)
    relax_steps: int = 12
    seed: int = 0

    def __post_init__(self) -> None:
        # Defensive: lock the arrays so accidental writes raise.
        for arr in (self.C_right, self.C_down, self.B_right, self.B_down,
                    self.read_positions):
            arr.setflags(write=False)
        self._read_idx = (self.read_positions[:, 0].astype(np.int64) * self.cfg.cols
                          + self.read_positions[:, 1].astype(np.int64))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_pattern_memory(
        cls,
        pm,
        n_readout: int = 512,
        relax_steps: int = 12,
        seed: int = 0,
    ) -> "FrozenSubstrate":
        f = pm.field
        rng = np.random.default_rng(seed)
        n_cells = f.cfg.rows * f.cfg.cols
        n_readout = min(n_readout, n_cells)
        idx = rng.choice(n_cells, size=n_readout, replace=False)
        positions = np.stack([idx // f.cfg.cols, idx % f.cfg.cols], axis=1).astype(np.int32)
        return cls(
            cfg=f.cfg,
            C_right=f.C_right.copy(),
            C_down=f.C_down.copy(),
            B_right=f.B_right.copy(),
            B_down=f.B_down.copy(),
            read_positions=positions,
            relax_steps=relax_steps,
            seed=seed,
        )

    @classmethod
    def load(cls, path) -> "FrozenSubstrate":
        d = np.load(str(Path(path)))
        cfg = FieldConfig(
            rows=int(d["cfg_rows"]), cols=int(d["cfg_cols"]),
            diffusion_rate=float(d["cfg_diffusion_rate"]),
            decay=float(d["cfg_decay"]),
            n_steps=int(d["cfg_n_steps"]),
            nonlinear_alpha=float(d["cfg_nonlinear_alpha"]),
        )
        return cls(
            cfg=cfg,
            C_right=d["C_right"].copy(),
            C_down=d["C_down"].copy(),
            B_right=d["B_right"].copy(),
            B_down=d["B_down"].copy(),
            read_positions=d["read_positions"].copy(),
            relax_steps=int(d["relax_steps"]),
            seed=int(d["seed"]),
        )

    def save(self, path) -> None:
        np.savez(
            str(Path(path)),
            cfg_rows=self.cfg.rows, cfg_cols=self.cfg.cols,
            cfg_diffusion_rate=self.cfg.diffusion_rate,
            cfg_decay=self.cfg.decay, cfg_n_steps=self.cfg.n_steps,
            cfg_nonlinear_alpha=self.cfg.nonlinear_alpha,
            C_right=self.C_right, C_down=self.C_down,
            B_right=self.B_right, B_down=self.B_down,
            read_positions=self.read_positions,
            relax_steps=self.relax_steps, seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Forward (frozen) pass
    # ------------------------------------------------------------------
    def _build_field(self) -> Field:
        f = Field(self.cfg, seed=self.seed)
        # Overwrite the random init with the frozen weights. We assign
        # to .data via a fresh writable copy because the frozen arrays
        # are read-only.
        f.C_right = self.C_right.copy()
        f.C_down = self.C_down.copy()
        f.B_right = self.B_right.copy()
        f.B_down = self.B_down.copy()
        return f

    def encode(self, cue: Pattern, *, clamp: bool = False) -> np.ndarray:
        """Inject `cue`, relax `relax_steps`, return temperature at read electrodes.

        Returns a float32 vector of length `n_readout`.
        """
        f = self._build_field()
        f.reset_temp()
        f.inject(list(cue), [1.0] * len(cue), clamp=clamp)
        for _ in range(self.relax_steps):
            f.step()
        flat = f.T.ravel()
        return flat[self._read_idx].astype(np.float32, copy=True)

    def encode_batch(self, cues: Iterable[Pattern], *, clamp: bool = False
                     ) -> np.ndarray:
        """Encode many cues. Returns (n_cues, n_readout) float32 array."""
        out = []
        for cue in cues:
            out.append(self.encode(cue, clamp=clamp))
        return np.stack(out, axis=0) if out else np.zeros(
            (0, len(self.read_positions)), dtype=np.float32)

    @property
    def n_readout(self) -> int:
        return int(self.read_positions.shape[0])

    @property
    def n_substrate_cells(self) -> int:
        return int(self.cfg.rows * self.cfg.cols)

    def fingerprint(self) -> str:
        """Hash of frozen weights — for sanity-checking that the substrate
        truly hasn't been modified after a long downstream training run.
        """
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        for arr in (self.C_right, self.C_down, self.B_right, self.B_down,
                    self.read_positions):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()
