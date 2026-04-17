"""Eşik 3 — seq_then_pat fix attempt: B-protection during pattern write.

Hypothesis: pattern Hebb on C indirectly washes the directional B-bias
because the resulting saturated C floods diffusion. If we SNAPSHOT the
top-k strongest B values BEFORE pat-storage and RESTORE them after each
write step, the sequence trace survives.

Sweep over protect_frac ∈ {0.0, 0.05, 0.1, 0.2, 0.4} where the top
fraction of |B| edges is preserved verbatim.
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
from thermofield.test_multitask import (  # type: ignore
    _seq_train, _seq_probe, _make_mem, _pattern_acc,
    GRID, N_PAT, SEEDS, N_TRIALS, COTRAIN_PARAMS,
)
from el.thermofield import FieldConfig
from el.thermofield.pattern_memory import random_pattern


def _store_with_b_protect(mem, pattern, protect_frac):
    """Same as PatternMemory.store but snapshots top-|B| edges and
    restores them after each Hebb step."""
    f = mem.field
    if protect_frac > 0:
        br = f.B_right; bd = f.B_down
        kr = max(1, int(br.size * protect_frac))
        kd = max(1, int(bd.size * protect_frac))
        ridx = np.argpartition(np.abs(br).ravel(), -kr)[-kr:]
        didx = np.argpartition(np.abs(bd).ravel(), -kd)[-kd:]
        rsnap = np.array(br.flat[ridx], copy=True)
        dsnap = np.array(bd.flat[didx], copy=True)
    f.reset_temp()
    f.inject(list(pattern), [1.0]*len(pattern))
    from el.thermofield.plasticity import hebbian_update
    from el.thermofield.inhibition import kwta_step
    for _ in range(mem.write_steps):
        f.step()
        if mem.wta_k > 0:
            kwta_step(f.T.reshape(-1), mem.wta_k, mem.wta_suppression)
        hebbian_update(f, lr=mem.write_lr, decay=mem.write_decay)
        if protect_frac > 0:
            f.B_right.flat[ridx] = rsnap
            f.B_down.flat[didx] = dsnap
    f.reset_temp()
    mem.patterns.append(list(pattern))


def run(protect_frac: float):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pa, sq = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mem, pat_size = _make_mem(seed, cfg, **COTRAIN_PARAMS)
        patterns = [random_pattern(GRID, GRID, k=pat_size, rng=rng)
                    for _ in range(N_PAT)]
        # seq first
        _seq_train(mem.field)
        # pat second, with B protection
        for p in patterns:
            _store_with_b_protect(mem, p, protect_frac)
        pa.append(_pattern_acc(mem, patterns, 0.5,
                  np.random.default_rng(seed*31+7), N_TRIALS))
        sq.append(_seq_probe(mem.field, seed))
    return np.array(pa), np.array(sq)


def baselines():
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pa, sq = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mem, pat_size = _make_mem(seed, cfg, **COTRAIN_PARAMS)
        patterns = [random_pattern(GRID, GRID, k=pat_size, rng=rng)
                    for _ in range(N_PAT)]
        for p in patterns: mem.store(p)
        pa.append(_pattern_acc(mem, patterns, 0.5,
                  np.random.default_rng(seed*31+7), N_TRIALS))
    pa_alone = float(np.mean(pa))
    sq2 = []
    for seed in SEEDS:
        from el.thermofield import Field
        f = Field(cfg, seed=seed)
        _seq_train(f)
        sq2.append(_seq_probe(f, seed))
    sq_alone = float(np.mean(sq2))
    return pa_alone, sq_alone


def main():
    pa0, sq0 = baselines()
    print(f"baselines: pat_only {pa0:.3f}   seq_only {sq0:+.4f}")
    print("-" * 70)
    print(f"{'protect_frac':>12s} | {'pat_acc':>10s} | {'seq_disc':>10s} | "
          f"{'pat_keep%':>10s} | {'seq_keep%':>10s}")
    print("-" * 70)
    for pf in (0.0, 0.05, 0.10, 0.20, 0.40, 0.80):
        pa, sq = run(pf)
        pam, sqm = float(np.mean(pa)), float(np.mean(sq))
        pk = 100*pam/max(pa0,1e-9); sk = 100*sqm/max(sq0,1e-9)
        flag = " <-- both ≥90 %" if (pk>=90 and sk>=90) else ""
        print(f"{pf:12.2f} | {pam:10.3f} | {sqm:+10.4f} | "
              f"{pk:9.1f}% | {sk:9.1f}%{flag}")


if __name__ == "__main__":
    main()
