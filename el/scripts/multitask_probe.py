"""Eşik 3 — multi-task substrate probe (CLI report wrapper).

Mirrors the protocol that test_multitask.py verifies in CI: same Field
hosts BOTH PatternMemory (writes C, symmetric Hebb) AND sequence STDP
(writes B, directional). With write_decay=0.005 + write_lr=0.07 +
write_steps=15, multi_pat_then_seq retains ≥90 % of both single-task
baselines simultaneously. Reverse ordering (seq_then_pat) needs the
elevated STDP lr=0.14 to recover.

Reports the four conditions and prints a Türkçe verdict.
"""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from el.thermofield.field import Field, FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern
from el.thermofield.sequence import (
    EligibilityTrace, present_event, relax_with_trace, stdp_hebbian_update,
)
from el.thermofield.plasticity import hebbian_update

GRID = 14
N_PAT = 8
A_POS = (12, 1)
B_POS = (12, 5)
SEEDS = list(range(8))
N_TRIALS = 15
COTRAIN_PARAMS = dict(write_lr=0.07, write_steps=15, write_decay=0.005)


def _noisy_cue(p, drop, rng, n_cells, cols):
    kn = max(1, int(round(len(p) * (1 - drop))))
    keep = [p[i] for i in sorted(rng.choice(len(p), kn, replace=False))]
    pset = set(p); ds = []
    while len(ds) < kn:
        idx = int(rng.integers(0, n_cells)); rc = (idx // cols, idx % cols)
        if rc not in pset and rc not in ds: ds.append(rc)
    return keep + ds


def _pattern_acc(mem, patterns, drop, rng, nt):
    cfg = mem.cfg; nc = cfg.rows * cfg.cols; correct = 0
    for _ in range(nt):
        i = int(rng.integers(0, len(patterns)))
        cue = _noisy_cue(patterns[i], drop, rng, nc, cfg.cols)
        b, _, _ = mem.recall(cue)
        if b == i: correct += 1
    return correct / nt


def _seq_train(field, lr=0.07, n_epochs=30, hold=5, gap=2):
    trace = EligibilityTrace((field.cfg.rows, field.cfg.cols), decay=0.80)
    for _ in range(n_epochs):
        field.reset_temp(); trace.reset()
        present_event(field, trace, [A_POS], [1.0], hold=hold)
        relax_with_trace(field, trace, gap)
        field._clamp_positions = []; field._clamp_values = []
        field.inject([B_POS], [1.0])
        stdp_hebbian_update(field, trace, lr=lr)
        for _ in range(hold):
            field.step(); trace.update(field.T)
        hebbian_update(field, lr=0.07, decay=0.001)
    field.reset_temp()


def _seq_probe(field, seed, hold=5, read_delay=6):
    field.reset_temp()
    tr = EligibilityTrace((field.cfg.rows, field.cfg.cols), decay=0.80)
    present_event(field, tr, [A_POS], [1.0], hold=hold)
    relax_with_trace(field, tr, read_delay)
    cue = float(field.T[B_POS])
    fresh = Field(field.cfg, seed=seed)
    ftr = EligibilityTrace((field.cfg.rows, field.cfg.cols), decay=0.80)
    present_event(fresh, ftr, [A_POS], [1.0], hold=hold)
    relax_with_trace(fresh, ftr, read_delay)
    return cue - float(fresh.T[B_POS])


def _make_mem(seed, cfg, **overrides):
    pat_size = max(4, int(0.05 * GRID * GRID))
    wta_k = max(pat_size + 2, int(0.075 * GRID * GRID))
    base = dict(write_steps=20, write_lr=0.30, write_decay=0.0)
    base.update(overrides)
    return PatternMemory(
        cfg=cfg, seed=seed,
        wta_k=wta_k, wta_suppression=0.3, rule="hebb", **base), pat_size


def _run(condition, mem_overrides, seq_lr=0.07):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pat_accs, discrims = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mem, pat_size = _make_mem(seed, cfg, **mem_overrides)
        patterns = [random_pattern(GRID, GRID, k=pat_size, rng=rng)
                    for _ in range(N_PAT)]
        field = mem.field
        if condition == "seq_only":
            _seq_train(field, lr=seq_lr)
        elif condition == "pattern_only":
            for p in patterns: mem.store(p)
        elif condition == "multi_pat_then_seq":
            for p in patterns: mem.store(p)
            _seq_train(field, lr=seq_lr)
        elif condition == "multi_seq_then_pat":
            _seq_train(field, lr=seq_lr)
            for p in patterns: mem.store(p)
        if mem.patterns:
            pat_accs.append(_pattern_acc(mem, patterns, 0.5,
                np.random.default_rng(seed*31+7), N_TRIALS))
        discrims.append(_seq_probe(field, seed))
    return np.array(pat_accs), np.array(discrims)


def main():
    print("=" * 78)
    print("EŞIK 3 — MULTI-TASK SUBSTRATE PROBE")
    print(f"  grid {GRID}×{GRID}, N_PAT={N_PAT}, seeds={len(SEEDS)} × {N_TRIALS} trials")
    print("  protocol: PatternMemory(C-Hebb) + sequence STDP(B) on SAME Field")
    print("=" * 78)

    pa_alone, _      = _run("pattern_only", COTRAIN_PARAMS)
    _,      sq_alone = _run("seq_only", COTRAIN_PARAMS)
    pa_pat_seq, sq_pat_seq = _run("multi_pat_then_seq", COTRAIN_PARAMS)
    pa_seq_pat, sq_seq_pat = _run("multi_seq_then_pat", COTRAIN_PARAMS)
    # known second WIN: seq_then_pat with elevated STDP lr
    pa_seq_pat14, sq_seq_pat14 = _run("multi_seq_then_pat", COTRAIN_PARAMS, seq_lr=0.14)

    def f(x): return f"{x.mean():+.4f}" if x.mean() < 0.5 else f"{x.mean():.3f}"

    print()
    print(f"{'condition':<32} | {'pattern acc':>14} | {'seq disc':>14}")
    print("-" * 78)
    print(f"{'pattern_only':<32} | {f(pa_alone):>14} | {'—':>14}")
    print(f"{'seq_only':<32} | {'—':>14} | {f(sq_alone):>14}")
    print(f"{'multi_pat_then_seq (lr=0.07)':<32} | {f(pa_pat_seq):>14} | {f(sq_pat_seq):>14}")
    print(f"{'multi_seq_then_pat (lr=0.07)':<32} | {f(pa_seq_pat):>14} | {f(sq_seq_pat):>14}")
    print(f"{'multi_seq_then_pat (lr=0.14)':<32} | {f(pa_seq_pat14):>14} | {f(sq_seq_pat14):>14}")

    pa_keep_v1 = pa_pat_seq.mean() / max(1e-9, pa_alone.mean())
    sq_keep_v1 = sq_pat_seq.mean() / max(1e-9, sq_alone.mean())
    sq_keep_v2 = sq_seq_pat.mean() / max(1e-9, sq_alone.mean())
    pa_keep_v3 = pa_seq_pat14.mean() / max(1e-9, pa_alone.mean())
    sq_keep_v3 = sq_seq_pat14.mean() / max(1e-9, sq_alone.mean())

    print()
    print(f"pat_then_seq:  pat_keep={pa_keep_v1:.0%}  seq_keep={sq_keep_v1:.0%}  "
          f"{'WIN' if pa_keep_v1 >= 0.90 and sq_keep_v1 >= 0.90 else 'partial'}")
    print(f"seq_then_pat (canonical lr=0.07): seq_keep={sq_keep_v2:.0%}  "
          f"{'WIN' if sq_keep_v2 >= 0.90 else 'KNOWN-OPEN'}")
    print(f"seq_then_pat (elevated lr=0.14):  pat_keep={pa_keep_v3:.0%}  "
          f"seq_keep={sq_keep_v3:.0%}  "
          f"{'WIN' if sq_keep_v3 >= 0.90 and pa_keep_v3 >= 0.80 else 'partial'}")


if __name__ == "__main__":
    main()
