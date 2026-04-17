"""Lighter version of extreme_capacity — fewer seeds, fewer trials per pattern,
to avoid sandbox OOM/timeout kills. Same probe; smaller statistics.
"""
from __future__ import annotations
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern, corrupt


def run(grid, N, seeds=(0, 1), trials_per_pat=2, drop=0.5):
    density_k = max(8, int(0.01 * grid * grid))
    accs, t_writes, t_recalls = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        patterns = [random_pattern(grid, grid, density_k, rng) for _ in range(N)]
        pm = PatternMemory(cfg=FieldConfig(rows=grid, cols=grid), seed=seed)
        t0 = time.time()
        for p in patterns: pm.store(p)
        t_writes.append(time.time() - t0)
        rng_t = np.random.default_rng(seed + 333)
        ok = tot = 0
        t0 = time.time()
        for true_idx, p in enumerate(patterns):
            for _ in range(trials_per_pat):
                pred, _, _ = pm.recall(corrupt(p, drop, rng_t))
                ok += int(pred == true_idx); tot += 1
        t_recalls.append(time.time() - t0)
        accs.append(ok / max(1, tot))
    a = np.asarray(accs)
    return dict(acc_mean=float(a.mean()), acc_sem=float(a.std() / np.sqrt(len(a))),
                write_s=float(np.mean(t_writes)), recall_s=float(np.mean(t_recalls)),
                density_k=density_k)


def main():
    import sys as _s
    row_idx = int(_s.argv[1]) if len(_s.argv) > 1 else -1
    rows_cfg = [(64, 32), (64, 64), (128, 64), (128, 128),
                (192, 128), (224, 128), (224, 256)]
    if row_idx >= 0:
        rows_cfg = [rows_cfg[row_idx]]
    print("=" * 78)
    print("EŞİK 2 EXTREME (lite)")
    print("=" * 78)
    for grid, N in rows_cfg:
        chance = 1.0 / N
        r = run(grid, N)
        v = "PASS" if r["acc_mean"] >= 0.5 else "WEAK" if r["acc_mean"] >= 0.25 else "FAIL"
        print(f"  grid={grid:>3}×{grid:<3}  N={N:>3}  k={r['density_k']:>3}  "
              f"chance={chance:.3f}  acc={r['acc_mean']:.3f}±{r['acc_sem']:.3f}  "
              f"write={r['write_s']:.1f}s  recall={r['recall_s']:.1f}s  [{v}]",
              flush=True)


if __name__ == "__main__":
    main()
