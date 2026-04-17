"""Eşik 2 zorla — extreme grid scaling probe (224×224, large N).

How many patterns can the substrate hold at MNIST-scale grid?
This is the single biggest "is it embedded-relevant" question.
"""
from __future__ import annotations
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern, corrupt


def run(grid, N, seeds=(0, 1, 2), trials_per_pat=4, density_k=None, drop=0.5):
    """Return mean recall accuracy and wall time per seed."""
    if density_k is None:
        # ~1% sparse pattern — typical neural code density
        density_k = max(8, int(0.01 * grid * grid))
    accs = []; t_writes = []; t_recalls = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        patterns = [random_pattern(grid, grid, density_k, rng) for _ in range(N)]
        cfg = FieldConfig(rows=grid, cols=grid)
        pm = PatternMemory(cfg=cfg, seed=seed)

        t0 = time.time()
        for p in patterns:
            pm.store(p)
        t_writes.append(time.time() - t0)

        rng_test = np.random.default_rng(seed + 333)
        correct = total = 0
        t0 = time.time()
        for true_idx, p in enumerate(patterns):
            for _ in range(trials_per_pat):
                cue = corrupt(p, drop, rng_test)
                pred, _, _ = pm.recall(cue)
                correct += int(pred == true_idx); total += 1
        t_recalls.append(time.time() - t0)
        accs.append(correct / max(1, total))
    accs = np.asarray(accs)
    return {
        "acc_mean": float(accs.mean()),
        "acc_sem": float(accs.std() / np.sqrt(len(accs))),
        "write_s": float(np.mean(t_writes)),
        "recall_s": float(np.mean(t_recalls)),
        "density_k": density_k,
    }


def main():
    print("=" * 78)
    print("EŞİK 2 EXTREME — large grid capacity")
    print("=" * 78)
    rows = []
    for grid, N in [(64, 32), (64, 64), (128, 64), (128, 128), (224, 128), (224, 256)]:
        chance = 1.0 / N
        r = run(grid, N)
        verdict = ("PASS" if r["acc_mean"] >= 0.5
                   else "WEAK" if r["acc_mean"] >= 0.25 else "FAIL")
        rows.append((grid, N, chance, r, verdict))
        print(f"  grid={grid:>3}×{grid:<3}  N={N:>3}  k={r['density_k']:>3}  "
              f"chance={chance:.3f}  acc={r['acc_mean']:.3f} ± {r['acc_sem']:.3f}  "
              f"write={r['write_s']:.1f}s  recall={r['recall_s']:.1f}s  [{verdict}]")
    print()
    print("Verdict per row written above.")


if __name__ == "__main__":
    main()
