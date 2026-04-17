"""EŞİK 4 — PERSISTENCE PROBE.

Train PatternMemory on N=16 patterns, save, reload in fresh state,
train N=16 MORE patterns. Compare final 32-pattern recall vs:
  (a) train 32 patterns from scratch in one shot
  (b) train second-batch-only from scratch (no first-batch memory)

PASS if loaded+continued >> scratch-on-second-batch (proves persistence).
"""
from __future__ import annotations
import sys, time, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern, corrupt


def fresh_pm(grid, seed):
    return PatternMemory(cfg=FieldConfig(rows=grid, cols=grid), seed=seed)


def measure(pm, patterns, drop=0.5, trials=4, rng=None, label_offset=0):
    """label_offset: when pm.patterns indices [0..k] correspond to labels
    [label_offset..label_offset+k] in the combined truth space."""
    if rng is None: rng = np.random.default_rng(0)
    correct = total = 0
    for i, p in enumerate(patterns):
        true_idx = i  # combined-space label
        for _ in range(trials):
            cue = corrupt(p, drop, rng)
            pred, _, _ = pm.recall(cue)
            mapped_pred = pred + label_offset
            correct += int(mapped_pred == true_idx); total += 1
    return correct / max(1, total)


def trial(seed, grid=28, n_first=16, n_second=16, density_k=20):
    rng = np.random.default_rng(seed)
    A = [random_pattern(grid, grid, density_k, rng) for _ in range(n_first)]
    B = [random_pattern(grid, grid, density_k, rng) for _ in range(n_second)]

    # condition 1: monolithic — all 32 from scratch
    pm_mono = fresh_pm(grid, seed)
    for p in A + B: pm_mono.store(p)
    acc_mono = measure(pm_mono, A + B, rng=np.random.default_rng(seed + 1))

    # condition 2: load+continue — train A, save, fresh load, train B
    pm1 = fresh_pm(grid, seed)
    for p in A: pm1.store(p)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    pm1.save(path)
    pm2 = PatternMemory.load(path)
    for p in B: pm2.store(p)
    acc_load = measure(pm2, A + B, rng=np.random.default_rng(seed + 1))

    # condition 3: scratch on B only (control — proves we need persistence)
    # pm only has B stored at indices 0..n_second-1; combined truth has B at
    # indices n_first..n_first+n_second-1, so for B test samples we offset
    # predictions by n_first (A samples are unrecallable by design).
    pm_scratch = fresh_pm(grid, seed)
    for p in B: pm_scratch.store(p)
    rng_s = np.random.default_rng(seed + 1)
    correct_A = 0  # pm has no A memory; can never match A label
    for i, p in enumerate(A):  # consume rng to keep parity
        for _ in range(4):
            corrupt(p, 0.5, rng_s)
    acc_B_scratch = measure(pm_scratch, B, rng=rng_s, label_offset=0)
    # combined accuracy: only B-half can contribute
    acc_scratch = (correct_A + acc_B_scratch * len(B)) / (len(A) + len(B))

    # round-trip check: did save→load preserve recall on A?
    pm1b = fresh_pm(grid, seed)
    for p in A: pm1b.store(p)
    pm1b.save(path)
    pm1c = PatternMemory.load(path)
    rng_rt = np.random.default_rng(seed + 7)
    rt_before = measure(pm1b, A, rng=np.random.default_rng(seed + 7))
    rt_after = measure(pm1c, A, rng=np.random.default_rng(seed + 7))
    return acc_mono, acc_load, acc_scratch, rt_before, rt_after


def main():
    print("=" * 78)
    print("EŞİK 4 — PERSISTENCE PROBE")
    print("  grid 28×28, n_first=16, n_second=16, seeds=8")
    print("=" * 78)
    seeds = list(range(8))
    rows = [trial(s) for s in seeds]
    arr = np.asarray(rows)
    means = arr.mean(0); sems = arr.std(0) / np.sqrt(len(seeds))
    labels = ["mono (32 scratch)", "load+continue (16+16)",
              "B-only scratch (no A memory)", "round-trip before save",
              "round-trip after  load"]
    print(f"\n{'condition':<32} {'acc':>10} {'sem':>8}")
    for lbl, m, s in zip(labels, means, sems):
        print(f"  {lbl:<30} {m:>10.3f} {s:>8.3f}")
    print()
    keep = means[1] / means[0] if means[0] > 0 else 0.0
    rt_drift = abs(means[3] - means[4])
    print(f"  load_keep = load_continue / mono       = {keep:.2%}")
    print(f"  round-trip drift |before - after|       = {rt_drift:.3f}")
    print(f"  beats B-only scratch?                   = "
          f"{'YES (' + f'{means[1]-means[2]:+.3f}' + ')' if means[1] > means[2] else 'NO'}")
    verdict = ("WIN" if (keep >= 0.85 and rt_drift < 0.05 and means[1] > means[2])
               else "PARTIAL")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
