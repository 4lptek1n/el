"""External war v4 — high-N pattern recall regime.

Hypothesis: in v3 (N=8) MLP wins because the task is trivially within
its parameter capacity. As N grows past the model's hidden width, MLP
saturates while the substrate's content-addressable storage continues
to scale (each pattern carves its own attractor; substrate is a
distributed sparse memory, not a parametric classifier).

Sweep N ∈ {16, 32, 64, 128} at grid 32×32 (small enough to be fair to
both, large enough to hold many patterns). MLP gets 1 clean
sample/class + 300 epochs (matches v7's 1-shot regime, fair).

This is the test the previous external wars avoided. Honest
publication of result regardless of outcome.
"""
from __future__ import annotations
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern, corrupt
from external_war_v1 import MLP

GRID = 32
N_PIX = GRID * GRID
SEEDS = [0, 1, 2]
DENSITY_K = max(8, int(0.04 * GRID * GRID))   # ~4% sparse → k≈40


def cues_for(p, n_cues, drop, rng):
    return [corrupt(p, drop, rng) for _ in range(n_cues)]


def pattern_to_vec(positions):
    v = np.zeros(N_PIX, dtype=np.float32)
    for r, c in positions:
        v[r * GRID + c] = 1.0
    return v


def eval_v7(seed, patterns, cues_per=4, drop=0.5):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pm = PatternMemory(cfg=cfg, seed=seed)
    t0 = time.time()
    for p in patterns:
        pm.store(p)
    write_s = time.time() - t0

    rng = np.random.default_rng(seed + 333)
    correct = total = 0
    t0 = time.time()
    for true_idx, p in enumerate(patterns):
        for _ in range(cues_per):
            cue = corrupt(p, drop, rng)
            if not cue:
                total += 1; continue
            pred, _, _ = pm.recall(cue)
            correct += int(pred == true_idx); total += 1
    return correct / max(1, total), write_s, time.time() - t0


def eval_mlp(seed, patterns, n_hidden=64, ep=300, lr=0.05,
             cues_per=4, drop=0.5):
    """1 clean sample/class, ep=300 — matches substrate's 1-shot budget."""
    n = len(patterns)
    Xs = np.array([pattern_to_vec(p) for p in patterns], dtype=np.float32)
    ys = np.arange(n, dtype=np.int64)
    mlp = MLP(N_PIX, n_hidden, n, seed=seed, lr=lr)
    rng = np.random.default_rng(seed + 444)
    t0 = time.time()
    for _ in range(ep):
        idx = rng.permutation(n)
        mlp.step(Xs[idx], ys[idx])
    train_s = time.time() - t0

    rng_test = np.random.default_rng(seed + 333)
    correct = total = 0
    t0 = time.time()
    for true_idx, p in enumerate(patterns):
        for _ in range(cues_per):
            cue = corrupt(p, drop, rng_test)
            cue_v = pattern_to_vec(cue).reshape(1, -1)
            pred = int(np.argmax(mlp.forward(cue_v)[0]))
            correct += int(pred == true_idx); total += 1
    return correct / max(1, total), train_s, time.time() - t0


def main():
    print("=" * 78)
    print(f"EXTERNAL WAR v4 — high-N regime, grid={GRID}×{GRID}, k={DENSITY_K}, drop=0.5")
    print(f"  Both 1-shot per class. MLP: hidden=64, 300 ep.")
    print("=" * 78)
    print(f"{'N':>4} | {'chance':>7} | {'v7 acc':>14} | {'MLP acc':>14} | {'winner':>10}")
    print("-" * 78)
    rows = []
    for N in [16, 32, 64, 128, 256]:
        chance = 1.0 / N
        v7s = []; mlps = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            patterns = [random_pattern(GRID, GRID, DENSITY_K, rng)
                        for _ in range(N)]
            a_v7, _, _ = eval_v7(seed, patterns)
            a_ml, _, _ = eval_mlp(seed, patterns)
            v7s.append(a_v7); mlps.append(a_ml)
        v7m, mlm = np.mean(v7s), np.mean(mlps)
        v7s_se = np.std(v7s) / np.sqrt(len(v7s))
        mls_se = np.std(mlps) / np.sqrt(len(mlps))
        if v7m > mlm + 0.03:
            winner = "v7 ✓"
        elif mlm > v7m + 0.03:
            winner = "MLP"
        else:
            winner = "tie"
        print(f"{N:>4} | {chance:>7.4f} | "
              f"{v7m:.3f} ± {v7s_se:.3f} | "
              f"{mlm:.3f} ± {mls_se:.3f} | {winner:>10}")
        rows.append((N, v7m, mlm, winner))
    print()
    wins = [r for r in rows if r[3] == "v7 ✓"]
    if wins:
        print(f"*** v7 BEATS MLP in {len(wins)}/{len(rows)} regimes: "
              f"N ∈ {{{', '.join(str(r[0]) for r in wins)}}} ***")
    else:
        print("Honest result: MLP wins or ties at every N tested.")


if __name__ == "__main__":
    main()
