"""External war v5 — compute-budget-matched comparison.

Previous wars gave MLP 300 epochs × N samples = 76800 gradient steps
while substrate got N store operations (1 epoch equivalent). 76× compute
asymmetry. This benchmark fixes that.

Compute budget = number of pattern presentations to the model.
- Substrate: 1 store per pattern → N total presentations
- MLP: ep × N gradient steps → ep=1 matches substrate exactly

We sweep ep ∈ {1, 3, 10, 30, 300} so the reader sees the trade-off
honestly.
"""
from __future__ import annotations
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern, corrupt
from external_war_v1 import MLP

GRID = 64
DENSITY_K = max(8, int(0.02 * GRID * GRID))   # ~2% sparse
N = 64
SEEDS = [0, 1, 2]
N_PIX = GRID * GRID


def vec(positions):
    v = np.zeros(N_PIX, dtype=np.float32)
    for r, c in positions:
        v[r * GRID + c] = 1.0
    return v


def eval_v7(seed, patterns, cues_per=4, drop=0.5):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pm = PatternMemory(cfg=cfg, seed=seed)
    for p in patterns:
        pm.store(p)
    rng = np.random.default_rng(seed + 333)
    correct = total = 0
    for true_idx, p in enumerate(patterns):
        for _ in range(cues_per):
            cue = corrupt(p, drop, rng)
            if not cue:
                total += 1; continue
            pred, _, _ = pm.recall(cue)
            correct += int(pred == true_idx); total += 1
    return correct / max(1, total)


def eval_mlp(seed, patterns, ep, n_hidden=64, lr=0.05, cues_per=4, drop=0.5):
    n = len(patterns)
    Xs = np.array([vec(p) for p in patterns], dtype=np.float32)
    ys = np.arange(n, dtype=np.int64)
    mlp = MLP(N_PIX, n_hidden, n, seed=seed, lr=lr)
    rng = np.random.default_rng(seed + 444)
    for _ in range(ep):
        idx = rng.permutation(n)
        mlp.step(Xs[idx], ys[idx])
    rng_test = np.random.default_rng(seed + 333)
    correct = total = 0
    for true_idx, p in enumerate(patterns):
        for _ in range(cues_per):
            cue = corrupt(p, drop, rng_test)
            cue_v = vec(cue).reshape(1, -1)
            pred = int(np.argmax(mlp.forward(cue_v)[0]))
            correct += int(pred == true_idx); total += 1
    return correct / max(1, total)


def main():
    print("=" * 78)
    print(f"WAR v5 — compute-matched, grid={GRID}, N={N}, k={DENSITY_K}, drop=0.5")
    print(f"  Substrate: 1 store/pattern (= 1 epoch equivalent compute)")
    print(f"  MLP: vary ep ∈ {{1, 3, 10, 30, 300}} = 1× to 300× substrate compute")
    print("=" * 78)

    rng_pool = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        rng_pool.append([random_pattern(GRID, GRID, DENSITY_K, rng)
                         for _ in range(N)])

    v7s = [eval_v7(seed, rng_pool[i]) for i, seed in enumerate(SEEDS)]
    v7m = np.mean(v7s); v7se = np.std(v7s) / np.sqrt(len(v7s))
    print(f"\n  v7 substrate (1-shot store)        | {v7m:.3f} ± {v7se:.3f}")
    print(f"  {'-'*72}")
    print(f"  {'MLP epochs':<22} | {'recall acc':>14} | {'compute ratio':>14}")

    rows = []
    for ep in [1, 3, 10, 30, 100, 300]:
        accs = [eval_mlp(seed, rng_pool[i], ep) for i, seed in enumerate(SEEDS)]
        m = np.mean(accs); se = np.std(accs) / np.sqrt(len(accs))
        verdict = "v7 wins" if v7m > m + 0.03 else \
                  "MLP wins" if m > v7m + 0.03 else "tie"
        rows.append((ep, m, verdict))
        print(f"  ep={ep:<19} | {m:.3f} ± {se:.3f} | {ep:>14}× | {verdict}")

    # Find compute-equivalent regime
    matched = next((r for r in rows if r[0] == 1), None)
    if matched and v7m > matched[1] + 0.03:
        print(f"\n*** AT MATCHED COMPUTE (ep=1): "
              f"v7={v7m:.3f} vs MLP={matched[1]:.3f} → v7 WINS by "
              f"{v7m - matched[1]:+.3f} ***")
    elif matched and matched[1] > v7m + 0.03:
        print(f"\nAt matched compute (ep=1): MLP wins anyway "
              f"({matched[1]:.3f} > {v7m:.3f}). Substrate has no compute advantage.")
    else:
        print(f"\nAt matched compute (ep=1): tie within noise.")


if __name__ == "__main__":
    main()
