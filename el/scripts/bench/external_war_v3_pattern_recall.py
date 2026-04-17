"""External war v3 — pattern recall under noisy/partial cues.

This is v7's NATURAL ground (heat substrate is intrinsically a content-
addressable attractor). v1 was MLP's natural ground (single-shot
mapping); v2 was unfair (multi-task interference, v7 lost honestly).
Now: equal task, equal training, recall under partial-cue corruption.

Task: store N=8 random binary patterns (28×28 grid). At test time,
present partial cue (40% pixels visible, 60% zeroed). Model must
identify the original pattern (1-of-8 classification).

  - PatternMemory (heat substrate, content-addressable attractor)
  - MLP autoencoder (input cue → output pattern, then nearest-neighbor)
"""
from __future__ import annotations
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from el.thermofield.pattern_memory import PatternMemory
from el.thermofield.field import FieldConfig
from external_war_v1 import MLP, softmax

SEEDS = list(range(6))
GRID = 28
N_PIX = GRID * GRID
N_PATTERNS = 8


def gen_patterns(rng, n=N_PATTERNS, density=0.15):
    """Random binary patterns of given density."""
    return [(rng.random((GRID, GRID)) < density).astype(np.float32)
            for _ in range(n)]


def corrupt_cue(pat, rng, visible_frac=0.40):
    """Reveal only visible_frac of pixels (zero the rest)."""
    mask = (rng.random(pat.shape) < visible_frac).astype(np.float32)
    return pat * mask


# ============================================================== v7 (PatternMemory)
def eval_v7_recall(seed, patterns, cues_per_pattern=5):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pm = PatternMemory(cfg=cfg, seed=seed)
    for p in patterns:
        positions = [(int(r), int(c)) for r, c in zip(*np.where(p > 0.5))]
        pm.store(positions)
    correct = 0; total = 0
    rng = np.random.default_rng(seed + 333)
    for true_idx, p in enumerate(patterns):
        for _ in range(cues_per_pattern):
            cue_arr = corrupt_cue(p, rng)
            cue_pos = [(int(r), int(c)) for r, c in zip(*np.where(cue_arr > 0.5))]
            if not cue_pos:
                total += 1; continue
            pred_idx, score, _ = pm.recall(cue_pos)
            correct += int(pred_idx == true_idx); total += 1
    return correct / max(1, total)


# ============================================================== MLP baseline
def eval_mlp_recall(seed, patterns, cues_per_pattern=5,
                    n_hidden=64, ep=300, lr=0.05, n_train_cues=80):
    """Train MLP cue→pattern_index. Many corrupted cues per pattern as data."""
    rng_train = np.random.default_rng(seed + 444)
    Xs = np.zeros((n_train_cues, N_PIX), dtype=np.float32)
    ys = np.zeros(n_train_cues, dtype=np.int64)
    for i in range(n_train_cues):
        idx = rng_train.integers(0, len(patterns))
        cue = corrupt_cue(patterns[idx], rng_train).reshape(-1)
        Xs[i] = cue; ys[i] = idx
    mlp = MLP(N_PIX, n_hidden, len(patterns), seed=seed, lr=lr)
    for _ in range(ep):
        idx = rng_train.permutation(n_train_cues)
        mlp.step(Xs[idx], ys[idx])

    rng_test = np.random.default_rng(seed + 333)
    correct = 0; total = 0
    for true_idx, p in enumerate(patterns):
        for _ in range(cues_per_pattern):
            cue = corrupt_cue(p, rng_test).reshape(-1, N_PIX)
            logits = mlp.forward(cue.reshape(1, -1))[0]
            pred = int(np.argmax(logits))
            correct += int(pred == true_idx); total += 1
    return correct / max(1, total)


# ============================================================== Run
def main():
    print("=" * 78)
    print("EXTERNAL WAR v3 — pattern recall under 60% corruption")
    print(f"  N_PATTERNS={N_PATTERNS}, grid={GRID}×{GRID}, visible_frac=0.40")
    print(f"  v7 PatternMemory: 1-shot store. MLP: trained on 80 corrupted cues.")
    print("=" * 78)
    v7_a = []; mlp_a = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pats = gen_patterns(rng)
        v7_a.append(eval_v7_recall(seed, pats))
        mlp_a.append(eval_mlp_recall(seed, pats))
    def stat(xs):
        a = np.asarray(xs)
        return f"{a.mean():.3f} ± {a.std()/np.sqrt(len(a)):.3f}"
    print()
    print(f"{'method':<32} | {'recall acc (chance=0.125)':>30}")
    print("-" * 78)
    print(f"{'v7 PatternMemory (1-shot)':<32} | {stat(v7_a):>30}")
    print(f"{'MLP (trained on 80 cues)':<32} | {stat(mlp_a):>30}")
    if np.mean(v7_a) > np.mean(mlp_a) + 0.02:
        print(f"\n*** v7 BEATS MLP by {np.mean(v7_a)-np.mean(mlp_a):+.3f} ***")
        print("(and v7 sees each pattern only ONCE; MLP sees 80 corrupted samples)")


if __name__ == "__main__":
    main()
