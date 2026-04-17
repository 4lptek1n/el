"""el_anomaly — substrate as streaming anomaly detector.

Substrate's natural turf: temporally-bound pattern memory. Train on a
clean signal stream (sine + noise), substrate stores the recurring
short-window patterns. At test time, score each window by how poorly
the substrate "predicts" it (low recall score = surprise = anomaly).
Compare to z-score baseline. Real ROC-AUC numbers.

This is the kind of thing a 1-MB-RAM edge device CAN do that ChatGPT
cannot: persistent on-device anomaly learning, no cloud, no gradient.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory


def make_stream(n: int = 4000, *, anomaly_rate: float = 0.02,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sine + harmonic + noise; inject spikes/dropouts as anomalies."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = (np.sin(2 * np.pi * t / 50)
         + 0.5 * np.sin(2 * np.pi * t / 17)
         + 0.15 * rng.standard_normal(n))
    labels = np.zeros(n, dtype=np.int8)
    n_anom = int(n * anomaly_rate)
    anom_idx = rng.choice(np.arange(50, n - 50), size=n_anom, replace=False)
    for i in anom_idx:
        kind = rng.integers(0, 3)
        if kind == 0:    # spike
            x[i] += 4.0 * rng.choice([-1, 1])
        elif kind == 1:  # plateau / dropout
            x[i:i + 5] = x[i]
        else:            # frequency burst
            x[i:i + 8] += 2.5 * np.sin(2 * np.pi * np.arange(8) / 3)
        labels[max(0, i - 2):min(n, i + 8)] = 1
    return x.astype(np.float32), labels


def window_to_pattern(win: np.ndarray, grid: int, bins: int = 16
                      ) -> list[tuple[int, int]]:
    """Quantize each window sample to a `bins`-bin level; place a cell at
    (i, level) — gives a 2D fingerprint of the window's shape."""
    L = len(win)
    lo, hi = -3.0, 3.0
    q = np.clip(((win - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    cells = set()
    for i in range(L):
        # spread row-position across rows: i -> row band
        row_band = (i * grid) // L
        col_band = (q[i] * grid) // bins
        for dr in range(2):
            cells.add((min(grid - 1, row_band + dr), col_band))
    return sorted(cells)


def substrate_scores(stream: np.ndarray, *, win: int = 32, grid: int = 48,
                     train_frac: float = 0.5, seed: int = 0
                     ) -> np.ndarray:
    """Train substrate on first `train_frac` of stream, score every window."""
    n = len(stream)
    pm = PatternMemory(cfg=FieldConfig(rows=grid, cols=grid), seed=seed)
    train_end = int(n * train_frac)
    # store windows densely in train region
    for s in range(0, train_end - win, max(1, win // 4)):
        pm.store(window_to_pattern(stream[s:s + win], grid))
    # score every window: low recall jaccard = anomaly
    scores = np.zeros(n, dtype=np.float32)
    for s in range(0, n - win):
        cue = window_to_pattern(stream[s:s + win], grid)
        idx, jacc, _ = pm.recall(cue)
        scores[s + win // 2] = 1.0 - float(jacc)  # surprise
    return scores


def zscore_scores(stream: np.ndarray, *, win: int = 32) -> np.ndarray:
    """Baseline: rolling |z-score| of each sample vs prior window."""
    n = len(stream)
    out = np.zeros(n, dtype=np.float32)
    for i in range(win, n):
        prev = stream[i - win:i]
        mu, sd = float(prev.mean()), float(prev.std() + 1e-6)
        out[i] = abs((stream[i] - mu) / sd)
    return out


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney U formulation."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # rank-based AUC
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--anomaly-rate", type=float, default=0.02)
    p.add_argument("--win", type=int, default=32)
    p.add_argument("--grid", type=int, default=48)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    print(f"[setup] n={args.n}, anomaly_rate={args.anomaly_rate}, "
          f"win={args.win}, grid={args.grid}, seeds={args.seeds}")
    print(f"[setup] substrate trained on first 50% of stream, "
          f"scored on full stream\n")

    sub_aucs, z_aucs = [], []
    for seed in range(args.seeds):
        stream, labels = make_stream(args.n, anomaly_rate=args.anomaly_rate,
                                     seed=seed)
        t0 = time.time()
        sub_s = substrate_scores(stream, win=args.win, grid=args.grid,
                                 seed=seed)
        z_s = zscore_scores(stream, win=args.win)
        # only score positions that have valid scores from both
        valid = np.arange(args.win, args.n - args.win)
        sub_auc = auc_roc(sub_s[valid], labels[valid])
        z_auc = auc_roc(z_s[valid], labels[valid])
        sub_aucs.append(sub_auc)
        z_aucs.append(z_auc)
        n_anom = int(labels.sum())
        print(f"  seed {seed}: anomalies={n_anom:4d}  "
              f"substrate_AUC={sub_auc:.3f}  z_AUC={z_auc:.3f}  "
              f"({time.time()-t0:.1f}s)")

    sa, za = np.array(sub_aucs), np.array(z_aucs)
    print(f"\n[RESULT] substrate AUC: {sa.mean():.3f} ± {sa.std():.3f}")
    print(f"[RESULT] z-score   AUC: {za.mean():.3f} ± {za.std():.3f}")
    print(f"[RESULT] delta     :    {(sa - za).mean():+.3f} "
          f"({'substrate WINS' if sa.mean() > za.mean() else 'z-score wins'})")


if __name__ == "__main__":
    main()
