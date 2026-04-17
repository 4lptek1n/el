"""Eşik 2 zorla — extreme grid scaling (224×224, N=128 ve N=256).

Tek-shot rapor scripti. Kapasite kırılma noktasını arar: pattern
memory @ 224×224 grid, N=128 ve 256 paterni ile, 6 seed × 10 trial.
Sonuç tablosu replit.md'ye girer; CI'ya eklenmez (yavaş).
"""
from __future__ import annotations
import sys, time, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern

GRID = 224
SEEDS = list(range(6))
TRIALS = 10
DROP = 0.5
PAT_SIZE = max(4, int(0.025 * GRID * GRID))
WTA_K = max(PAT_SIZE + 2, int(0.04 * GRID * GRID))


def noisy_cue(p, drop, rng, n_cells, cols):
    kn = max(1, int(round(len(p) * (1 - drop))))
    keep = [p[i] for i in sorted(rng.choice(len(p), kn, replace=False))]
    pset = set(p); ds = []
    while len(ds) < kn:
        idx = int(rng.integers(0, n_cells)); rc = (idx // cols, idx % cols)
        if rc not in pset and rc not in ds: ds.append(rc)
    return keep + ds


def run(N, seed):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    rng = np.random.default_rng(seed)
    mem = PatternMemory(cfg=cfg, seed=seed,
                        write_steps=15, write_lr=0.07, write_decay=0.005,
                        recall_steps=8,
                        wta_k=WTA_K, wta_suppression=0.3, rule="hebb")
    patterns = [random_pattern(GRID, GRID, k=PAT_SIZE, rng=rng) for _ in range(N)]
    t0 = time.time()
    for p in patterns: mem.store(p)
    store_t = time.time() - t0
    nc = GRID * GRID; correct = 0
    rng2 = np.random.default_rng(seed * 31 + 7)
    t0 = time.time()
    for _ in range(TRIALS):
        i = int(rng2.integers(0, N))
        cue = noisy_cue(patterns[i], DROP, rng2, nc, GRID)
        b, _, _ = mem.recall(cue)
        if b == i: correct += 1
    rec_t = time.time() - t0
    return correct / TRIALS, store_t, rec_t


def main():
    print(f"=== Extreme capacity probe @ {GRID}×{GRID} grid ===")
    print(f"PAT_SIZE={PAT_SIZE}  WTA_K={WTA_K}  drop={DROP}  "
          f"seeds={len(SEEDS)}  trials={TRIALS}")
    print(f"{'N':>5} | {'mean_acc':>10} | {'std':>6} | {'avg_store_s':>12} | {'avg_recall_ms':>14}")
    for N in [128, 256]:
        accs = []; sts = []; rts = []
        for seed in SEEDS:
            t0 = time.time()
            acc, st, rt = run(N, seed)
            accs.append(acc); sts.append(st); rts.append(rt)
            print(f"  seed={seed} N={N:3d}  acc={acc:.2f}  "
                  f"store={st:.1f}s  recall={rt*1000/TRIALS:.1f}ms/cue  "
                  f"({time.time()-t0:.1f}s wall)", flush=True)
        a = np.array(accs)
        print(f"{N:>5} | {a.mean():>10.3f} | {a.std(ddof=1):>6.3f} | "
              f"{np.mean(sts):>12.1f} | {np.mean(rts)*1000/TRIALS:>14.2f}", flush=True)


if __name__ == "__main__":
    main()
