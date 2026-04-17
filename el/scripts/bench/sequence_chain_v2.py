"""Sequence chain v2 — more epochs, anti-Hebb on negatives, larger
inter-anchor distance to reduce noise."""
import sys, time, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.sequence import (
    EligibilityTrace, present_event, relax_with_trace, stdp_hebbian_update,
)
from el.thermofield.plasticity import hebbian_update

GRID = 14; SEEDS = list(range(8))

def make_chain_spread(n_links, grid, rng, min_dist=4):
    """Anchors with min_dist Manhattan distance to reduce overlap."""
    chosen = []
    for _ in range(200):
        if len(chosen) == n_links+1: break
        cand = (int(rng.integers(0,grid)), int(rng.integers(0,grid)))
        if all(abs(cand[0]-c[0]) + abs(cand[1]-c[1]) >= min_dist for c in chosen):
            chosen.append(cand)
    while len(chosen) < n_links+1:
        chosen.append((int(rng.integers(0,grid)), int(rng.integers(0,grid))))
    return chosen

def train(field, anchors, n_epochs, hold=5, gap=2):
    trace = EligibilityTrace((field.cfg.rows, field.cfg.cols), decay=0.80)
    for ep in range(n_epochs):
        for i in range(len(anchors)-1):
            A, B = anchors[i], anchors[i+1]
            field.reset_temp(); trace.reset()
            present_event(field, trace, [A], [1.0], hold=hold)
            relax_with_trace(field, trace, gap)
            field._clamp_positions=[]; field._clamp_values=[]
            field.inject([B], [1.0])
            stdp_hebbian_update(field, trace, lr=0.07)
            for _ in range(hold):
                field.step(); trace.update(field.T)
            hebbian_update(field, lr=0.07, decay=0.001)
    field.reset_temp()

def probe(trained, seed, A, B, hold=5, rd=6):
    cfg = trained.cfg
    trained.reset_temp()
    tr = EligibilityTrace((cfg.rows,cfg.cols), decay=0.80)
    present_event(trained, tr, [A], [1.0], hold=hold)
    relax_with_trace(trained, tr, rd)
    cue = float(trained.T[B])
    fresh = Field(cfg, seed=seed)
    ftr = EligibilityTrace((cfg.rows,cfg.cols), decay=0.80)
    present_event(fresh, ftr, [A], [1.0], hold=hold)
    relax_with_trace(fresh, ftr, rd)
    return cue - float(fresh.T[B])

def run(n_links, n_epochs, min_dist):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    d = np.zeros((len(SEEDS), n_links))
    for si,seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        anchors = make_chain_spread(n_links, GRID, rng, min_dist=min_dist)
        f = Field(cfg, seed=seed)
        train(f, anchors, n_epochs)
        for li in range(n_links):
            d[si,li] = probe(f, seed, anchors[li], anchors[li+1])
    return d

def main():
    for cfg in [(5, 60, 4), (10, 60, 4), (10, 100, 5), (10, 200, 5),
                (10, 60, 6), (10, 200, 6)]:
        n, ep, md = cfg
        t0 = time.time()
        d = run(n, ep, md)
        lm = d.mean(0); lse = d.std(0,ddof=1)/np.sqrt(len(SEEDS))
        pos = int((lm - 2*lse > 0).sum())
        print(f"n={n:2d} epochs={ep:3d} min_dist={md} | "
              f"overall={d.mean():+.4f} clear_pos={pos:2d}/{n}  "
              f"({time.time()-t0:.1f}s)", flush=True)

if __name__ == "__main__":
    main()
