"""Eşik 3-extension: N>>1 sequence chain learning A→B→C→...→J on
the same field. Honest report: per-transition discrim positive?"""
import sys, time, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.sequence import (
    EligibilityTrace, present_event, relax_with_trace, stdp_hebbian_update,
)
from el.thermofield.plasticity import hebbian_update

GRID = 14
SEEDS = list(range(8))

def make_chain(n_links, grid, rng):
    """Place n_links+1 distinct anchor cells on the grid."""
    flat = rng.choice(grid*grid, size=n_links+1, replace=False)
    return [(int(i // grid), int(i % grid)) for i in flat]

def train_chain(field, anchors, n_epochs=20, hold=5, gap=2):
    trace = EligibilityTrace((field.cfg.rows, field.cfg.cols), decay=0.80)
    for _ in range(n_epochs):
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

def probe_link(trained_field, fresh_seed, A, B, hold=5, read_delay=6):
    """Discrim = T[B] after presenting A | trained - fresh."""
    cfg = trained_field.cfg
    trained_field.reset_temp()
    tr = EligibilityTrace((cfg.rows, cfg.cols), decay=0.80)
    present_event(trained_field, tr, [A], [1.0], hold=hold)
    relax_with_trace(trained_field, tr, read_delay)
    cue = float(trained_field.T[B])
    fresh = Field(cfg, seed=fresh_seed)
    ftr = EligibilityTrace((cfg.rows, cfg.cols), decay=0.80)
    present_event(fresh, ftr, [A], [1.0], hold=hold)
    relax_with_trace(fresh, ftr, read_delay)
    return cue - float(fresh.T[B])

def run(n_links):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    per_link = np.zeros((len(SEEDS), n_links))
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        anchors = make_chain(n_links, GRID, rng)
        field = Field(cfg, seed=seed)
        train_chain(field, anchors)
        for li in range(n_links):
            per_link[si, li] = probe_link(field, seed, anchors[li], anchors[li+1])
    return per_link

def main():
    for n in [1, 3, 5, 10]:
        t0 = time.time()
        d = run(n)
        # mean per-link discrim, fraction of links with mean>0
        link_means = d.mean(axis=0)
        link_se = d.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
        pos = (link_means - 2*link_se > 0).sum()
        overall = d.mean()
        print(f"chain={n:2d} links | overall_discrim={overall:+.4f}  "
              f"links_clearly_positive={pos}/{n}  ({time.time()-t0:.1f}s)")
        for i, (m, se) in enumerate(zip(link_means, link_se)):
            mark = "+" if m-2*se>0 else "-"
            print(f"     link {i}: {m:+.4f} ± {se:.4f}  [{mark}]")

if __name__ == "__main__":
    main()
