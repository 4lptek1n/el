"""Sequence chain v3 — slower trace decay, longer hold per epoch,
random pair sampling per epoch (not strict order)."""
import sys, time, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.sequence import (
    EligibilityTrace, present_event, relax_with_trace, stdp_hebbian_update,
)
from el.thermofield.plasticity import hebbian_update

GRID=14; SEEDS=list(range(8))

def make_chain_spread(n, grid, rng, md=6):
    chosen=[]
    for _ in range(300):
        if len(chosen)==n+1: break
        c=(int(rng.integers(0,grid)),int(rng.integers(0,grid)))
        if all(abs(c[0]-x[0])+abs(c[1]-x[1])>=md for x in chosen):
            chosen.append(c)
    while len(chosen)<n+1:
        chosen.append((int(rng.integers(0,grid)),int(rng.integers(0,grid))))
    return chosen

def train(field, anchors, n_epochs, hold, gap, trace_decay, lr, hebb_lr):
    trace = EligibilityTrace((field.cfg.rows,field.cfg.cols), decay=trace_decay)
    pairs = [(anchors[i], anchors[i+1]) for i in range(len(anchors)-1)]
    rng = np.random.default_rng(0)
    for ep in range(n_epochs):
        order = rng.permutation(len(pairs))
        for j in order:
            A, B = pairs[j]
            field.reset_temp(); trace.reset()
            present_event(field, trace, [A], [1.0], hold=hold)
            relax_with_trace(field, trace, gap)
            field._clamp_positions=[]; field._clamp_values=[]
            field.inject([B], [1.0])
            stdp_hebbian_update(field, trace, lr=lr)
            for _ in range(hold):
                field.step(); trace.update(field.T)
            hebbian_update(field, lr=hebb_lr, decay=0.001)
    field.reset_temp()

def probe(trained, seed, A, B):
    cfg = trained.cfg
    trained.reset_temp()
    tr = EligibilityTrace((cfg.rows,cfg.cols), decay=0.80)
    present_event(trained, tr, [A], [1.0], hold=5)
    relax_with_trace(trained, tr, 6)
    cue=float(trained.T[B])
    fresh=Field(cfg, seed=seed)
    ftr=EligibilityTrace((cfg.rows,cfg.cols), decay=0.80)
    present_event(fresh, ftr, [A], [1.0], hold=5)
    relax_with_trace(fresh, ftr, 6)
    return cue-float(fresh.T[B])

def run(n, ep, hold, td, lr, hlr, md):
    cfg=FieldConfig(rows=GRID,cols=GRID)
    d=np.zeros((len(SEEDS),n))
    for si,seed in enumerate(SEEDS):
        rng=np.random.default_rng(seed)
        anchors=make_chain_spread(n,GRID,rng,md=md)
        f=Field(cfg,seed=seed)
        train(f, anchors, ep, hold=hold, gap=2, trace_decay=td, lr=lr, hebb_lr=hlr)
        for li in range(n):
            d[si,li]=probe(f,seed,anchors[li],anchors[li+1])
    return d

def main():
    print(f"{'cfg':50} | {'overall':>10} | {'clear_pos':>10}")
    for n in [5, 10]:
      for ep in [80, 150]:
        for td in [0.85, 0.92]:
          for lr in [0.10]:
            for hlr in [0.03, 0.10]:
              for md in [6]:
                t0=time.time()
                d = run(n, ep, hold=5, td=td, lr=lr, hlr=hlr, md=md)
                lm=d.mean(0); lse=d.std(0,ddof=1)/np.sqrt(len(SEEDS))
                pos=int((lm-2*lse>0).sum())
                tag=f"n={n} ep={ep} td={td} lr={lr} hlr={hlr} md={md}"
                print(f"{tag:50} | {d.mean():+.4f}    | {pos:>3}/{n}  ({time.time()-t0:.1f}s)", flush=True)

if __name__ == "__main__":
    main()
