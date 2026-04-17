"""Sequence chain v4 — try 3 architectural fixes:
  (a) chain-as-stream: single eligibility trace across whole chain
  (b) context-gated STDP: scale lr by 1/(1+|B|) — saturated edges
      get reinforced less, fresh ones grab the trace
  (c) anti-Hebb B-decay: each STDP step also decays B globally,
      so older A→B paths fade as new B→C grows
"""
import sys, time, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.sequence import EligibilityTrace, present_event, relax_with_trace

GRID=14; SEEDS=list(range(8))

def stdp_gated(field, trace, lr, gate_strength=0.0, b_decay=0.0):
    """STDP with optional context-gating + global B-decay."""
    T = field.T; E = trace.E
    co_h = T[:, 1:] * E[:, :-1]   # right neighbor active now × left active recently
    co_v = T[1:, :] * E[:-1, :]
    if gate_strength > 0:
        gate_h = 1.0 / (1.0 + gate_strength * np.abs(field.B_right))
        gate_v = 1.0 / (1.0 + gate_strength * np.abs(field.B_down))
    else:
        gate_h = gate_v = 1.0
    field.B_right += lr * co_h * gate_h
    field.B_down  += lr * co_v * gate_v
    if b_decay > 0:
        field.B_right *= (1 - b_decay)
        field.B_down  *= (1 - b_decay)
    np.clip(field.B_right, -1.0, 1.0, out=field.B_right)
    np.clip(field.B_down,  -1.0, 1.0, out=field.B_down)

def make_chain(n, grid, rng, md=5):
    chosen=[]
    for _ in range(300):
        if len(chosen)==n+1: break
        c=(int(rng.integers(0,grid)),int(rng.integers(0,grid)))
        if all(abs(c[0]-x[0])+abs(c[1]-x[1])>=md for x in chosen):
            chosen.append(c)
    while len(chosen)<n+1:
        chosen.append((int(rng.integers(0,grid)),int(rng.integers(0,grid))))
    return chosen

def train_stream(field, anchors, n_epochs, hold, gap, trace_decay, lr, gate, b_decay):
    trace = EligibilityTrace((field.cfg.rows, field.cfg.cols), decay=trace_decay)
    for _ in range(n_epochs):
        field.reset_temp(); trace.reset()
        for i, A in enumerate(anchors):
            field._clamp_positions=[]; field._clamp_values=[]
            field.inject([A], [1.0])
            for _ in range(hold):
                field.step(); trace.update(field.T)
            if i > 0:
                # STDP fires with current trace + recent activity
                stdp_gated(field, trace, lr, gate_strength=gate, b_decay=b_decay)
            for _ in range(gap):
                field.step(); trace.update(field.T)
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

def run(n, ep, td, lr, gate, b_decay, md=5):
    cfg=FieldConfig(rows=GRID,cols=GRID); d=np.zeros((len(SEEDS),n))
    for si,seed in enumerate(SEEDS):
        rng=np.random.default_rng(seed)
        anchors=make_chain(n,GRID,rng,md=md)
        f=Field(cfg,seed=seed)
        train_stream(f, anchors, ep, hold=4, gap=2, trace_decay=td,
                     lr=lr, gate=gate, b_decay=b_decay)
        for li in range(n):
            d[si,li]=probe(f,seed,anchors[li],anchors[li+1])
    return d

def main():
    print(f"{'cfg':70} | {'overall':>10} | {'clear+':>6}")
    print("-"*95)
    for n in [5, 10]:
      for ep in [80, 200]:
        for td in [0.85, 0.92]:
          for gate in [0.0, 2.0, 5.0]:
            for bd in [0.0, 0.005]:
              t0=time.time()
              d=run(n, ep, td=td, lr=0.10, gate=gate, b_decay=bd)
              lm=d.mean(0); lse=d.std(0,ddof=1)/np.sqrt(len(SEEDS))
              pos=int((lm-2*lse>0).sum())
              tag=f"n={n} ep={ep} td={td} gate={gate} bd={bd}"
              print(f"{tag:70} | {d.mean():+.4f}    | {pos:>3}/{n}  ({time.time()-t0:.1f}s)", flush=True)

if __name__ == "__main__":
    main()
