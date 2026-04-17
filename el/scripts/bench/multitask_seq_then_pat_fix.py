"""Symmetric multi-task: try aggressive write_decay + lower write_lr
to fix seq_then_pat ordering (pattern Hebb after seq must not destroy
seq B-bias). Also try a "B-protect" mode where pattern store skips
the few edges with highest |B|."""
import sys, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.pattern_memory import PatternMemory, random_pattern
from el.thermofield.sequence import (
    EligibilityTrace, present_event, relax_with_trace, stdp_hebbian_update,
)
from el.thermofield.plasticity import hebbian_update
from el.thermofield.inhibition import kwta_step

GRID=14; N_PAT=8; SEEDS=list(range(8)); N_TRIALS=15
A_POS=(12,1); B_POS=(12,5); SEQ_EPOCHS=30

def noisy_cue(p,d,rng,nc,co):
    kn=max(1,int(round(len(p)*(1-d))))
    keep=[p[i] for i in sorted(rng.choice(len(p),kn,replace=False))]
    pset=set(p); ds=[]
    while len(ds)<kn:
        idx=int(rng.integers(0,nc)); rc=(idx//co,idx%co)
        if rc not in pset and rc not in ds: ds.append(rc)
    return keep+ds

def pat_acc(mem,patterns,d,rng,nt):
    cfg=mem.cfg; nc=cfg.rows*cfg.cols; c=0
    for _ in range(nt):
        i=int(rng.integers(0,len(patterns)))
        cue=noisy_cue(patterns[i],d,rng,nc,cfg.cols)
        b,_,_=mem.recall(cue)
        if b==i: c+=1
    return c/nt

def seq_train(field):
    trace=EligibilityTrace((field.cfg.rows,field.cfg.cols),decay=0.80)
    for _ in range(SEQ_EPOCHS):
        field.reset_temp(); trace.reset()
        present_event(field,trace,[A_POS],[1.0],hold=5)
        relax_with_trace(field,trace,2)
        field._clamp_positions=[]; field._clamp_values=[]
        field.inject([B_POS],[1.0])
        stdp_hebbian_update(field,trace,lr=0.07)
        for _ in range(5):
            field.step(); trace.update(field.T)
        hebbian_update(field,lr=0.07,decay=0.001)
    field.reset_temp()

def seq_probe(field,seed):
    field.reset_temp()
    tr=EligibilityTrace((field.cfg.rows,field.cfg.cols),decay=0.80)
    present_event(field,tr,[A_POS],[1.0],hold=5)
    relax_with_trace(field,tr,6)
    cue=float(field.T[B_POS])
    fresh=Field(field.cfg,seed=seed)
    ftr=EligibilityTrace((field.cfg.rows,field.cfg.cols),decay=0.80)
    present_event(fresh,ftr,[A_POS],[1.0],hold=5)
    relax_with_trace(fresh,ftr,6)
    return cue-float(fresh.T[B_POS])

def store_b_protect(mem, pattern, b_protect_frac=0.10):
    """Skip C updates on the top fraction of |B| edges (sequence-trained).
    B kanalı korunmuş ama C oraya yazma → diffusion path bozulmaz."""
    f = mem.field
    f.reset_temp()
    f.inject(list(pattern),[1.0]*len(pattern))
    # Determine protected edges ONCE based on current B
    b_h_thresh = np.quantile(np.abs(f.B_right), 1-b_protect_frac) if f.B_right.size else 0
    b_v_thresh = np.quantile(np.abs(f.B_down), 1-b_protect_frac) if f.B_down.size else 0
    h_mask = np.abs(f.B_right) < b_h_thresh   # write here
    v_mask = np.abs(f.B_down)  < b_v_thresh
    for _ in range(mem.write_steps):
        f.step()
        if mem.wta_k>0:
            kwta_step(f.T.reshape(-1), mem.wta_k, mem.wta_suppression)
        T = f.T
        co_h = T[:, :-1]*T[:, 1:]
        co_v = T[:-1, :]*T[1:, :]
        f.C_right += mem.write_lr * co_h * h_mask - mem.write_decay * f.C_right
        f.C_down  += mem.write_lr * co_v * v_mask - mem.write_decay * f.C_down
        np.clip(f.C_right, 0.05, 1.0, out=f.C_right)
        np.clip(f.C_down,  0.05, 1.0, out=f.C_down)
    f.reset_temp()
    mem.patterns.append(list(pattern))

def make_mem(seed, cfg, **kw):
    pat=max(4,int(0.05*GRID*GRID)); k=max(pat+2,int(0.075*GRID*GRID))
    base=dict(write_steps=15, write_lr=0.07, write_decay=0.005)
    base.update(kw)
    return PatternMemory(cfg=cfg,seed=seed,wta_k=k,wta_suppression=0.3,
                         rule="hebb", **base), pat

def run(condition, decay, b_protect):
    cfg=FieldConfig(rows=GRID,cols=GRID)
    pa_l=[]; sd_l=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        mem, ps=make_mem(seed, cfg, write_decay=decay)
        patterns=[random_pattern(GRID,GRID,k=ps,rng=rng) for _ in range(N_PAT)]
        f=mem.field
        def store(p):
            if b_protect>0: store_b_protect(mem, p, b_protect_frac=b_protect)
            else:           mem.store(p)
        if condition=="pattern_only":
            for p in patterns: store(p)
        elif condition=="seq_only":
            seq_train(f)
        elif condition=="multi_seq_then_pat":
            seq_train(f)
            for p in patterns: store(p)
        if mem.patterns:
            pa_l.append(pat_acc(mem,patterns,0.5,
                np.random.default_rng(seed*31+7),N_TRIALS))
        sd_l.append(seq_probe(f,seed))
    return np.array(pa_l) if pa_l else None, np.array(sd_l)

def main():
    print(f"{'cfg':30} | {'pa':>10} {'sd':>11} | "
          f"{'pa_keep':>7} {'sq_keep':>7} {'verdict':>8}")
    for decay in [0.005, 0.01, 0.02, 0.05]:
        for bp in [0.0, 0.05, 0.10, 0.20]:
            pa_a, sd_a = run("pattern_only", decay, bp)
            _,    sd_s = run("seq_only", decay, bp)
            pa_m, sd_m = run("multi_seq_then_pat", decay, bp)
            pk = pa_m.mean()/pa_a.mean() if pa_a.mean()>0 else 0
            sk = sd_m.mean()/sd_s.mean() if sd_s.mean()!=0 else 0
            ok = "✓" if (pk>=0.90 and sk>=0.90) else "✗"
            tag=f"dec={decay} bp={bp}"
            print(f"{tag:30} | pa={pa_m.mean():.3f} sd={sd_m.mean():+.4f} | "
                  f"{pk:.0%}{'':>3} {sk:+.0%}{'':>2} {ok}", flush=True)

if __name__ == "__main__":
    main()
