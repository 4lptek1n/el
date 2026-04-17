"""Eşik 3' — channel separation via random edge mask.
PatternMemory only writes Hebb to edges where mask=True; sequence STDP
only writes to edges where mask=False. Both share C/B but on disjoint
edge sets → no overwrite interference."""
import sys, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.pattern_memory import PatternMemory, random_pattern
from el.thermofield.sequence import (
    EligibilityTrace, present_event, relax_with_trace,
)
from el.thermofield.plasticity import hebbian_update
from el.thermofield.inhibition import kwta_step

GRID=14; N_PAT=8; SEEDS=list(range(8)); N_TRIALS=15
A_POS=(12,1); B_POS=(12,5); SEQ_EPOCHS=30

def noisy(p,d,rng,nc,co):
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
        cue=noisy(patterns[i],d,rng,nc,cfg.cols)
        b,_,_=mem.recall(cue)
        if b==i: c+=1
    return c/nt

def store_masked(mem, pattern, mask_h, mask_v):
    """Pattern Hebb only on mask=True edges."""
    f = mem.field
    f.reset_temp()
    f.inject(list(pattern),[1.0]*len(pattern))
    for _ in range(mem.write_steps):
        f.step()
        if mem.wta_k>0:
            kwta_step(f.T.reshape(-1), mem.wta_k, mem.wta_suppression)
        T = f.T
        co_h = T[:, :-1]*T[:, 1:]
        co_v = T[:-1, :]*T[1:, :]
        f.C_right += mem.write_lr * co_h * mask_h - mem.write_decay * f.C_right
        f.C_down  += mem.write_lr * co_v * mask_v - mem.write_decay * f.C_down
        np.clip(f.C_right, 0.05, 1.0, out=f.C_right)
        np.clip(f.C_down,  0.05, 1.0, out=f.C_down)
    f.reset_temp()
    mem.patterns.append(list(pattern))

def stdp_masked(field, trace, lr, mask_h, mask_v):
    T=field.T; E=trace.E
    co_h = T[:, 1:] * E[:, :-1]
    co_v = T[1:, :] * E[:-1, :]
    field.B_right += lr * co_h * mask_h
    field.B_down  += lr * co_v * mask_v
    np.clip(field.B_right, -1.0, 1.0, out=field.B_right)
    np.clip(field.B_down,  -1.0, 1.0, out=field.B_down)

def seq_train_masked(field, mask_h_seq, mask_v_seq, epochs=SEQ_EPOCHS):
    trace=EligibilityTrace((field.cfg.rows,field.cfg.cols),decay=0.80)
    for _ in range(epochs):
        field.reset_temp(); trace.reset()
        present_event(field,trace,[A_POS],[1.0],hold=5)
        relax_with_trace(field,trace,2)
        field._clamp_positions=[]; field._clamp_values=[]
        field.inject([B_POS],[1.0])
        stdp_masked(field, trace, lr=0.07, mask_h=mask_h_seq, mask_v=mask_v_seq)
        for _ in range(5):
            field.step(); trace.update(field.T)
        # No global hebb — that would update C and break separation
    field.reset_temp()

def seq_probe(field, seed):
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

def make_masks(cfg, seq_frac, seed):
    """Random partition of edges. seq_frac fraction goes to sequence."""
    rng=np.random.default_rng(seed*101)
    mh = (rng.random((cfg.rows, cfg.cols-1)) > seq_frac).astype(np.float32)  # 1=pattern
    mv = (rng.random((cfg.rows-1, cfg.cols)) > seq_frac).astype(np.float32)
    seq_h = 1.0 - mh; seq_v = 1.0 - mv
    return mh, mv, seq_h, seq_v

def make_mem(seed, cfg, decay=0.005):
    pat=max(4,int(0.05*GRID*GRID)); k=max(pat+2,int(0.075*GRID*GRID))
    return PatternMemory(cfg=cfg, seed=seed, write_steps=15, write_lr=0.07,
                         write_decay=decay,
                         wta_k=k, wta_suppression=0.3, rule="hebb"), pat

def run(condition, seq_frac):
    cfg=FieldConfig(rows=GRID,cols=GRID); pa_l=[]; sd_l=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        mem, ps=make_mem(seed, cfg)
        patterns=[random_pattern(GRID,GRID,k=ps,rng=rng) for _ in range(N_PAT)]
        f=mem.field
        mh, mv, sh, sv = make_masks(cfg, seq_frac, seed)
        if condition=="pattern_only":
            for p in patterns: store_masked(mem, p, mh, mv)
        elif condition=="seq_only":
            seq_train_masked(f, sh, sv)
        elif condition=="multi_seq_then_pat":
            seq_train_masked(f, sh, sv)
            for p in patterns: store_masked(mem, p, mh, mv)
        elif condition=="interleaved":
            for p in patterns:
                store_masked(mem, p, mh, mv)
                seq_train_masked(f, sh, sv, epochs=4)
        if mem.patterns:
            pa_l.append(pat_acc(mem,patterns,0.5,
                np.random.default_rng(seed*31+7),N_TRIALS))
        sd_l.append(seq_probe(f,seed))
    return np.array(pa_l) if pa_l else None, np.array(sd_l)

def main():
    print(f"{'cfg':40} | {'pa':>10} | {'sd':>11} | {'pa_keep':>7} | {'sq_keep':>7} | {'verdict':>7}")
    print("-"*90)
    for seq_frac in [0.3, 0.5, 0.7]:
        pa_p, _ = run("pattern_only", seq_frac)
        _, sd_s = run("seq_only", seq_frac)
        for cond in ["multi_seq_then_pat", "interleaved"]:
            pa_m, sd_m = run(cond, seq_frac)
            pk = pa_m.mean()/pa_p.mean() if pa_p.mean()>0 else 0
            sk = sd_m.mean()/sd_s.mean() if sd_s.mean()!=0 else 0
            ok = "✓" if (pk>=0.90 and sk>=0.90) else "✗"
            tag=f"seq_frac={seq_frac} cond={cond}"
            print(f"{tag:40} | pa={pa_m.mean():.3f} | sd={sd_m.mean():+.4f} | "
                  f"{pk:>5.0%} | {sk:>+5.0%} | {ok}", flush=True)

if __name__ == "__main__":
    main()
