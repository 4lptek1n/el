"""Continual MNIST using SUBSTRATE FEATURES + linear readout.
Substrate stays frozen across tasks; only readout grows.
Compare to MLP-naive on raw pixels."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.plasticity import hebbian_update
torch.set_num_threads(2)

GRID=28; THRESH=0.4

def load():
    d=np.load('/tmp/mnist_cache/mnist_arr.npz')
    return (d['X'][:60000].astype(np.float32)/255.0, d['y'][:60000].astype(np.int32),
            d['X'][60000:].astype(np.float32)/255.0, d['y'][60000:].astype(np.int32))

def featurize(field, img, snaps=(2,5,8)):
    field.reset_temp()
    B=img.reshape(GRID,GRID)>THRESH; rs,cs=np.where(B)
    pos=list(zip([int(r) for r in rs],[int(c) for c in cs]))
    if not pos: return np.zeros(GRID*GRID*len(snaps), dtype=np.float32)
    field.inject(pos,[1.0]*len(pos), clamp=False)
    out=[]; mx=max(snaps)
    for t in range(mx+1):
        if t in snaps: out.append(field.T.flatten().copy())
        if t<mx: field.step()
    return np.concatenate(out)

def make_field():
    cfg=FieldConfig(rows=GRID, cols=GRID)
    f=Field(cfg, seed=0)
    return f

def hebb_pretrain(field, X, n_imgs=500, lr=0.02):
    rng=np.random.default_rng(0)
    for i in rng.choice(len(X), n_imgs, replace=False):
        B=X[i].reshape(GRID,GRID)>THRESH; rs,cs=np.where(B)
        pos=list(zip([int(r) for r in rs],[int(c) for c in cs]))
        if not pos: continue
        field.reset_temp()
        field.inject(pos,[1.0]*len(pos), clamp=False)
        for _ in range(4): field.step()
        hebbian_update(field, lr=lr, decay=0.005)
    field.reset_temp()

class LinReadout(nn.Module):
    def __init__(self, dim): super().__init__(); self.lin=nn.Linear(dim,10)
    def forward(self,x): return self.lin(x)

def eval_class_subset(net, F, y, classes_seen):
    mask=np.isin(y, classes_seen)
    if mask.sum()==0: return 0.0
    with torch.no_grad():
        return (net(torch.from_numpy(F[mask]).float()).argmax(1).numpy()==y[mask]).mean()

def main():
    X_tr,y_tr,X_te,y_te = load()
    pairs=[(0,1),(2,3),(4,5),(6,7),(8,9)]
    print("Continual MNIST: substrate-features (frozen) vs MLP-naive (raw)")
    print(f"{'after':10} | {'classes':12} | {'sub_feat':>8} | {'mlp_naive':>9}")
    print("-"*55)
    field=make_field()
    hebb_pretrain(field, X_tr, n_imgs=800)
    # Pre-featurize all test images once (frozen substrate)
    Fte = np.stack([featurize(field, X_te[i]) for i in range(2000)])
    yte_sub = y_te[:2000]
    feat_dim=Fte.shape[1]
    # Two readouts: ours (incremental) and a fresh MLP-naive on raw pixels
    torch.manual_seed(0)
    readout = LinReadout(feat_dim)
    opt = torch.optim.Adam(readout.parameters(), lr=2e-3, weight_decay=1e-4)
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(784,64), nn.ReLU(), nn.Linear(64,10))
    opt_m = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    seen=[]
    for ti, pair in enumerate(pairs):
        # Train both on this pair only (no replay, no EWC = naive)
        idx = np.where(np.isin(y_tr, pair))[0][:1000]
        # Featurize this pair's training images
        Ftr_pair = np.stack([featurize(field, X_tr[i]) for i in idx])
        Xt = torch.from_numpy(Ftr_pair).float()
        yt = torch.from_numpy(y_tr[idx].astype(np.int64))
        for ep in range(3):
            perm=torch.randperm(len(Xt))
            for i in range(0,len(Xt),64):
                bi=perm[i:i+64]
                loss=loss_fn(readout(Xt[bi]), yt[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        # MLP on raw pixels, naive
        Xtr=torch.from_numpy(X_tr[idx]); ytr=torch.from_numpy(y_tr[idx].astype(np.int64))
        for ep in range(3):
            perm=torch.randperm(len(Xtr))
            for i in range(0,len(Xtr),64):
                bi=perm[i:i+64]
                loss=loss_fn(mlp(Xtr[bi]), ytr[bi])
                opt_m.zero_grad(); loss.backward(); opt_m.step()
        seen.extend(pair)
        # Eval on accumulated classes seen so far
        s_acc = eval_class_subset(readout, Fte, yte_sub, seen)
        with torch.no_grad():
            mask=np.isin(yte_sub, seen)
            m_acc = (mlp(torch.from_numpy(X_te[:2000][mask])).argmax(1).numpy()==yte_sub[mask]).mean()
        print(f"task{ti+1:2d}    | {str(seen):12} | {s_acc:>8.3f} | {m_acc:>9.3f}", flush=True)

if __name__ == "__main__":
    main()
