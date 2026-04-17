"""Substrate as FEATURE EXTRACTOR + linear readout. Hypothesis:
PatternMemory kötü classifier ama field state iyi feature olabilir.
Inject MNIST image -> let substrate relax -> read T as feature vector
-> train LogisticRegression on top.
"""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.plasticity import hebbian_update
torch.set_num_threads(2)

GRID=28; THRESH=0.4

def load():
    d = np.load('/tmp/mnist_cache/mnist_arr.npz')
    return (d['X'][:60000].astype(np.float32)/255.0, d['y'][:60000].astype(np.int32),
            d['X'][60000:].astype(np.float32)/255.0, d['y'][60000:].astype(np.int32))

def featurize(field, img, n_steps=8, hebb_pretrain=False):
    """Inject pattern (binarized image), relax, return T as feature vec."""
    field.reset_temp()
    B = img.reshape(GRID, GRID) > THRESH
    rs, cs = np.where(B)
    pos = list(zip([int(r) for r in rs], [int(c) for c in cs]))
    if not pos:
        return field.T.flatten().copy()
    field.inject(pos, [1.0]*len(pos), clamp=False)
    for _ in range(n_steps):
        field.step()
    return field.T.flatten().copy()

def main():
    X_tr, y_tr, X_te, y_te = load()
    N_TR=3000; N_TE=1000
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(X_tr), N_TR, replace=False)
    te_idx = rng.choice(len(X_te), N_TE, replace=False)
    cfg = FieldConfig(rows=GRID, cols=GRID)

    # Try: 1) raw field with random init, 2) Hebb-pretrained on training data
    for label, hebb_pre in [("raw", False), ("hebb_pre", True)]:
        for n_steps in [4, 8, 16]:
            field = Field(cfg, seed=0)
            if hebb_pre:
                # Light Hebb pre-training on 200 random training imgs
                for i in tr_idx[:200]:
                    B = X_tr[i].reshape(GRID,GRID) > THRESH
                    rs, cs = np.where(B)
                    pos = list(zip([int(r) for r in rs], [int(c) for c in cs]))
                    if not pos: continue
                    field.reset_temp()
                    field.inject(pos, [1.0]*len(pos), clamp=False)
                    for _ in range(4):
                        field.step()
                    hebbian_update(field, lr=0.02, decay=0.001)
                field.reset_temp()
            t0 = time.time()
            Ftr = np.stack([featurize(field, X_tr[i], n_steps=n_steps) for i in tr_idx])
            Fte = np.stack([featurize(field, X_te[i], n_steps=n_steps) for i in te_idx])
            ft = time.time() - t0
            # Train logistic regression on features
            net = nn.Linear(GRID*GRID, 10)
            opt = torch.optim.Adam(net.parameters(), lr=5e-3)
            Xt = torch.from_numpy(Ftr).float(); yt = torch.from_numpy(y_tr[tr_idx].astype(np.int64))
            Xv = torch.from_numpy(Fte).float(); yv = torch.from_numpy(y_te[te_idx].astype(np.int64))
            for ep in range(80):
                opt.zero_grad()
                loss = nn.functional.cross_entropy(net(Xt), yt)
                loss.backward(); opt.step()
            with torch.no_grad():
                acc = (net(Xv).argmax(1) == yv).float().mean().item()
            print(f"{label:9} steps={n_steps:2d} | feat_acc={acc:.3f}  feat_time={ft:.1f}s")

    # Reference: linear directly on pixels
    Xt = torch.from_numpy(X_tr[tr_idx]).float()
    yt = torch.from_numpy(y_tr[tr_idx].astype(np.int64))
    Xv = torch.from_numpy(X_te[te_idx]).float()
    yv = torch.from_numpy(y_te[te_idx].astype(np.int64))
    net = nn.Linear(784, 10); opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for ep in range(80):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(Xt), yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = (net(Xv).argmax(1) == yv).float().mean().item()
    print(f"[ref] linear on raw pixels: {acc:.3f}")

if __name__ == "__main__":
    main()
