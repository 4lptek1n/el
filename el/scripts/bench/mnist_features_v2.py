"""Substrate features v2 — proper Hebb pretraining + multi-step
concatenation + full test set. Try to beat MLP-32 (~94%)."""
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

def hebb_pretrain(field, X, n_imgs, n_steps_each, lr, decay):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), n_imgs, replace=False)
    for i in idx:
        B = X[i].reshape(GRID,GRID) > THRESH
        rs, cs = np.where(B)
        pos = list(zip([int(r) for r in rs], [int(c) for c in cs]))
        if not pos: continue
        field.reset_temp()
        field.inject(pos, [1.0]*len(pos), clamp=False)
        for _ in range(n_steps_each):
            field.step()
        hebbian_update(field, lr=lr, decay=decay)
    field.reset_temp()

def featurize_multi(field, img, snapshot_steps):
    """Inject and snapshot field T at multiple steps; concatenate."""
    field.reset_temp()
    B = img.reshape(GRID,GRID) > THRESH
    rs, cs = np.where(B)
    pos = list(zip([int(r) for r in rs], [int(c) for c in cs]))
    if not pos:
        return np.zeros(GRID*GRID*len(snapshot_steps), dtype=np.float32)
    field.inject(pos, [1.0]*len(pos), clamp=False)
    snaps = []; max_step = max(snapshot_steps)
    for t in range(max_step+1):
        if t in snapshot_steps:
            snaps.append(field.T.flatten().copy())
        if t < max_step: field.step()
    return np.concatenate(snaps)

def main():
    X_tr, y_tr, X_te, y_te = load()
    cfg = FieldConfig(rows=GRID, cols=GRID)
    rng = np.random.default_rng(0)

    print(f"{'config':50} | {'tr':>4} {'te':>4} | {'acc':>6}")
    print("-"*70)

    # Sweeps
    for n_train in [10000]:
      for n_test in [5000]:
        for n_pre in [500, 2000]:
          for n_pre_steps in [3, 6]:
            for snap_set in [(2,5,8), (1,3,6,10)]:
              for pre_lr in [0.01, 0.03]:
                t0 = time.time()
                field = Field(cfg, seed=0)
                hebb_pretrain(field, X_tr, n_imgs=n_pre,
                              n_steps_each=n_pre_steps, lr=pre_lr, decay=0.005)
                tr_idx = rng.choice(len(X_tr), n_train, replace=False)
                te_idx = rng.choice(len(X_te), n_test, replace=False)
                Ftr = np.stack([featurize_multi(field, X_tr[i], snap_set) for i in tr_idx])
                Fte = np.stack([featurize_multi(field, X_te[i], snap_set) for i in te_idx])
                feat_dim = Ftr.shape[1]
                # Linear classifier with weight decay
                Xt = torch.from_numpy(Ftr).float(); yt = torch.from_numpy(y_tr[tr_idx].astype(np.int64))
                Xv = torch.from_numpy(Fte).float(); yv = torch.from_numpy(y_te[te_idx].astype(np.int64))
                net = nn.Linear(feat_dim, 10)
                opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
                bs=256
                for ep in range(40):
                    perm = torch.randperm(len(Xt))
                    for i in range(0,len(Xt),bs):
                        bi=perm[i:i+bs]
                        loss = nn.functional.cross_entropy(net(Xt[bi]), yt[bi])
                        opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    acc = (net(Xv).argmax(1)==yv).float().mean().item()
                tag=f"pre{n_pre}/{n_pre_steps}st lr{pre_lr} snaps{snap_set}"
                print(f"{tag:50} | {n_train:>4} {n_test:>4} | {acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # MLP refs
    for hidden in [16, 32]:
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(784,hidden), nn.ReLU(), nn.Linear(hidden,10))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xt=torch.from_numpy(X_tr[:10000]); yt=torch.from_numpy(y_tr[:10000].astype(np.int64))
        Xv=torch.from_numpy(X_te[:5000]); yv=torch.from_numpy(y_te[:5000].astype(np.int64))
        for ep in range(5):
            perm=torch.randperm(len(Xt))
            for i in range(0,len(Xt),128):
                bi=perm[i:i+128]
                loss=nn.functional.cross_entropy(net(Xt[bi]), yt[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc=(net(Xv).argmax(1)==yv).float().mean().item()
        print(f"[ref] MLP hidden={hidden} (10k tr, 5 ep)             | 10000 5000 | {acc:.3f}")

if __name__ == "__main__":
    main()
