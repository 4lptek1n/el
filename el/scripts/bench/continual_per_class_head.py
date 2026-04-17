"""Per-class readout head over substrate features.
Each class c has its own weight vector w_c trained ONLY on samples
of c (positive vs negative subsample). Argmax of w_c · phi(x) at
inference. No shared softmax → no interference.

Compare to:
- shared softmax linear readout (catastrophic forgetting)
- nearest-mean classifier on features (NMC)
"""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.plasticity import hebbian_update
torch.set_num_threads(2)
GRID=28; THRESH=0.4; SNAPS=(2,5,8)

def load():
    d=np.load('/tmp/mnist_cache/mnist_arr.npz')
    return (d['X'][:60000].astype(np.float32)/255.0, d['y'][:60000].astype(np.int32),
            d['X'][60000:].astype(np.float32)/255.0, d['y'][60000:].astype(np.int32))

def featurize(field, img):
    field.reset_temp()
    B=img.reshape(GRID,GRID)>THRESH; rs,cs=np.where(B)
    pos=list(zip([int(r) for r in rs],[int(c) for c in cs]))
    if not pos: return np.zeros(GRID*GRID*len(SNAPS), dtype=np.float32)
    field.inject(pos,[1.0]*len(pos), clamp=False)
    out=[]; mx=max(SNAPS)
    for t in range(mx+1):
        if t in SNAPS: out.append(field.T.flatten().copy())
        if t<mx: field.step()
    return np.concatenate(out)

def make_field():
    cfg=FieldConfig(rows=GRID,cols=GRID); f=Field(cfg,seed=0)
    return f

def hebb_pretrain(field, X, n_imgs=500):
    rng=np.random.default_rng(0)
    for i in rng.choice(len(X), n_imgs, replace=False):
        B=X[i].reshape(GRID,GRID)>THRESH; rs,cs=np.where(B)
        pos=list(zip([int(r) for r in rs],[int(c) for c in cs]))
        if not pos: continue
        field.reset_temp()
        field.inject(pos,[1.0]*len(pos), clamp=False)
        for _ in range(4): field.step()
        hebbian_update(field, lr=0.02, decay=0.005)
    field.reset_temp()

class PerClassHead:
    """Each class: linear scorer w_c·phi(x). Trained 1-vs-rest only on
    the class's introduction. Frozen after that → no interference."""
    def __init__(self, dim, n_classes=10):
        self.dim=dim; self.n=n_classes
        self.w = np.zeros((n_classes, dim), dtype=np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.trained = [False]*n_classes
    def train_class(self, F_pos, F_neg, lr=1e-2, epochs=60):
        # Pure 1-vs-rest logistic regression for one class
        for c_count, (F_p, F_n) in enumerate(zip(F_pos, F_neg)):
            if F_p is None: continue
            cls = c_count
            X = np.concatenate([F_p, F_n], axis=0)
            y = np.concatenate([np.ones(len(F_p)), np.zeros(len(F_n))]).astype(np.float32)
            Xt=torch.from_numpy(X).float(); yt=torch.from_numpy(y)
            w = torch.zeros(self.dim, requires_grad=True)
            b = torch.zeros(1, requires_grad=True)
            opt=torch.optim.Adam([w,b], lr=lr, weight_decay=1e-4)
            for _ in range(epochs):
                logits = Xt @ w + b
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yt)
                opt.zero_grad(); loss.backward(); opt.step()
            self.w[cls] = w.detach().numpy()
            self.b[cls] = float(b.detach().numpy().item())
            self.trained[cls] = True
    def predict(self, F):
        # Only consider classes that have been trained
        scores = F @ self.w.T + self.b   # (N, n_classes)
        # Mask untrained
        mask = np.array([1.0 if t else -1e9 for t in self.trained])
        scores = scores + mask
        return scores.argmax(axis=1)

def nmc(F_te, prototypes, classes_seen, dim):
    """Nearest mean classifier over substrate features."""
    means = np.stack([prototypes[c] if prototypes[c] is not None else np.zeros(dim, dtype=np.float32) for c in range(10)])
    d = ((F_te[:,None,:] - means[None,:,:])**2).sum(-1)
    mask = np.array([0.0 if c in classes_seen else 1e9 for c in range(10)])
    d = d + mask
    return d.argmin(axis=1)

def main():
    X_tr,y_tr,X_te,y_te = load()
    pairs=[(0,1),(2,3),(4,5),(6,7),(8,9)]
    field=make_field()
    hebb_pretrain(field, X_tr, n_imgs=800)
    # Pre-featurize a fixed test set
    Fte = np.stack([featurize(field, X_te[i]) for i in range(2000)])
    yte = y_te[:2000]
    feat_dim=Fte.shape[1]

    # Per-class head: train each class as it arrives
    head = PerClassHead(dim=feat_dim, n_classes=10)
    # Class prototypes for NMC
    protos = {c: None for c in range(10)}

    # Shared softmax baseline (catastrophic)
    torch.manual_seed(0)
    shared = nn.Linear(feat_dim, 10)
    opt_s = torch.optim.Adam(shared.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn=nn.CrossEntropyLoss()

    print(f"Per-class head vs NMC vs shared-softmax (substrate features)")
    print(f"{'after':10} | {'classes':12} | {'per_cls':>8} | {'nmc':>6} | {'shared':>7}")
    print("-"*60)

    seen=[]
    for ti, pair in enumerate(pairs):
        # Featurize this task's training data
        F_pos_list = [None]*10; F_neg_list = [None]*10
        for cls in pair:
            cidx = np.where(y_tr==cls)[0][:300]
            F_pos = np.stack([featurize(field, X_tr[i]) for i in cidx])
            # Negative samples: random from other already-seen-or-incoming classes
            other_pool = np.where(y_tr != cls)[0][:600]
            F_neg = np.stack([featurize(field, X_tr[i]) for i in other_pool[:300]])
            F_pos_list[cls] = F_pos
            F_neg_list[cls] = F_neg
            # Update class prototype (mean feature)
            protos[cls] = F_pos.mean(0)
            seen.append(cls)
        head.train_class(F_pos_list, F_neg_list)

        # Train shared softmax on this pair only (naive)
        idx = np.where(np.isin(y_tr, pair))[0][:600]
        F_b = np.stack([featurize(field, X_tr[i]) for i in idx])
        Xt=torch.from_numpy(F_b).float(); yt=torch.from_numpy(y_tr[idx].astype(np.int64))
        for ep in range(3):
            perm=torch.randperm(len(Xt))
            for i in range(0,len(Xt),64):
                bi=perm[i:i+64]
                loss=loss_fn(shared(Xt[bi]), yt[bi])
                opt_s.zero_grad(); loss.backward(); opt_s.step()

        # Eval all methods on accumulated classes
        mask = np.isin(yte, seen); Fts=Fte[mask]; yts=yte[mask]
        pc_pred = head.predict(Fts)
        pc_acc = (pc_pred==yts).mean()
        nmc_pred = nmc(Fts, protos, seen, feat_dim)
        nmc_acc = (nmc_pred==yts).mean()
        with torch.no_grad():
            sh_pred = shared(torch.from_numpy(Fts).float()).argmax(1).numpy()
        sh_acc = (sh_pred==yts).mean()
        print(f"task{ti+1:2d}    | {str(seen):12} | "
              f"{pc_acc:>8.3f} | {nmc_acc:>6.3f} | {sh_acc:>7.3f}", flush=True)

if __name__ == "__main__":
    main()
