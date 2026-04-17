"""Continual MNIST v2 — substrate gets MULTIPLE prototypes per class
(K-subprototypes via simple K-means-lite on training images).
"""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "src")
from el.thermofield import FieldConfig
from el.thermofield.pattern_memory import PatternMemory
torch.set_num_threads(2)

GRID=28; THRESH=0.4; K=10  # K subprototypes per class

def load():
    d = np.load('/tmp/mnist_cache/mnist_arr.npz')
    return (d['X'][:60000].astype(np.float32)/255.0, d['y'][:60000].astype(np.int32),
            d['X'][60000:].astype(np.float32)/255.0, d['y'][60000:].astype(np.int32))

def img_pat(img):
    B = (img.reshape(GRID, GRID) > THRESH); rs, cs = np.where(B)
    return [(int(r), int(c)) for r, c in zip(rs, cs)]

def kmeans_protos(X_cls, K=K, n_iters=5, k_pixels=60):
    """Quick K-means on binarized images, return K prototype patterns."""
    Xb = (X_cls > THRESH).astype(np.float32)  # (N, 784)
    rng = np.random.default_rng(0)
    init = rng.choice(len(Xb), K, replace=False)
    centers = Xb[init].copy()
    for _ in range(n_iters):
        d = ((Xb[:,None,:] - centers[None,:,:])**2).sum(-1)
        a = d.argmin(1)
        for k in range(K):
            sel = Xb[a==k]
            if len(sel)>0: centers[k] = sel.mean(0)
    protos = []
    for c in centers:
        flat = c.ravel()
        idx = np.argpartition(flat, -k_pixels)[-k_pixels:]
        protos.append([(int(i // GRID), int(i % GRID)) for i in idx])
    return protos

def make_substrate():
    cfg = FieldConfig(rows=GRID, cols=GRID)
    return PatternMemory(cfg=cfg, seed=0, write_steps=15, write_lr=0.07,
                         write_decay=0.005, recall_steps=10,
                         wta_k=120, wta_suppression=0.3, rule="hebb")

def eval_sub(mem, labels, X_te, y_te, classes_seen, n=300):
    mask = np.isin(y_te, classes_seen)
    Xt = X_te[mask]; yt = y_te[mask]
    rng = np.random.default_rng(123)
    if len(Xt)>n:
        idx=rng.choice(len(Xt),n,replace=False); Xt=Xt[idx]; yt=yt[idx]
    correct=0
    for i in range(len(Xt)):
        cue = img_pat(Xt[i])
        if not cue: continue
        bi,_,_=mem.recall(cue)
        if labels[bi]==yt[i]: correct+=1
    return correct/len(Xt)

class MLP(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784,h), nn.ReLU(), nn.Linear(h,10))
    def forward(self,x): return self.net(x)

def eval_mlp(net, X_te, y_te, classes_seen, n=300):
    mask = np.isin(y_te, classes_seen); Xt=X_te[mask]; yt=y_te[mask]
    rng=np.random.default_rng(123)
    if len(Xt)>n:
        idx=rng.choice(len(Xt),n,replace=False); Xt=Xt[idx]; yt=yt[idx]
    with torch.no_grad():
        return (net(torch.from_numpy(Xt)).argmax(1).numpy()==yt).mean()

def run():
    X_tr, y_tr, X_te, y_te = load()
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    print(f"Continual MNIST v2 — substrate uses K={K} subprototypes/class")
    print(f"{'after':10} | {'classes':12} | {'sub':>6} | {'mlp_n':>6} | {'mlp_r':>6}")
    print("-"*55)

    mem = make_substrate(); sub_labels=[]; seen=[]
    torch.manual_seed(0)
    net = MLP(); opt = torch.optim.Adam(net.parameters(),lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    net_r = MLP(); opt_r = torch.optim.Adam(net_r.parameters(),lr=1e-3)
    rep_X=[]; rep_y=[]

    for ti, pair in enumerate(pairs):
        for cls in pair:
            cidx = np.where(y_tr==cls)[0][:300]
            for proto in kmeans_protos(X_tr[cidx]):
                mem.store(proto); sub_labels.append(cls)
            seen.append(cls)
        sub_labels_arr = np.array(sub_labels)
        # MLP naive
        idx = np.where(np.isin(y_tr, pair))[0][:2000]
        Xb = torch.from_numpy(X_tr[idx]); yb=torch.from_numpy(y_tr[idx].astype(np.int64))
        for ep in range(2):
            perm=torch.randperm(len(Xb))
            for i in range(0,len(Xb),64):
                bi=perm[i:i+64]
                logits=net(Xb[bi]); loss=loss_fn(logits,yb[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        # MLP replay
        for cls in pair:
            cidx = np.where(y_tr==cls)[0][:200]
            rep_X.append(X_tr[cidx]); rep_y.append(y_tr[cidx])
        Xa = torch.from_numpy(np.concatenate(rep_X))
        ya = torch.from_numpy(np.concatenate(rep_y).astype(np.int64))
        for ep in range(2):
            perm=torch.randperm(len(Xa))
            for i in range(0,len(Xa),64):
                bi=perm[i:i+64]
                logits=net_r(Xa[bi]); loss=loss_fn(logits,ya[bi])
                opt_r.zero_grad(); loss.backward(); opt_r.step()
        s = eval_sub(mem, sub_labels_arr, X_te, y_te, seen)
        m = eval_mlp(net, X_te, y_te, seen)
        r = eval_mlp(net_r, X_te, y_te, seen)
        print(f"task{ti+1:2d}    | {str(seen):12} | {s:>6.3f} | {m:>6.3f} | {r:>6.3f}", flush=True)

if __name__ == "__main__":
    run()
