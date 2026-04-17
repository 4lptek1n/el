"""Class-incremental MNIST: see classes 0-1, then 2-3, ... measure
forgetting on previous classes. Compare substrate (no forgetting by
construction) vs naive MLP (catastrophic forgetting expected)."""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "src")
torch.set_num_threads(2)
from el.thermofield import FieldConfig
from el.thermofield.pattern_memory import PatternMemory

GRID=28; THRESH=0.4

def load():
    d = np.load('/tmp/mnist_cache/mnist_arr.npz')
    return (d['X'][:60000].astype(np.float32)/255.0, d['y'][:60000].astype(np.int32),
            d['X'][60000:].astype(np.float32)/255.0, d['y'][60000:].astype(np.int32))

def discriminative_proto(X_tr, y_tr, cls, n_imgs=200, k=60):
    in_cls = X_tr[y_tr == cls][:n_imgs] > THRESH
    out_cls = X_tr[y_tr != cls][:n_imgs*2] > THRESH
    score = in_cls.mean(0) - 0.6*out_cls.mean(0)
    flat = score.ravel(); idx = np.argpartition(flat, -k)[-k:]
    return [(int(i // GRID), int(i % GRID)) for i in idx]

def img_pat(img):
    B = (img.reshape(GRID, GRID) > THRESH); rs, cs = np.where(B)
    return [(int(r), int(c)) for r, c in zip(rs, cs)]

def make_substrate():
    cfg = FieldConfig(rows=GRID, cols=GRID)
    return PatternMemory(cfg=cfg, seed=0, write_steps=15, write_lr=0.07,
                         write_decay=0.005, recall_steps=10,
                         wta_k=120, wta_suppression=0.3, rule="hebb")

def eval_substrate(mem, labels, X_te, y_te, classes_seen, n=300):
    """Eval only on test images whose label is in classes_seen."""
    mask = np.isin(y_te, classes_seen)
    Xt = X_te[mask]; yt = y_te[mask]
    rng = np.random.default_rng(123)
    if len(Xt) > n:
        idx = rng.choice(len(Xt), n, replace=False)
        Xt = Xt[idx]; yt = yt[idx]
    correct = 0
    for i in range(len(Xt)):
        cue = img_pat(Xt[i])
        if not cue: continue
        bi,_,_ = mem.recall(cue)
        if labels[bi] == yt[i]: correct += 1
    return correct / len(Xt)

class MLP(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784,h), nn.ReLU(), nn.Linear(h,10))
    def forward(self,x): return self.net(x)

def eval_mlp(net, X_te, y_te, classes_seen, n=300):
    mask = np.isin(y_te, classes_seen)
    Xt = X_te[mask]; yt = y_te[mask]
    rng = np.random.default_rng(123)
    if len(Xt) > n:
        idx = rng.choice(len(Xt), n, replace=False)
        Xt = Xt[idx]; yt = yt[idx]
    with torch.no_grad():
        logits = net(torch.from_numpy(Xt))
    pred = logits.argmax(1).numpy()
    return (pred == yt).mean()

def run():
    X_tr, y_tr, X_te, y_te = load()
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    print("Class-incremental MNIST: train on (0,1)→(2,3)→(4,5)→(6,7)→(8,9)")
    print(f"{'after task':12} | {'classes':10} | {'substrate':>10} | "
          f"{'mlp_naive':>10} | {'mlp_replay':>10}")
    print("-"*70)

    # Substrate: store class prototypes incrementally
    mem = make_substrate(); sub_labels = []
    seen = []

    # MLP naive
    torch.manual_seed(0)
    net = MLP(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # MLP with replay (small buffer)
    torch.manual_seed(0)
    net_r = MLP(); opt_r = torch.optim.Adam(net_r.parameters(), lr=1e-3)
    replay_buf_X = []; replay_buf_y = []  # 200 per class

    for ti, pair in enumerate(pairs):
        # Substrate: store prototype for each new class
        for cls in pair:
            proto = discriminative_proto(X_tr, y_tr, cls)
            mem.store(proto); sub_labels.append(cls)
            seen.append(cls)
        sub_labels_arr = np.array(sub_labels)

        # MLP naive: train only on this pair
        idx = np.where(np.isin(y_tr, pair))[0][:2000]
        Xb = torch.from_numpy(X_tr[idx]); yb = torch.from_numpy(y_tr[idx].astype(np.int64))
        for ep in range(2):
            perm = torch.randperm(len(Xb))
            for i in range(0, len(Xb), 64):
                bi = perm[i:i+64]
                logits = net(Xb[bi]); loss = loss_fn(logits, yb[bi])
                opt.zero_grad(); loss.backward(); opt.step()

        # MLP with replay: add 200 of new-class to buffer, train on buffer + new
        for cls in pair:
            cidx = np.where(y_tr == cls)[0][:200]
            replay_buf_X.append(X_tr[cidx]); replay_buf_y.append(y_tr[cidx])
        Xall = np.concatenate(replay_buf_X)
        yall = np.concatenate(replay_buf_y).astype(np.int64)
        Xa = torch.from_numpy(Xall); ya = torch.from_numpy(yall)
        for ep in range(2):
            perm = torch.randperm(len(Xa))
            for i in range(0, len(Xa), 64):
                bi = perm[i:i+64]
                logits = net_r(Xa[bi]); loss = loss_fn(logits, ya[bi])
                opt_r.zero_grad(); loss.backward(); opt_r.step()

        s_acc = eval_substrate(mem, sub_labels_arr, X_te, y_te, seen)
        m_acc = eval_mlp(net, X_te, y_te, seen)
        r_acc = eval_mlp(net_r, X_te, y_te, seen)
        print(f"task{ti+1:2d} done   | {str(seen):10} | "
              f"{s_acc:>10.3f} | {m_acc:>10.3f} | {r_acc:>10.3f}", flush=True)

if __name__ == "__main__":
    run()
