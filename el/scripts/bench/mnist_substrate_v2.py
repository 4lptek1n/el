"""MNIST substrate v2 — discriminative prototypes + WTA/recall tune."""
import sys, time, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig
from el.thermofield.pattern_memory import PatternMemory

GRID = 28; THRESH = 0.4

def load():
    d = np.load('/tmp/mnist_cache/mnist_arr.npz')
    return (d['X'][:60000].astype(np.float32)/255.0, d['y'][:60000].astype(np.int32),
            d['X'][60000:].astype(np.float32)/255.0, d['y'][60000:].astype(np.int32))

def discriminative_proto(X_tr, y_tr, cls, n_imgs=200, k=None):
    """Pixels that are ON for THIS class but typically OFF for others.
    Sharper signature than raw class mean.
    """
    in_cls = X_tr[y_tr == cls][:n_imgs] > THRESH
    out_cls = X_tr[y_tr != cls][:n_imgs*2] > THRESH
    score = in_cls.mean(0) - 0.6 * out_cls.mean(0)
    if k is None: k = 60
    flat = score.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    return [(int(i // GRID), int(i % GRID)) for i in idx]

def img_pat(img):
    B = (img.reshape(GRID, GRID) > THRESH); rs, cs = np.where(B)
    return [(int(r), int(c)) for r, c in zip(rs, cs)]

def main():
    X_tr, y_tr, X_te, y_te = load()
    rng = np.random.default_rng(42)
    test_idx = rng.choice(len(X_te), 1000, replace=False)
    print(f"{'cfg':50} | {'acc':>6}")
    print("-"*65)
    for k_proto in [40, 60, 100]:
      for wta_mult in [1.5, 2.5, 4.0]:
        for recall_steps in [10, 30]:
          for write_decay in [0.005, 0.02]:
            cfg = FieldConfig(rows=GRID, cols=GRID)
            wta = max(int(k_proto*wta_mult), 50)
            mem = PatternMemory(cfg=cfg, seed=0,
                write_steps=15, write_lr=0.07, write_decay=write_decay,
                recall_steps=recall_steps,
                wta_k=wta, wta_suppression=0.3, rule="hebb")
            labels=[]
            for cls in range(10):
                proto = discriminative_proto(X_tr, y_tr, cls, k=k_proto)
                mem.store(proto); labels.append(cls)
            labels = np.array(labels)
            preds=[]
            for i in test_idx:
                cue = img_pat(X_te[i])
                if not cue: preds.append(0); continue
                bi,_,_ = mem.recall(cue); preds.append(int(labels[bi]))
            acc = (np.array(preds)==y_te[test_idx]).mean()
            tag = f"k={k_proto} wta={wta} steps={recall_steps} dec={write_decay}"
            print(f"{tag:50} | {acc:.3f}", flush=True)

if __name__ == "__main__":
    main()
