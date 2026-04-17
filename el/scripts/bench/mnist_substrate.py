"""MNIST classification on the thermofield substrate as
prototype-based associative memory. Honest comparison vs MLP.

Setup: grid = 28×28, binarize MNIST at threshold 0.5. Store M
prototypes per class (mean image of M training samples), recall
closest pattern, label = stored class.
"""
import sys, time, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig
from el.thermofield.pattern_memory import PatternMemory

GRID = 28
THRESH = 0.5

def load_mnist():
    d = np.load('/tmp/mnist_cache/mnist_arr.npz')
    X = d['X'].astype(np.float32) / 255.0
    y = d['y'].astype(np.int32)
    # Standard MNIST split: first 60k train, last 10k test
    return X[:60000], y[:60000], X[60000:], y[60000:]

def img_to_pattern(img, thresh=THRESH):
    """28x28 image -> list of (r,c) where pixel > thresh."""
    B = (img.reshape(GRID, GRID) > thresh)
    rs, cs = np.where(B)
    return [(int(r), int(c)) for r, c in zip(rs, cs)]

def make_prototype(imgs):
    """Mean of M binarized images, then re-threshold to get a pattern."""
    mean = (imgs > THRESH).mean(axis=0).reshape(GRID, GRID)
    # Take top ~k pixels by activation as the prototype
    k = max(20, int(mean.sum() * 0.6))  # tighter prototype
    flat = mean.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    return [(int(i // GRID), int(i % GRID)) for i in idx], k

def make_substrate_classifier(X_tr, y_tr, M_per_class=1, write_lr=0.07,
                               write_steps=15, write_decay=0.005,
                               wta_k_factor=2.0, seed=0):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    # Stack all class prototypes' k to set wta_k
    prototypes = []; pat_sizes = []; labels = []
    for cls in range(10):
        idx = np.where(y_tr == cls)[0]
        for m in range(M_per_class):
            chunk = idx[m*50:(m+1)*50]   # 50 imgs per prototype
            if len(chunk) == 0: chunk = idx[:50]
            proto, k = make_prototype(X_tr[chunk])
            prototypes.append(proto); pat_sizes.append(k); labels.append(cls)
    wta_k = max(int(np.mean(pat_sizes) * wta_k_factor), 50)
    mem = PatternMemory(cfg=cfg, seed=seed,
                        write_steps=write_steps, write_lr=write_lr,
                        write_decay=write_decay,
                        wta_k=wta_k, wta_suppression=0.3, rule="hebb")
    for p in prototypes:
        mem.store(p)
    return mem, np.array(labels)

def predict(mem, labels, img):
    cue = img_to_pattern(img)
    if not cue: return 0
    best_i, _, _ = mem.recall(cue)
    return int(labels[best_i])

def main():
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"MNIST: train={X_tr.shape} test={X_te.shape}")
    # Test different M per class
    for M in [1, 5, 10]:
        t0 = time.time()
        mem, labels = make_substrate_classifier(X_tr, y_tr, M_per_class=M)
        build_t = time.time() - t0
        # Evaluate on 1000 random test images (full would be slow)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_te), 1000, replace=False)
        preds = []
        t0 = time.time()
        for i in idx:
            preds.append(predict(mem, labels, X_te[i]))
        pred_t = time.time() - t0
        preds = np.array(preds); truth = y_te[idx]
        acc = (preds == truth).mean()
        chance = 0.10
        print(f"M={M:2d}/class  N_proto={len(labels):3d}  "
              f"acc={acc:.3f} (chance=0.10)  "
              f"build={build_t:.1f}s  predict={pred_t:.1f}s "
              f"({pred_t/1000*1000:.1f}ms/img)")

if __name__ == "__main__":
    main()
