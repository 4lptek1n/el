"""Tiny MLP MNIST baseline (1 hidden layer, no fancy tricks)."""
import sys, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)

def load_mnist():
    d = np.load('/tmp/mnist_cache/mnist_arr.npz')
    return (d['X'][:60000].astype(np.float32)/255.0, d['y'][:60000].astype(np.int64),
            d['X'][60000:].astype(np.float32)/255.0, d['y'][60000:].astype(np.int64))

class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden), nn.ReLU(),
            nn.Linear(hidden, 10))
    def forward(self, x): return self.net(x)

def main():
    X_tr, y_tr, X_te, y_te = load_mnist()
    Xtr = torch.from_numpy(X_tr); ytr = torch.from_numpy(y_tr)
    Xte = torch.from_numpy(X_te); yte = torch.from_numpy(y_te)
    for hidden in [16, 32, 128]:
        torch.manual_seed(0)
        net = MLP(hidden=hidden); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        bs = 128; n_ep = 3
        t0 = time.time()
        for ep in range(n_ep):
            perm = torch.randperm(60000)
            for i in range(0, 60000, bs):
                b = perm[i:i+bs]
                logits = net(Xtr[b]); loss = loss_fn(logits, ytr[b])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (net(Xte).argmax(1) == yte).float().mean().item()
        n_params = sum(p.numel() for p in net.parameters())
        print(f"MLP hidden={hidden:3d}  params={n_params:6d}  "
              f"acc={acc:.3f}  train={time.time()-t0:.1f}s ({n_ep} epochs)")

if __name__ == "__main__":
    main()
