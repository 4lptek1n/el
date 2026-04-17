"""External benchmark war — round 1: MLP vs v7 hybrid on link discrimination.

Common task: given anchor A in a 14×14 grid, identify the true successor B
from a distractor B' (random other anchor). Both models given identical
(A, B) training pairs sampled from chain anchors.

Metric (apples-to-apples, single number):
  link_discrimination_accuracy = P(score(A,B_true) > score(A,B_distractor))

For v7:  score(A, X) = cue_T_at_X_after_propagation
For MLP: score(A, X) = predicted-class-logit at X (softmax over 196 cells)

Same anchors, same seeds, same task. Honest head-to-head.

Run: python scripts/bench/external_war_v1.py
"""
from __future__ import annotations
import sys, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from el.thermofield.field import Field, FieldConfig
import seq_chain_v7_hybrid as V7

GRID = 14
N_CELLS = GRID * GRID
SEEDS = list(range(8))


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================== MLP baseline
class MLP:
    """Minimal 1-hidden-layer MLP, trained with vanilla SGD on cross-entropy.
    Input: one-hot A (N), output: softmax over B (N)."""
    def __init__(self, n_in, n_hidden, n_out, seed=0, lr=0.05):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, (n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)
        self.lr = lr

    def forward(self, x):
        self.h_pre = x @ self.W1 + self.b1
        self.h = np.maximum(0, self.h_pre)
        self.logits = self.h @ self.W2 + self.b2
        return self.logits

    def step(self, x, y_true_idx):
        # x: (B, N_in), y_true_idx: (B,) target class
        logits = self.forward(x)
        p = softmax(logits)
        B = x.shape[0]
        dlogits = p.copy()
        dlogits[np.arange(B), y_true_idx] -= 1
        dlogits /= B
        dW2 = self.h.T @ dlogits
        db2 = dlogits.sum(0)
        dh = dlogits @ self.W2.T
        dh_pre = dh * (self.h_pre > 0)
        dW1 = x.T @ dh_pre
        db1 = dh_pre.sum(0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


def cell_to_idx(rc):
    return rc[0] * GRID + rc[1]


def collect_pairs(anchors_per_seed):
    """Each chain: yield (A, B) pairs from consecutive anchors."""
    pairs = []
    for anchors in anchors_per_seed:
        for j in range(len(anchors) - 1):
            pairs.append((anchors[j], anchors[j+1]))
    return pairs


# ============================================================== Eval
def link_discrim_v7(seed, anchors, K=4, md_min=3, ep=120, lr=0.20, eta=0.05):
    """For each link, score = cue at true B - cue at one random distractor.
    Distractor = anchor not equal to A or B from same chain."""
    tr = V7.HybridTrainer(rows=GRID, cols=GRID, seed=seed, K=K, md_min=md_min,
                          skip_eta=eta)
    V7.train_chain_hybrid(tr, anchors, n_epochs=ep, lr=lr)
    rng = np.random.default_rng(seed + 999)
    correct = 0; total = 0
    for j in range(len(anchors) - 1):
        A, B = anchors[j], anchors[j+1]
        # propagate A through trained bank
        cfg = FieldConfig(rows=GRID, cols=GRID)
        f = Field(cfg, seed=seed)
        f.inject([A], [1.0])
        for _ in range(12):
            f.step()
            T_flat = f.T.reshape(-1)
            T_flat += tr.bank.propagate(T_flat, eta=eta)
        T = f.T
        # pick distractor (any anchor except A, B)
        for _ in range(20):
            other = anchors[rng.integers(0, len(anchors))]
            if other != A and other != B:
                break
        s_true = float(T[B[0], B[1]])
        s_dist = float(T[other[0], other[1]])
        correct += int(s_true > s_dist)
        total += 1
    return correct / max(1, total)


def link_discrim_mlp(seed, anchors, n_hidden=64, ep=200, lr=0.05):
    rng = np.random.default_rng(seed + 777)
    pairs = [(cell_to_idx(a), cell_to_idx(b)) for a, b in
             zip(anchors[:-1], anchors[1:])]
    if not pairs:
        return 0.0
    Xs = np.zeros((len(pairs), N_CELLS), dtype=np.float32)
    ys = np.zeros(len(pairs), dtype=np.int64)
    for i, (a, b) in enumerate(pairs):
        Xs[i, a] = 1.0; ys[i] = b
    mlp = MLP(N_CELLS, n_hidden, N_CELLS, seed=seed, lr=lr)
    for _ in range(ep):
        idx = rng.permutation(len(pairs))
        mlp.step(Xs[idx], ys[idx])
    # eval link discrim
    correct = 0; total = 0
    for j in range(len(anchors) - 1):
        A, B = anchors[j], anchors[j+1]
        x = np.zeros((1, N_CELLS), dtype=np.float32)
        x[0, cell_to_idx(A)] = 1.0
        logits = mlp.forward(x)[0]
        for _ in range(20):
            other = anchors[rng.integers(0, len(anchors))]
            if other != A and other != B:
                break
        s_true = float(logits[cell_to_idx(B)])
        s_dist = float(logits[cell_to_idx(other)])
        correct += int(s_true > s_dist)
        total += 1
    return correct / max(1, total)


# ============================================================== Run
def head_to_head(n, md, ep_v7=120, ep_mlp=200):
    """Same chains, both models. Returns (mlp_acc_mean, v7_acc_mean)."""
    mlp_accs = []; v7_accs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        anchors = V7.make_chain(n, GRID, rng, md=md)
        mlp_accs.append(link_discrim_mlp(seed, anchors, ep=ep_mlp))
        v7_accs.append(link_discrim_v7(seed, anchors, ep=ep_v7))
    return float(np.mean(mlp_accs)), float(np.mean(v7_accs)), mlp_accs, v7_accs


def main():
    print("=" * 78)
    print("EXTERNAL WAR v1 — link discrimination accuracy (chance = 0.50)")
    print("Same anchors, same seeds. MLP (64 hidden) vs v7 hybrid.")
    print("=" * 78)
    print(f"{'cfg':>14} | {'MLP':>8} {'v7':>8} | {'delta':>8}")
    print("-" * 78)
    configs = [(3, 2), (5, 3), (10, 3), (5, 5), (10, 5)]
    for (n, md) in configs:
        t0 = time.time()
        mlp_m, v7_m, mlp_a, v7_a = head_to_head(n, md)
        dt = time.time() - t0
        winner = "v7" if v7_m > mlp_m else "MLP" if mlp_m > v7_m else "tie"
        print(f"  n={n} md={md:>2}  | {mlp_m:>7.3f}  {v7_m:>7.3f} | "
              f"{v7_m-mlp_m:>+7.3f}  ({winner}, {dt:.0f}s)")


if __name__ == "__main__":
    main()
