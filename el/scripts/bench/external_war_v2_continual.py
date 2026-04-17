"""External war v2 — CONTINUAL sequence learning (catastrophic forgetting test).

v1 was a single-shot memorization task — MLP's natural strength. v7's
true ground is INTERFERENCE: learn 5 sequential mini-tasks, each with
its own (A_i, B_i) chain, then test recall on ALL of them.

MLP without replay/EWC catastrophically forgets old tasks.
v7's sparse skip bank should preserve old links because each chain
writes to a different sparse subset of edges.

Metric: average link discrimination accuracy across ALL 5 tasks
after sequential training.
"""
from __future__ import annotations
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from el.thermofield.field import Field, FieldConfig
import seq_chain_v7_hybrid as V7
from external_war_v1 import MLP, softmax, cell_to_idx, N_CELLS, GRID

SEEDS = list(range(6))
N_TASKS = 5
N_PER_TASK = 4   # 4 anchors per task = 3 (A,B) pairs


def make_disjoint_chains(n_tasks, n_per_task, grid, rng, md=3):
    """Disjoint anchor sets across tasks (no anchor reuse)."""
    used = set()
    chains = []
    for _ in range(n_tasks):
        ch = []
        for _ in range(400):
            if len(ch) == n_per_task: break
            c = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
            if c in used: continue
            if all(abs(c[0]-x[0]) + abs(c[1]-x[1]) >= md for x in ch):
                ch.append(c); used.add(c)
        while len(ch) < n_per_task:
            c = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
            if c not in used:
                ch.append(c); used.add(c)
        chains.append(ch)
    return chains


def eval_link_discrim(score_fn, chains, rng):
    """Across all chains, measure P(score(A,B_true) > score(A, distractor))."""
    correct = 0; total = 0
    all_anchors = [a for ch in chains for a in ch]
    for ch in chains:
        for j in range(len(ch) - 1):
            A, B = ch[j], ch[j+1]
            for _ in range(20):
                other = all_anchors[rng.integers(0, len(all_anchors))]
                if other != A and other != B: break
            s_t = score_fn(A, B); s_d = score_fn(A, other)
            correct += int(s_t > s_d); total += 1
    return correct / max(1, total)


def continual_mlp(seed, chains, ep_per_task=200, lr=0.05, n_hidden=64):
    """Train MLP sequentially on each task. NO replay, NO EWC.
    Then evaluate accuracy across ALL tasks."""
    mlp = MLP(N_CELLS, n_hidden, N_CELLS, seed=seed, lr=lr)
    rng = np.random.default_rng(seed + 555)
    for ch in chains:
        pairs = list(zip(ch[:-1], ch[1:]))
        Xs = np.zeros((len(pairs), N_CELLS), dtype=np.float32)
        ys = np.zeros(len(pairs), dtype=np.int64)
        for i, (a, b) in enumerate(pairs):
            Xs[i, cell_to_idx(a)] = 1.0; ys[i] = cell_to_idx(b)
        for _ in range(ep_per_task):
            idx = rng.permutation(len(pairs))
            mlp.step(Xs[idx], ys[idx])

    def score(A, B):
        x = np.zeros((1, N_CELLS), dtype=np.float32); x[0, cell_to_idx(A)] = 1.0
        return float(mlp.forward(x)[0, cell_to_idx(B)])
    eval_rng = np.random.default_rng(seed + 12345)
    return eval_link_discrim(score, chains, eval_rng)


def continual_v7(seed, chains, ep_per_task=120, lr=0.20, K=4, md_min=3, eta=0.05):
    """Train v7 sequentially on each task."""
    tr = V7.HybridTrainer(rows=GRID, cols=GRID, seed=seed,
                          K=K, md_min=md_min, skip_eta=eta)
    for ch in chains:
        V7.train_chain_hybrid(tr, ch, n_epochs=ep_per_task, lr=lr)

    def score(A, B):
        cfg = FieldConfig(rows=GRID, cols=GRID)
        f = Field(cfg, seed=seed)
        f.inject([A], [1.0])
        for _ in range(12):
            f.step()
            T_flat = f.T.reshape(-1)
            T_flat += tr.bank.propagate(T_flat, eta=eta)
        return float(f.T[B[0], B[1]])
    eval_rng = np.random.default_rng(seed + 12345)
    return eval_link_discrim(score, chains, eval_rng)


# Replay baseline: MLP with full-replay buffer
def continual_mlp_replay(seed, chains, ep_per_task=200, lr=0.05, n_hidden=64):
    mlp = MLP(N_CELLS, n_hidden, N_CELLS, seed=seed, lr=lr)
    rng = np.random.default_rng(seed + 555)
    seen = []
    for ch in chains:
        pairs = list(zip(ch[:-1], ch[1:]))
        seen.extend(pairs)
        Xs = np.zeros((len(seen), N_CELLS), dtype=np.float32)
        ys = np.zeros(len(seen), dtype=np.int64)
        for i, (a, b) in enumerate(seen):
            Xs[i, cell_to_idx(a)] = 1.0; ys[i] = cell_to_idx(b)
        for _ in range(ep_per_task):
            idx = rng.permutation(len(seen))
            mlp.step(Xs[idx], ys[idx])

    def score(A, B):
        x = np.zeros((1, N_CELLS), dtype=np.float32); x[0, cell_to_idx(A)] = 1.0
        return float(mlp.forward(x)[0, cell_to_idx(B)])
    eval_rng = np.random.default_rng(seed + 12345)
    return eval_link_discrim(score, chains, eval_rng)


def main():
    print("=" * 80)
    print("EXTERNAL WAR v2 — CONTINUAL sequence learning across 5 disjoint tasks")
    print("After learning task 5, accuracy on ALL 5 tasks (catastrophic forget test)")
    print(f"  {N_TASKS} tasks × {N_PER_TASK} anchors each, md=3, 14×14 grid")
    print("=" * 80)

    mlp_naive = []; mlp_replay = []; v7s = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        chains = make_disjoint_chains(N_TASKS, N_PER_TASK, GRID, rng, md=3)
        mlp_naive.append(continual_mlp(seed, chains))
        mlp_replay.append(continual_mlp_replay(seed, chains))
        v7s.append(continual_v7(seed, chains))

    def stat(xs):
        a = np.asarray(xs)
        return f"{a.mean():.3f} ± {a.std()/np.sqrt(len(a)):.3f}"

    print()
    print(f"{'method':<28} | {'avg link-discrim acc (chance=0.50)':>40}")
    print("-" * 80)
    print(f"{'MLP naive (no replay)':<28} | {stat(mlp_naive):>40}")
    print(f"{'MLP + full replay':<28} | {stat(mlp_replay):>40}")
    print(f"{'v7 hybrid (no replay)':<28} | {stat(v7s):>40}")

    # Headline winner
    if np.mean(v7s) > np.mean(mlp_naive) + 0.05:
        print(f"\n*** v7 BEATS naive MLP by {np.mean(v7s)-np.mean(mlp_naive):+.3f} ***")
    if np.mean(v7s) > np.mean(mlp_replay) + 0.02:
        print(f"*** v7 also beats MLP+replay by {np.mean(v7s)-np.mean(mlp_replay):+.3f} ***")


if __name__ == "__main__":
    main()
