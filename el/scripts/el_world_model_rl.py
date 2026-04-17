"""el_world_model_rl — substrate as a TRUE world model on CartPole-v1.

This is the right test (Ha-Schmidhuber sense):
  Learn p(s_{t+1} | s_t, a_t) from real RL transitions, then USE the
  model: (i) measure 1-step prediction fidelity, (ii) k-step open-loop
  rollout fidelity vs ground truth, (iii) model-based planning — pick
  the action whose imagined rollout keeps the pole upright longest,
  compare resulting episode length to a random baseline.

Substrate role:
  PatternMemory stores (state_bin, action) -> next_state_bin. Recall
  by injecting (state, action) cue, retrieve the closest stored
  context, return the associated next state.

Baselines:
  - identity: s' = s (cart-pole has very smooth dynamics, this is hard)
  - kNN-1: vanilla brute-force nearest-neighbor over (s,a) feature vec
  - tabular MLE: dict-of-counts p(s'|s,a) over discretized states
"""
from __future__ import annotations
import sys, time, hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory


# ---------- environment & data ----------
def collect_transitions(n: int = 20000, seed: int = 0
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random-policy rollouts from real CartPole-v1.
    Returns (S, A, S_next, done) where S/S_next are float32 [n,4]."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    S, A, Sn, D = [], [], [], []
    s, _ = env.reset(seed=seed)
    while len(S) < n:
        a = int(rng.integers(0, 2))
        s2, _, term, trunc, _ = env.step(a)
        S.append(s); A.append(a); Sn.append(s2); D.append(term or trunc)
        s = s2 if not (term or trunc) else env.reset(seed=int(rng.integers(1<<30)))[0]
    env.close()
    return (np.array(S, dtype=np.float32), np.array(A, dtype=np.int8),
            np.array(Sn, dtype=np.float32), np.array(D, dtype=np.bool_))


def fit_bin_edges(S: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-dim equal-frequency bin edges (4 dims × (n_bins+1) edges)."""
    edges = np.empty((S.shape[1], n_bins + 1), dtype=np.float32)
    for d in range(S.shape[1]):
        edges[d] = np.quantile(S[:, d], np.linspace(0, 1, n_bins + 1))
        edges[d, 0] -= 1e-3; edges[d, -1] += 1e-3
    return edges


def bin_state(s: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Discretize a single state vec to per-dim bin indices, [4]."""
    out = np.empty(s.shape[-1], dtype=np.int32)
    for d in range(s.shape[-1]):
        out[d] = np.searchsorted(edges[d], s[d], side="right") - 1
        out[d] = max(0, min(edges.shape[1] - 2, int(out[d])))
    return out


# ---------- substrate encoder ----------
def sa_to_pattern(s_bins: np.ndarray, a: int, n_bins: int, grid: int
                  ) -> list[tuple[int, int]]:
    """Encode (state_bins[4], action) into a sparse cell pattern."""
    cells = set()
    # one row band per state dim, column = bin
    for d, b in enumerate(s_bins):
        row_band = (d * grid) // 4
        col_band = (int(b) * grid) // n_bins
        for dr in range(grid // 8):  # thick row stripe
            r = min(grid - 1, row_band + dr)
            cells.add((r, min(grid - 1, col_band)))
    # action bottom band
    a_band = grid - 1 - int(a)
    for c in range(grid):
        cells.add((a_band, c))
    # redundant hash cells for robustness
    for d, b in enumerate(s_bins):
        for s in range(4):
            h = hashlib.blake2b(f"{d}|{int(b)}|{int(a)}|{s}".encode(),
                                digest_size=4).digest()
            idx = int.from_bytes(h, "big") % (grid * grid)
            cells.add((min(grid - 1, idx // grid),
                       min(grid - 1, idx % grid)))
    return sorted(cells)


# ---------- substrate world model ----------
class SubstrateWM:
    def __init__(self, n_bins: int, grid: int = 64, max_store: int = 3000,
                 seed: int = 0):
        self.n_bins, self.grid = n_bins, grid
        self.cfg = FieldConfig(rows=grid, cols=grid)
        self.pm = PatternMemory(
            cfg=self.cfg, seed=seed,
            write_lr=0.07, write_steps=10, write_decay=0.005,
            recall_steps=6,
        )
        self.labels: list[np.ndarray] = []   # next_state_bins per stored
        self.max_store = max_store

    def fit(self, S_bins: np.ndarray, A: np.ndarray, Sn_bins: np.ndarray):
        n = len(S_bins)
        idx = np.linspace(0, n - 1, min(n, self.max_store)).astype(int)
        for i in idx:
            self.pm.store(sa_to_pattern(S_bins[i], int(A[i]),
                                        self.n_bins, self.grid))
            self.labels.append(Sn_bins[i].copy())

    def predict_bins(self, s_bins: np.ndarray, a: int) -> np.ndarray:
        cue = sa_to_pattern(s_bins, a, self.n_bins, self.grid)
        idx, _, _ = self.pm.recall(cue)
        if 0 <= idx < len(self.labels):
            return self.labels[idx]
        return s_bins.copy()


# ---------- baselines ----------
class TabularMLE:
    """Count-based p(s'|s,a). Falls back to identity on unseen contexts."""
    def __init__(self): self.tbl: dict = defaultdict(Counter)

    def fit(self, S_bins, A, Sn_bins):
        for s, a, sn in zip(S_bins, A, Sn_bins):
            key = (tuple(s.tolist()), int(a))
            self.tbl[key][tuple(sn.tolist())] += 1

    def predict_bins(self, s_bins, a):
        key = (tuple(s_bins.tolist()), int(a))
        if key in self.tbl:
            best = self.tbl[key].most_common(1)[0][0]
            return np.array(best, dtype=np.int32)
        return s_bins.copy()


class KNN1:
    def __init__(self, max_store: int = 3000):
        self.X = None; self.Y = None; self.max_store = max_store
    def fit(self, S_bins, A, Sn_bins):
        n = len(S_bins)
        idx = np.linspace(0, n - 1, min(n, self.max_store)).astype(int)
        feats = np.concatenate([S_bins[idx].astype(np.float32),
                                A[idx].astype(np.float32)[:, None] * 5.0],
                               axis=1)
        self.X = feats; self.Y = Sn_bins[idx].copy()
    def predict_bins(self, s_bins, a):
        f = np.concatenate([s_bins.astype(np.float32),
                            np.array([float(a) * 5.0], dtype=np.float32)])
        d = np.linalg.norm(self.X - f, axis=1)
        return self.Y[int(np.argmin(d))].copy()


class Identity:
    def fit(self, *a, **k): pass
    def predict_bins(self, s_bins, a): return s_bins.copy()


# ---------- evaluation ----------
def per_dim_acc(pred_bins: np.ndarray, true_bins: np.ndarray) -> tuple[float, float]:
    """Returns (exact_match_all_dims, mean_per_dim_acc)."""
    exact = float(np.all(pred_bins == true_bins, axis=1).mean())
    perdim = float((pred_bins == true_bins).mean())
    return exact, perdim


def eval_1step(model, S_bins, A, Sn_bins, n_eval: int = 1000):
    pred = np.empty_like(Sn_bins[:n_eval])
    for i in range(n_eval):
        pred[i] = model.predict_bins(S_bins[i], int(A[i]))
    return per_dim_acc(pred, Sn_bins[:n_eval])


def eval_kstep_rollout(model, S_bins, A, Sn_bins, k: int = 5, n_eval: int = 200):
    """Open-loop k-step rollout: feed model's own predictions back."""
    correct = total = 0
    perdim_correct = perdim_total = 0
    for start in range(0, n_eval):
        # find a contiguous run of k transitions in the test set without resets
        # (we approximate by just rolling k steps from each start)
        s = S_bins[start].copy()
        for step in range(k):
            j = start + step
            if j >= len(A): break
            s_pred = model.predict_bins(s, int(A[j]))
            true = Sn_bins[j]
            correct += int(np.all(s_pred == true)); total += 1
            perdim_correct += int((s_pred == true).sum())
            perdim_total += 4
            s = s_pred
    return correct / max(1, total), perdim_correct / max(1, perdim_total)


# ---------- model-based planning ----------
def plan_action(model, s_bins, depth: int = 4) -> int:
    """Roll out both actions for `depth` steps under random follow-up,
    pick the one that imagines the longest survival (no terminal bin)."""
    best_a, best_score = 0, -1
    for a0 in (0, 1):
        s = model.predict_bins(s_bins, a0)
        # heuristic: extreme bins on dim 2 (pole angle) ≈ falling
        n_bins = model.n_bins if hasattr(model, "n_bins") else 8
        score = 0
        for _ in range(depth - 1):
            # pole angle bin distance from center
            score += abs(int(s[2]) - n_bins // 2)
            a = 0 if int(s[2]) < n_bins // 2 else 1   # naive corrective
            s = model.predict_bins(s, a)
        # lower score = pole stayed near upright
        if -score > best_score:
            best_score = -score; best_a = a0
    return best_a


def play_episode(model, edges, max_steps: int = 100, seed: int = 0,
                 use_model: bool = True) -> int:
    env = gym.make("CartPole-v1")
    s, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 99991)
    for t in range(max_steps):
        if use_model and model is not None:
            a = plan_action(model, bin_state(s, edges))
        else:
            a = int(rng.integers(0, 2))
        s, _, term, trunc, _ = env.step(a)
        if term or trunc: env.close(); return t + 1
    env.close()
    return max_steps


def main():
    print("=" * 78)
    print("EL WORLD MODEL — REAL CartPole-v1 dynamics (gymnasium)")
    print("=" * 78)

    print("[data] collecting 8K random-policy transitions ...")
    t0 = time.time()
    S, A, Sn, _ = collect_transitions(n=8000, seed=0)
    print(f"[data] collected {len(S)} transitions in {time.time()-t0:.1f}s")

    n_bins = 8
    edges = fit_bin_edges(S, n_bins)
    S_bins = np.stack([bin_state(s, edges) for s in S])
    Sn_bins = np.stack([bin_state(s, edges) for s in Sn])
    print(f"[bins] {n_bins} bins/dim → {n_bins**4} discrete states. "
          f"unique (s,a) seen: {len(set(zip(map(tuple, S_bins.tolist()), A.tolist())))}")

    # 80/20 split
    cut = int(0.80 * len(S))
    Str, Atr, Sntr = S_bins[:cut], A[:cut], Sn_bins[:cut]
    Ste, Ate, Snte = S_bins[cut:], A[cut:], Sn_bins[cut:]

    models = {
        "identity (s'=s)": Identity(),
        "tabular MLE":     TabularMLE(),
        "kNN-1 over (s,a)": KNN1(max_store=1500),
        "substrate WM":    SubstrateWM(n_bins=n_bins, grid=48, max_store=1500),
    }

    print("\n[fit]")
    for name, m in models.items():
        t0 = time.time()
        m.fit(Str, Atr, Sntr)
        print(f"  {name:<22} fit={time.time()-t0:.1f}s")

    print("\n[1-step prediction]  (n_eval=400 held-out transitions)")
    print(f"  {'model':<22} {'exact-all-4-dims':>18} {'per-dim acc':>14}")
    rs1 = {}
    for name, m in models.items():
        ex, pd = eval_1step(m, Ste, Ate, Snte, n_eval=400)
        rs1[name] = (ex, pd)
        print(f"  {name:<22} {ex:>18.3f} {pd:>14.3f}")

    print("\n[3-step open-loop rollout]  (n_eval=100 starts)")
    print(f"  {'model':<22} {'exact-all-4-dims':>18} {'per-dim acc':>14}")
    rs5 = {}
    for name, m in models.items():
        ex, pd = eval_kstep_rollout(m, Ste, Ate, Snte, k=3, n_eval=100)
        rs5[name] = (ex, pd)
        print(f"  {name:<22} {ex:>18.3f} {pd:>14.3f}")

    print("\n[model-based planning]  (5 episodes, max 100 steps each)")
    print("  (controller picks action whose imagined rollout keeps pole upright)")
    print(f"  {'controller':<22} {'mean episode length':>22}")
    rsP = {}
    for name in ["random (no model)", "identity", "tabular MLE",
                 "kNN-1", "substrate WM"]:
        if name == "random (no model)":
            lens = [play_episode(None, edges, seed=s, use_model=False)
                    for s in range(5)]
        else:
            key = {"identity": "identity (s'=s)", "tabular MLE": "tabular MLE",
                   "kNN-1": "kNN-1 over (s,a)", "substrate WM": "substrate WM"}[name]
            lens = [play_episode(models[key], edges, seed=s, use_model=True)
                    for s in range(5)]
        rsP[name] = float(np.mean(lens))
        print(f"  {name:<22} {np.mean(lens):>22.1f}  (range {min(lens)}..{max(lens)})")

    # ranking summary
    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(f"{'task':<32} {'winner':<24} {'top score'}")
    for tag, rs in [("1-step exact", {k: v[0] for k, v in rs1.items()}),
                    ("5-step exact", {k: v[0] for k, v in rs5.items()}),
                    ("planning ep length", rsP)]:
        winner = max(rs, key=rs.get)
        print(f"{tag:<32} {winner:<24} {rs[winner]:.3f}")
        sub_key = "substrate WM"
        sub_score = rs.get(sub_key, 0)
        is_sub_win = winner == sub_key
        print(f"  substrate score: {sub_score:.3f}  "
              f"({'WIN' if is_sub_win else 'loses to ' + winner})")


if __name__ == "__main__":
    main()
