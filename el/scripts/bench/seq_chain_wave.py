"""Sequence chain v6 — WAVE substrate prototype. NEGATIVE RESULT, FROZEN.

================================================================
STATUS: FROZEN as negative result, do not tune further.
================================================================

OUTCOME:
  - Wave PROPAGATION works: T(d=10)=0.018 vs heat ≈ 0 (kategori-açıcı)
  - Wave SEQUENCE LEARNING fails: 3 STDP variants tried (bidirectional
    |T|, unidirectional |T|, signed velocity), all yield 0-1/N pos
    links at n=5+, md=3+. See log in replit.md.

ROOT CAUSE (architectural, not tuning):
  Wave equation is time-reversible. On a finite grid, waves reflect
  and superpose into standing+traveling interference. Local STDP
  measuring T*E at any cell sees this global mixture and cannot
  recover the source direction. Damping (gamma) trades distance
  for direction: gamma↑ collapses to heat (no distance), gamma↓
  loses direction. No gamma satisfies both.

LESSON FOR v7 HYBRID:
  - Heat substrate gives DIRECTION but no distance (v4 result).
  - Wave substrate gives DISTANCE but no direction (v6 result).
  - The synthesis is NOT pure-physics; it requires a separate
    sparse temporal skip-edge bank carrying long-range signals
    over the existing heat substrate. See seq_chain_v7_hybrid.py.

Hypothesis (original): heat diffusion erases information exponentially with
distance, so STDP cannot bridge anchor pairs with min_dist > 1.
Wave equation propagates energy WITHOUT dissipation (other than
optional viscous damping), so A's signal reaches B even at d=10.
With local STDP unchanged, sequence chains of length 5-10 should
become learnable.

Discrete wave equation (2D, finite difference, leap-frog):
  T_next = 2*T - T_prev + (c·dt/dx)² · ∇²T  -  γ·dt·(T - T_prev)
CFL stability: c·dt/dx ≤ 1/√2  in 2D.

This script is SELF-CONTAINED (does not touch core Field). It
reuses the v4 benchmark protocol (8 seeds, 14×14 grid, anchors
with min_dist=2/3/5, chain lengths 3/5/10) so numbers are directly
comparable to the v4 negative result.
"""
from __future__ import annotations
import sys, time, numpy as np

GRID = 14
SEEDS = list(range(8))


# ============================================================== Wave
class WaveField:
    """Self-contained 2D wave field with light viscous damping.
    State: T (current), T_prev (one step back). Update via leap-frog.
    """
    def __init__(self, rows, cols, c=0.5, dt=1.0, dx=1.0, gamma=0.05, seed=0):
        self.rows = rows; self.cols = cols
        self.c = c; self.dt = dt; self.dx = dx; self.gamma = gamma
        self.alpha = (c * dt / dx) ** 2
        # CFL: c*dt/dx <= 1/sqrt(2) ≈ 0.707 in 2D
        if c * dt / dx > 0.707:
            raise ValueError(f"CFL violated: c·dt/dx={c*dt/dx:.3f} > 0.707")
        self.T = np.zeros((rows, cols), dtype=np.float32)
        self.T_prev = np.zeros_like(self.T)
        # B-bias (directional) — used at probe time for STDP-modulated propagation
        self.B_right = np.zeros((rows, cols-1), dtype=np.float32)
        self.B_down  = np.zeros((rows-1, cols), dtype=np.float32)

    def reset(self):
        self.T.fill(0.0); self.T_prev.fill(0.0)

    def inject(self, positions, values):
        for (r, c), v in zip(positions, values):
            self.T[r, c] = v
            self.T_prev[r, c] = v   # zero initial velocity

    def laplacian(self, T):
        L = np.zeros_like(T)
        L[1:-1, 1:-1] = (T[2:, 1:-1] + T[:-2, 1:-1] +
                         T[1:-1, 2:] + T[1:-1, :-2] - 4*T[1:-1, 1:-1])
        # reflective boundaries
        L[0, 1:-1]  = T[1, 1:-1]  + T[0, 2:]   + T[0, :-2]   - 3*T[0, 1:-1]
        L[-1, 1:-1] = T[-2, 1:-1] + T[-1, 2:]  + T[-1, :-2]  - 3*T[-1, 1:-1]
        L[1:-1, 0]  = T[2:, 0]    + T[:-2, 0]  + T[1:-1, 1]  - 3*T[1:-1, 0]
        L[1:-1, -1] = T[2:, -1]   + T[:-2, -1] + T[1:-1, -2] - 3*T[1:-1, -1]
        L[0, 0]   = T[1, 0]   + T[0, 1]   - 2*T[0, 0]
        L[0, -1]  = T[1, -1]  + T[0, -2]  - 2*T[0, -1]
        L[-1, 0]  = T[-2, 0]  + T[-1, 1]  - 2*T[-1, 0]
        L[-1, -1] = T[-2, -1] + T[-1, -2] - 2*T[-1, -1]
        return L

    def step(self):
        # advection-modulated lap: B_right shifts horizontal lap, B_down vertical
        L = self.laplacian(self.T)
        # B-bias: add directional drift via gradient × B
        if np.any(self.B_right) or np.any(self.B_down):
            grad_h = self.T[:, 1:] - self.T[:, :-1]
            grad_v = self.T[1:, :] - self.T[:-1, :]
            drift_h = self.B_right * grad_h    # shape (R, C-1)
            drift_v = self.B_down  * grad_v    # shape (R-1, C)
            # divergence of drift adds to L
            div = np.zeros_like(self.T)
            div[:, :-1] -= drift_h
            div[:, 1:]  += drift_h
            div[:-1, :] -= drift_v
            div[1:, :]  += drift_v
            L = L + 0.3 * div   # weight on directional bias
        new_T = (2*self.T - self.T_prev + self.alpha * L
                 - self.gamma * self.dt * (self.T - self.T_prev))
        self.T_prev = self.T
        self.T = new_T


# ============================================================== Probes
def propagation_test():
    """Quick sanity: inject at center, measure energy at d=5 and d=10
    after 20 steps. Heat would decay to ~0; wave should retain signal.
    """
    print("\n[propagation test] center inject, energy at d=5 / d=10 after 20 steps")
    f = WaveField(28, 28, c=0.5, gamma=0.02)
    f.inject([(14, 14)], [1.0])
    for _ in range(20):
        f.step()
    e5  = float(np.abs(f.T[14, 19]))   # 5 cells right
    e10 = float(np.abs(f.T[14, 24]))   # 10 cells right
    e0  = float(np.abs(f.T[14, 14]))
    print(f"  T(0)={e0:.4f}  T(5)={e5:.4f}  T(10)={e10:.4f}")
    return e5, e10


# ============================================================== STDP
class EligibilityTrace:
    def __init__(self, shape, decay=0.85):
        self.E = np.zeros(shape, dtype=np.float32)
        self.decay = decay
    def reset(self): self.E.fill(0.0)
    def update(self, T):
        self.E = self.decay * self.E + (1 - self.decay) * np.abs(T)


def stdp_into(B_right, B_down, T, E, lr, clip=1.0):
    """Bidirectional STDP on |T| × |E| (wave amplitude is signed)."""
    aT = np.abs(T); aE = np.abs(E)
    co_h_fwd = aT[:, 1:]  * aE[:, :-1]
    co_h_rev = aT[:, :-1] * aE[:, 1:]
    co_v_fwd = aT[1:, :]  * aE[:-1, :]
    co_v_rev = aT[:-1, :] * aE[1:, :]
    B_right += lr * (co_h_fwd - co_h_rev)
    B_down  += lr * (co_v_fwd - co_v_rev)
    np.clip(B_right, -clip, clip, out=B_right)
    np.clip(B_down,  -clip, clip, out=B_down)


# ============================================================== Chain
def make_chain(n, grid, rng, md=2):
    chosen = []
    for _ in range(400):
        if len(chosen) == n + 1: break
        c = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        if all(abs(c[0]-x[0]) + abs(c[1]-x[1]) >= md for x in chosen):
            chosen.append(c)
    while len(chosen) < n + 1:
        chosen.append((int(rng.integers(0, grid)), int(rng.integers(0, grid))))
    return chosen


def train_chain_wave(field, anchors, n_epochs=120, lr=0.10,
                     hold=4, gap=2, decays=(0.5, 0.75, 0.92)):
    """Pair-episode training with K parallel traces + bidirectional STDP."""
    K = len(decays)
    R, C = field.rows, field.cols
    Br = [np.zeros((R, C-1), dtype=np.float32) for _ in range(K)]
    Bd = [np.zeros((R-1, C), dtype=np.float32) for _ in range(K)]
    traces = [EligibilityTrace((R, C), decay=d) for d in decays]
    for _ in range(n_epochs):
        for j in range(len(anchors) - 1):
            A, Bn = anchors[j], anchors[j+1]
            field.reset()
            for t in traces: t.reset()
            field.inject([A], [1.0])
            for _ in range(hold):
                field.step()
                for t in traces: t.update(field.T)
            for _ in range(gap):
                field.step()
                for t in traces: t.update(field.T)
            field.inject([Bn], [1.0])
            for _ in range(hold):
                for k, t in enumerate(traces):
                    stdp_into(Br[k], Bd[k], field.T, t.E, lr / hold)
                field.step()
                for t in traces: t.update(field.T)
    return Br, Bd


def probe_link(seed, A, B, Br_aux, Bd_aux, c=0.5, gamma=0.05, hold=4, rd=8):
    """Inject A, propagate (rd+hold) steps with B-bias active, read |T| at B.
    Compare to fresh wave field with B=0."""
    f = WaveField(GRID, GRID, c=c, gamma=gamma, seed=seed)
    f.B_right = sum(Br_aux).copy()
    f.B_down  = sum(Bd_aux).copy()
    f.inject([A], [1.0])
    for _ in range(hold + rd):
        f.step()
    cue = float(np.abs(f.T[B[0], B[1]]))
    f0 = WaveField(GRID, GRID, c=c, gamma=gamma, seed=seed)
    f0.inject([A], [1.0])
    for _ in range(hold + rd):
        f0.step()
    base = float(np.abs(f0.T[B[0], B[1]]))
    return cue - base


# ============================================================== Run
def run_chain(n, md, lr=0.10, ep=120, decays=(0.5, 0.75, 0.92)):
    discrim = np.zeros((len(SEEDS), n), dtype=np.float32)
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        anchors = make_chain(n, GRID, rng, md=md)
        f = WaveField(GRID, GRID, c=0.5, gamma=0.05, seed=seed)
        Br, Bd = train_chain_wave(f, anchors, n_epochs=ep, lr=lr, decays=decays)
        for li in range(n):
            discrim[si, li] = probe_link(seed, anchors[li], anchors[li+1], Br, Bd)
    link_means = discrim.mean(axis=0)
    link_se = discrim.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    pos_links = int(((link_means - 2*link_se) > 0).sum())
    return discrim.mean(), pos_links, link_means


def main():
    propagation_test()
    print("\n[chain head-to-head]  v4 baseline at md=2-5: 0/N pos for n>=5")
    print("=" * 88)
    print(f"{'n':>2} {'md':>2} {'lr':>5} {'ep':>4} | {'overall':>9} {'pos/N':>6} | "
          f"per-link means")
    print("-" * 88)
    configs = [
        (3,  2, 0.10, 120),
        (3,  3, 0.10, 120),
        (5,  2, 0.10, 120),
        (5,  3, 0.10, 120),
        (5,  5, 0.10, 200),
        (7,  3, 0.10, 200),
        (10, 3, 0.10, 200),
        (10, 5, 0.10, 300),
    ]
    for (n, md, lr, ep) in configs:
        t0 = time.time()
        overall, pos, lm = run_chain(n, md, lr, ep)
        dt = time.time() - t0
        gate = max(1, n // 2)
        flag = "  <-- WIN" if pos >= gate else ""
        per_link = " ".join(f"{x:+.3f}" for x in lm)
        print(f"{n:>2} {md:>2} {lr:>5.2f} {ep:>4} | {overall:>+9.4f} {pos:>2}/{n}  | "
              f"{per_link}  ({dt:.0f}s){flag}")


if __name__ == "__main__":
    main()
