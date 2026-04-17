"""Sequence chain v5 — MULTI-LAG B channels (architectural extension).

Hypothesis: a SCALAR B channel cannot represent ordinal context, which
is why v4 (gating + b_decay + stream) hit 0/N links 2σ-positive at
n>=5. Replace the single B with K parallel B channels, each driven by
its OWN eligibility trace at a different timescale. Different lags
specialize for different chain depths:

  decay=0.50   →   short-range (i, i+1)
  decay=0.75   →   medium-range (i, i+1, i+2)
  decay=0.92   →   long-range (whole-chain context)

At probe time, the diffusion uses (C ± Σ_k B_lag_k) — all K lags
contribute additively. This is implemented WITHOUT touching core Field;
the K extra B tensors live in this script and are summed into
field.B_right/down at probe time, then restored.

Sweep: K ∈ {2, 3, 4}, n ∈ {3, 5, 7}, lr ∈ {0.05, 0.10}, ep=120.
Success gate: ≥ ⌈n/2⌉ links 2σ-positive across 8 seeds.
"""
from __future__ import annotations
import sys, time, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.sequence import EligibilityTrace, present_event, relax_with_trace

GRID = 14
SEEDS = list(range(8))
HOLD = 4
GAP = 2


def make_chain(n, grid, rng, md=5):
    chosen = []
    for _ in range(400):
        if len(chosen) == n + 1: break
        c = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        if all(abs(c[0]-x[0]) + abs(c[1]-x[1]) >= md for x in chosen):
            chosen.append(c)
    while len(chosen) < n + 1:
        chosen.append((int(rng.integers(0, grid)), int(rng.integers(0, grid))))
    return chosen


def stdp_into(B_right_aux, B_down_aux, T, E, lr, clip=1.0):
    """Bidirectional STDP: B_right>0 biases left→right, B_right<0
    biases right→left (because Field uses C_right ± B_right separately
    for fwd/bwd diffusion). Single-direction STDP (the v4 bug) only
    learned chain links whose anchors happened to lie left→right or
    top→bottom; the others got 0. This rule symmetrizes that."""
    co_h_fwd = T[:, 1:]  * E[:, :-1]   # left→right
    co_h_rev = T[:, :-1] * E[:, 1:]    # right→left
    co_v_fwd = T[1:, :]  * E[:-1, :]   # up→down
    co_v_rev = T[:-1, :] * E[1:, :]    # down→up
    B_right_aux += lr * (co_h_fwd - co_h_rev)
    B_down_aux  += lr * (co_v_fwd - co_v_rev)
    np.clip(B_right_aux, -clip, clip, out=B_right_aux)
    np.clip(B_down_aux,  -clip, clip, out=B_down_aux)


def train_multilag(field, anchors, decays, lr, n_epochs, mode="pairs"):
    """Train chain. Two modes:
      "stream" — single eligibility trace runs over the WHOLE chain
                 (A→B→C→D→E in one episode). Trace contaminates across
                 anchors → only first/last links learn.
      "pairs"  — each consecutive pair (A,B), (B,C), (C,D), ... is its
                 own mini-episode with full trace reset. Eliminates
                 cross-link interference; each B-channel position can
                 be carved cleanly.
    """
    K = len(decays)
    R, C = field.cfg.rows, field.cfg.cols
    Br = [np.zeros((R, C-1), dtype=np.float32) for _ in range(K)]
    Bd = [np.zeros((R-1, C), dtype=np.float32) for _ in range(K)]
    traces = [EligibilityTrace((R, C), decay=d) for d in decays]
    if mode == "pairs":
        for _ in range(n_epochs):
            for j in range(len(anchors) - 1):
                A, Bn = anchors[j], anchors[j+1]
                field.reset_temp()
                for t in traces: t.reset()
                # present A, let trace soak up A's diffusion
                field._clamp_positions = []; field._clamp_values = []
                field.inject([A], [1.0])
                for _ in range(HOLD):
                    field.step()
                    for t in traces: t.update(field.T)
                for _ in range(GAP):
                    field.step()
                    for t in traces: t.update(field.T)
                # present B, then fire STDP at EVERY diffusion step so the
                # B-bias cascades along the heat path A→…→B (single-shot
                # STDP at B-inject only learned adjacent pairs because the
                # rule is local).
                field._clamp_positions = []; field._clamp_values = []
                field.inject([Bn], [1.0])
                for step_i in range(HOLD):
                    for k, t in enumerate(traces):
                        stdp_into(Br[k], Bd[k], field.T, t.E, lr / HOLD)
                    field.step()
                    for t in traces: t.update(field.T)
        field.reset_temp()
        return Br, Bd
    # stream mode (legacy)
    for _ in range(n_epochs):
        field.reset_temp()
        for t in traces: t.reset()
        for i, A in enumerate(anchors):
            field._clamp_positions = []; field._clamp_values = []
            field.inject([A], [1.0])
            if i > 0:
                for k, t in enumerate(traces):
                    stdp_into(Br[k], Bd[k], field.T, t.E, lr)
            for _ in range(HOLD):
                field.step()
                for t in traces: t.update(field.T)
            for _ in range(GAP):
                field.step()
                for t in traces: t.update(field.T)
    field.reset_temp()
    return Br, Bd


def probe_with_aux(field_seed, cfg, A, B, Br_aux, Bd_aux):
    """Probe T at B after presenting A, with field.B_right/down
    temporarily set to Σ aux B."""
    f = Field(cfg, seed=field_seed)
    Br_total = sum(Br_aux); Bd_total = sum(Bd_aux)
    f.B_right = Br_total.copy().astype(np.float32)
    f.B_down  = Bd_total.copy().astype(np.float32)
    tr = EligibilityTrace((cfg.rows, cfg.cols), decay=0.80)
    present_event(f, tr, [A], [1.0], hold=HOLD)
    relax_with_trace(f, tr, 6)
    cue = float(f.T[B])
    # baseline: same probe with B all zero (no learning)
    f0 = Field(cfg, seed=field_seed)
    tr0 = EligibilityTrace((cfg.rows, cfg.cols), decay=0.80)
    present_event(f0, tr0, [A], [1.0], hold=HOLD)
    relax_with_trace(f0, tr0, 6)
    return cue - float(f0.T[B])


def run(n, K, decays, lr, ep, md=5):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    discrim = np.zeros((len(SEEDS), n), dtype=np.float32)
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        anchors = make_chain(n, GRID, rng, md=md)
        f = Field(cfg, seed=seed)
        Br, Bd = train_multilag(f, anchors, decays, lr, ep)
        for li in range(n):
            discrim[si, li] = probe_with_aux(seed, cfg, anchors[li],
                                              anchors[li+1], Br, Bd)
    # Per-link 2σ-positive count
    link_means = discrim.mean(axis=0)
    link_se = discrim.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    pos_links = int(((link_means - 2*link_se) > 0).sum())
    return discrim.mean(), pos_links, link_means


def main():
    print(f"Multi-lag chain — {len(SEEDS)} seeds, grid {GRID}x{GRID}, hold={HOLD}, gap={GAP}")
    print("=" * 80)
    print(f"{'n':>2} {'K':>2} {'lr':>5} {'ep':>4} | {'overall':>8} {'pos/N':>6} | "
          f"per-link means")
    print("-" * 80)
    configs = [
        # (n, K, decays, lr, ep, md)
        (3, 1, (0.80,), 0.10, 120, 2),                       # baseline single-lag, close anchors
        (3, 3, (0.5, 0.75, 0.92), 0.10, 120, 2),
        (5, 1, (0.80,), 0.10, 120, 2),
        (5, 3, (0.5, 0.75, 0.92), 0.10, 120, 2),
        (5, 3, (0.5, 0.75, 0.92), 0.10, 200, 2),
        (5, 4, (0.4, 0.6, 0.8, 0.95), 0.10, 200, 2),
        (5, 3, (0.5, 0.75, 0.92), 0.10, 120, 3),             # mid distance
        (7, 3, (0.5, 0.75, 0.92), 0.10, 200, 2),
    ]
    for (n, K, decays, lr, ep, md) in configs:
        t0 = time.time()
        overall, pos, lm = run(n, K, decays, lr, ep, md=md)
        dt = time.time() - t0
        gate = max(1, n // 2)
        flag = "  <-- WIN" if pos >= gate else ""
        per_link = " ".join(f"{x:+.3f}" for x in lm)
        print(f"{n:>2} {K:>2} md={md} lr={lr:.2f} ep={ep:>3} | "
              f"{overall:>+8.4f} {pos:>2}/{n} | {per_link}  ({dt:.0f}s){flag}")


if __name__ == "__main__":
    main()
