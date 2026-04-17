"""Eşik 3 — multi-task substrate probe.

Question: can a SINGLE Field instance learn pattern memory (uses C
channel via Hebb co-activation) AND a sequence A→B association (uses B
channel via STDP) without one task wrecking the other?

This is the actual kızıl elma claim: a unified backprop-free substrate
where multiple cognitive functions COEXIST. MLP cannot do this with one
network — it would need two heads with shared trunk and joint training.

Protocol:
  1. Single-task A: pattern memory only (N=8 patterns, recall acc).
  2. Single-task B: sequence association only (A→B discrimination).
  3. Joint: same Field, alternate batches of A and B training.
  4. Compare joint A-acc vs single-task A-acc, joint B-disc vs single B.

Acceptance: each task retains ≥80% of its single-task score.
"""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from el.thermofield.field import Field, FieldConfig
from el.thermofield.pattern_memory import PatternMemory
from el.thermofield.sequence import (
    EligibilityTrace, present_event, relax_with_trace, stdp_hebbian_update,
)

GRID = 28
N_PATTERNS = 8
SEEDS = list(range(6))


def gen_patterns(rng, n=N_PATTERNS, density=0.10):
    pats = []
    for _ in range(n):
        mask = rng.random((GRID, GRID)) < density
        pos = [(int(r), int(c)) for r, c in zip(*np.where(mask))]
        if not pos:
            pos = [(int(rng.integers(GRID)), int(rng.integers(GRID)))]
        pats.append(pos)
    return pats


def corrupt(pat, rng, keep_frac=0.4):
    n = max(1, int(round(len(pat) * keep_frac)))
    idx = rng.choice(len(pat), size=n, replace=False)
    return [pat[i] for i in sorted(idx)]


# --- pattern recall accuracy on a PatternMemory --------------------------
def recall_acc(pm, patterns, seed, cues_per=5):
    rng = np.random.default_rng(seed + 333)
    correct = total = 0
    for true_idx, p in enumerate(patterns):
        for _ in range(cues_per):
            cue = corrupt(p, rng)
            pred, _, _ = pm.recall(cue)
            correct += int(pred == true_idx); total += 1
    return correct / max(1, total)


# --- sequence A→B discrimination on a Field ------------------------------
def train_sequence(field, trace, A_pos, B_pos, *, epochs=30, hold=5, gap=2,
                   stdp_lr=0.07):
    for _ in range(epochs):
        field.reset_temp(); trace.reset()
        present_event(field, trace, A_pos, [1.0]*len(A_pos), hold=hold)
        relax_with_trace(field, trace, gap)
        present_event(field, trace, B_pos, [1.0]*len(B_pos), hold=hold)
        stdp_hebbian_update(field, trace, lr=stdp_lr)


def probe_seq_response(field, trace, A_pos, B_pos, *, hold=5, read_delay=6):
    """Present A alone, read mean T at B's region after relax."""
    field.reset_temp(); trace.reset()
    present_event(field, trace, A_pos, [1.0]*len(A_pos), hold=hold)
    relax_with_trace(field, trace, read_delay)
    return float(np.mean([field.T[r, c] for (r, c) in B_pos]))


def seq_discrimination(field_factory, A_pos, B_pos, *, epochs=30):
    """Trained response @ B - untrained response @ B (same field, fresh)."""
    trace = EligibilityTrace((GRID, GRID), decay=0.80)
    f_trained = field_factory()
    train_sequence(f_trained, trace, A_pos, B_pos, epochs=epochs)
    cue = probe_seq_response(f_trained, trace, A_pos, B_pos)

    f_naive = field_factory()
    trace2 = EligibilityTrace((GRID, GRID), decay=0.80)
    naive = probe_seq_response(f_naive, trace2, A_pos, B_pos)
    return cue - naive, cue, naive


# --- single-task baselines -----------------------------------------------
def single_task_pattern(seed, patterns):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pm = PatternMemory(cfg=cfg, seed=seed)
    for p in patterns:
        pm.store(p)
    return recall_acc(pm, patterns, seed)


def single_task_sequence(seed, A_pos, B_pos):
    cfg = FieldConfig(rows=GRID, cols=GRID)
    return seq_discrimination(lambda: Field(cfg, seed=seed), A_pos, B_pos)


# --- joint: same Field, alternate ----------------------------------------
def joint_task(seed, patterns, A_pos, B_pos, *, seq_epochs=30):
    """Share ONE Field across pattern memory and sequence learning.

    Schedule: store all patterns first (touches C only), then run
    sequence training on the same field (touches B only). C/B are
    independent channels but the runtime activity dynamics couple
    them — sequence training's heat propagation rides on C edges
    that pattern memory wrote.
    """
    cfg = FieldConfig(rows=GRID, cols=GRID)
    field = Field(cfg, seed=seed)
    # Build PatternMemory on top of the SAME field (shared substrate).
    pm = PatternMemory(cfg=cfg, seed=seed, field=field)
    for p in patterns:
        pm.store(p)
    # Sequence training on the same field.
    trace = EligibilityTrace((GRID, GRID), decay=0.80)
    train_sequence(field, trace, A_pos, B_pos, epochs=seq_epochs)
    # Pattern recall after sequence training (substrate now has B-bias too).
    a_acc = recall_acc(pm, patterns, seed)
    # Sequence discrimination on the joint field (trained) vs fresh naive.
    cue = probe_seq_response(field, trace, A_pos, B_pos)
    naive_cfg = FieldConfig(rows=GRID, cols=GRID)
    naive_field = Field(naive_cfg, seed=seed)
    naive_trace = EligibilityTrace((GRID, GRID), decay=0.80)
    naive = probe_seq_response(naive_field, naive_trace, A_pos, B_pos)
    return a_acc, cue - naive, cue, naive


def main():
    print("=" * 78)
    print("EŞİK 3 — MULTI-TASK SUBSTRATE PROBE")
    print(f"  grid {GRID}×{GRID}, N_PATTERNS={N_PATTERNS}, seeds={len(SEEDS)}")
    print("=" * 78)

    A_pos = [(1, 1), (1, 2), (2, 1)]
    B_pos = [(GRID-2, GRID-2), (GRID-2, GRID-3), (GRID-3, GRID-2)]

    pat_solo = []; seq_solo = []
    pat_joint = []; seq_joint = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        patterns = gen_patterns(rng)
        pat_solo.append(single_task_pattern(seed, patterns))
        d_solo, _, _ = single_task_sequence(seed, A_pos, B_pos)
        seq_solo.append(d_solo)
        a_j, d_j, _, _ = joint_task(seed, patterns, A_pos, B_pos)
        pat_joint.append(a_j); seq_joint.append(d_j)

    def stat(xs):
        a = np.asarray(xs)
        return f"{a.mean():.3f} ± {a.std()/np.sqrt(len(a)):.3f}"

    print()
    print(f"{'task':<32} | {'solo':>20} | {'joint':>20}")
    print("-" * 78)
    print(f"{'pattern recall (chance=0.125)':<32} | "
          f"{stat(pat_solo):>20} | {stat(pat_joint):>20}")
    print(f"{'seq A->B discrimination':<32} | "
          f"{stat(seq_solo):>20} | {stat(seq_joint):>20}")

    pat_ratio = np.mean(pat_joint) / max(1e-9, np.mean(pat_solo))
    seq_ratio = np.mean(seq_joint) / max(1e-9, np.mean(seq_solo))
    print()
    print(f"Pattern joint/solo ratio = {pat_ratio:.3f}  (need ≥0.80)")
    print(f"Sequence joint/solo ratio = {seq_ratio:.3f} (need ≥0.80)")

    if pat_ratio >= 0.80 and seq_ratio >= 0.80:
        print("\n*** EŞİK 3 GEÇTİ: substrate hosts BOTH tasks ≥80% ***")
    else:
        which = []
        if pat_ratio < 0.80: which.append(f"pattern degraded to {pat_ratio:.0%}")
        if seq_ratio < 0.80: which.append(f"sequence degraded to {seq_ratio:.0%}")
        print(f"\n!!! EŞİK 3 KIRILDI: {', '.join(which)} — interference !!!")


if __name__ == "__main__":
    main()
