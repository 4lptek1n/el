"""Probe: does the substrate's stored Hebb actually do recall work, or is
recall mostly cue overlap?

Honest cue protocol:
  cue = (50% kept subset of pattern) + (k distractor cells from outside)

Under this cue, the cue alone has low Jaccard with the original pattern
(half the cue is noise). Only a substrate that ROUTES heat from kept
pattern cells to the missing pattern cells via stored Hebb edges can
beat plain cue overlap.
"""
from __future__ import annotations

import numpy as np

from el.thermofield import FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern


def noisy_cue(pattern, drop_frac, rng, n_cells, cols):
    keep_n = max(1, int(round(len(pattern) * (1.0 - drop_frac))))
    keep = [pattern[i] for i in
            sorted(rng.choice(len(pattern), keep_n, replace=False))]
    pattern_set = set(pattern)
    # Same number of distractors as kept cells
    distractors = []
    while len(distractors) < keep_n:
        idx = int(rng.integers(0, n_cells))
        rc = (idx // cols, idx % cols)
        if rc not in pattern_set and rc not in distractors:
            distractors.append(rc)
    return keep + distractors


def run(write_steps, write_lr, n_patterns=3, n_trials=20, seed=0):
    rng = np.random.default_rng(seed)
    cfg = FieldConfig(rows=14, cols=14)
    n_cells = cfg.rows * cfg.cols
    mem = PatternMemory(cfg=cfg, seed=seed,
                        write_steps=write_steps, write_lr=write_lr)
    patterns = [random_pattern(cfg.rows, cfg.cols, k=10, rng=rng)
                for _ in range(n_patterns)]
    for p in patterns:
        mem.store(p)

    rng_eval = np.random.default_rng(seed + 1)
    correct = 0
    for _ in range(n_trials):
        i = int(rng_eval.integers(0, n_patterns))
        cue = noisy_cue(patterns[i], 0.5, rng_eval, n_cells, cfg.cols)
        best_i, _, _ = mem.recall(cue)
        if best_i == i:
            correct += 1
    return correct / n_trials


def main():
    print("Honest cue (50% subset + 50% distractors):")
    untrained = run(write_steps=0, write_lr=0.0, seed=0)
    trained = run(write_steps=8, write_lr=0.15, seed=0)
    print(f"  untrained acc: {untrained:.2f}")
    print(f"  trained   acc: {trained:.2f}")
    print(f"  chance     acc: {1/3:.2f}")
    print(f"  lift over baseline: {trained - untrained:+.2f}")

    print("\nWith stronger / longer write:")
    trained2 = run(write_steps=20, write_lr=0.25, seed=0)
    print(f"  trained2  acc: {trained2:.2f}")


if __name__ == "__main__":
    main()
