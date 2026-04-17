"""el_continual — substrate as a CONTINUALLY LIVING memory.

Local Hebbian plasticity means the field never stops learning. This
test streams Gutenberg text chunks one at a time, never resetting,
and periodically probes recall on:
  - the OLDEST chunks (forgetting curve — does old knowledge survive?)
  - the NEWEST chunks (assimilation — is new info being absorbed?)
  - a HOLDOUT random subset (overall content-addressable health)

Persistence (Eşik 4) already proved state survives process death.
This proves the substrate KEEPS LEARNING in a single live process
without catastrophic interference resetting old patterns.
"""
from __future__ import annotations
import sys, time, random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory
from el_text_massive import (
    load_books, chunks, text_to_pattern,
    SubstrateTextMemory, eval_on,
)


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    print("=" * 78)
    print("EL CONTINUAL — substrate streams Gutenberg chunks, never resets")
    print("=" * 78)
    text = load_books(data_dir)
    print(f"[corpus] {len(text):,} chars")

    win, cue_len = 120, 80
    grid = 192
    total = 3000             # stream length
    probe_every = 500        # probe interval
    probe_n = 100            # eval set size per probe
    print(f"[setup] grid={grid}x{grid}={grid*grid} cells, "
          f"stream={total}, probe every {probe_every}, n_eval={probe_n}")

    cs = list(dict.fromkeys(chunks(text, win, total + 200, seed=42)))[:total]
    print(f"[stream] {len(cs)} unique chunks ready")

    sub = SubstrateTextMemory(grid=grid)
    rng = random.Random(7)

    print(f"\n{'streamed':>8} {'oldest_ex':>10} {'oldest_n':>10} "
          f"{'newest_ex':>10} {'newest_n':>10} {'random_ex':>10} {'wall_s':>8}")
    print("-" * 78)
    t0 = time.time()
    for i, c in enumerate(cs, 1):
        sub.store(c)
        if i % probe_every == 0 or i == total:
            # OLDEST: indices 0..min(probe_n,i)
            old_idx = list(range(min(probe_n, i)))
            # NEWEST: indices i-probe_n..i
            new_idx = list(range(max(0, i - probe_n), i))
            # RANDOM: subsample of all stored
            rnd_idx = sorted(rng.sample(range(i), min(probe_n, i)))

            ex_o, _ = eval_on(sub, sub.full_texts, old_idx,
                              cue_len, noise_chars=0, seed=11)
            ex_o_n, _ = eval_on(sub, sub.full_texts, old_idx,
                                cue_len, noise_chars=3, seed=12)
            ex_n, _ = eval_on(sub, sub.full_texts, new_idx,
                              cue_len, noise_chars=0, seed=13)
            ex_n_n, _ = eval_on(sub, sub.full_texts, new_idx,
                                cue_len, noise_chars=3, seed=14)
            ex_r, _ = eval_on(sub, sub.full_texts, rnd_idx,
                              cue_len, noise_chars=0, seed=15)

            print(f"{i:>8} {ex_o:>10.3f} {ex_o_n:>10.3f} "
                  f"{ex_n:>10.3f} {ex_n_n:>10.3f} {ex_r:>10.3f} "
                  f"{time.time()-t0:>8.1f}", flush=True)

    print("-" * 78)
    print("legend: oldest_ex = recall on first 100 stored (clean cue)")
    print("        oldest_n  = same, with 3-char cue noise")
    print("        newest_ex = recall on most recent 100 stored")
    print("        newest_n  = same, noisy cue")
    print("        random_ex = recall on 100 random stored chunks")


if __name__ == "__main__":
    main()
