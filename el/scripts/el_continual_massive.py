"""el_continual_massive — substrate alive AT SCALE: 10K chunks streamed
into a 256x256 grid, probed every 1000 for old/new/random recall."""
from __future__ import annotations
import sys, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from el_text_massive import load_books, chunks, SubstrateTextMemory, eval_on


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    print("=" * 78)
    print("EL CONTINUAL @ MASSIVE — 10K chunks streamed, grid 256x256")
    print("=" * 78)
    text = load_books(data_dir)
    print(f"[corpus] {len(text):,} chars")

    win, cue_len, grid = 120, 80, 256
    total, probe_every, probe_n = 5000, 500, 100
    print(f"[setup] grid={grid}x{grid}={grid*grid} cells, "
          f"stream={total}, probe every {probe_every}, n_eval={probe_n}")

    cs = list(dict.fromkeys(chunks(text, win, total + 500, seed=42)))[:total]
    print(f"[stream] {len(cs)} unique chunks ready")

    sub = SubstrateTextMemory(grid=grid)
    rng = random.Random(7)
    print(f"\n{'streamed':>8} {'oldest_ex':>10} {'old_noisy':>10} "
          f"{'newest_ex':>10} {'new_noisy':>10} {'random_ex':>10} {'wall_s':>8}")
    print("-" * 78)
    t0 = time.time()
    for i, c in enumerate(cs, 1):
        sub.store(c)
        if i % probe_every == 0 or i == total:
            old_idx = list(range(min(probe_n, i)))
            new_idx = list(range(max(0, i - probe_n), i))
            rnd_idx = sorted(rng.sample(range(i), min(probe_n, i)))
            ex_o, _ = eval_on(sub, sub.full_texts, old_idx, cue_len, 0, 11)
            ex_on, _ = eval_on(sub, sub.full_texts, old_idx, cue_len, 3, 12)
            ex_n, _ = eval_on(sub, sub.full_texts, new_idx, cue_len, 0, 13)
            ex_nn, _ = eval_on(sub, sub.full_texts, new_idx, cue_len, 3, 14)
            ex_r, _ = eval_on(sub, sub.full_texts, rnd_idx, cue_len, 0, 15)
            print(f"{i:>8} {ex_o:>10.3f} {ex_on:>10.3f} {ex_n:>10.3f} "
                  f"{ex_nn:>10.3f} {ex_r:>10.3f} {time.time()-t0:>8.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
