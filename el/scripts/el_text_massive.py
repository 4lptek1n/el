"""el_text_massive — substrate as content-addressable memory at MASSIVE scale.

Real corpus: 5 Project Gutenberg books concatenated (Pride & Prejudice,
Moby Dick, Sherlock Holmes, Frankenstein, Alice, Tom Sawyer ~3.5 MB).

Sweep N up to 20000 chunks. Subsample evaluation set (n_eval=200) for
speed at the largest N.
"""
from __future__ import annotations
import sys, time, hashlib, random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory


def load_books(data_dir: Path) -> str:
    parts = []
    for name in ["pride.txt", "moby.txt", "sherlock.txt", "frankenstein.txt",
                 "alice.txt", "tomsawyer.txt"]:
        p = data_dir / name
        if not p.exists(): continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        s, e = raw.find("*** START OF"), raw.find("*** END OF")
        if s >= 0: s = raw.find("\n", s) + 1
        if e < 0: e = len(raw)
        parts.append(" ".join(raw[s:e].split()))
    return "  ".join(parts)


def chunks(text: str, win: int, n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(0, len(text) - win), n))
    return [text[s:s + win] for s in starts]


def text_to_pattern(s: str, grid: int, n_gram: int = 4, k_per: int = 4
                    ) -> list[tuple[int, int]]:
    cells = set()
    s2 = s.lower()
    for i in range(max(1, len(s2) - n_gram + 1)):
        gram = s2[i:i + n_gram]
        for r in range(k_per):
            h = hashlib.blake2b(f"{gram}|{r}".encode(), digest_size=4).digest()
            v = int.from_bytes(h, "big") % (grid * grid)
            cells.add((v // grid, v % grid))
    return sorted(cells)


class TrigramKNN:
    def __init__(self): self.items = []
    def store(self, full):
        s2 = full.lower()
        tris = frozenset(s2[i:i + 3] for i in range(len(s2) - 2))
        self.items.append((tris, full))
    def recall(self, cue):
        c2 = cue.lower()
        ctris = frozenset(c2[i:i + 3] for i in range(len(c2) - 2))
        best, best_j = "", -1.0
        for tris, full in self.items:
            inter = len(ctris & tris)
            j = inter / max(1, len(ctris | tris))
            if j > best_j: best_j, best = j, full
        return best


class PrefixDict:
    def __init__(self, pl): self.tbl = {}; self.pl = pl
    def store(self, full): self.tbl[full[:self.pl]] = full
    def recall(self, cue): return self.tbl.get(cue[:self.pl], "")


class SubstrateTextMemory:
    def __init__(self, grid: int, seed: int = 0):
        self.grid = grid
        self.cfg = FieldConfig(rows=grid, cols=grid)
        self.pm = PatternMemory(
            cfg=self.cfg, seed=seed,
            write_lr=0.12, write_steps=20, write_decay=0.002,
            recall_steps=20,
        )
        self.full_texts: list[str] = []

    def store(self, full):
        self.full_texts.append(full)
        self.pm.store(text_to_pattern(full, self.grid))

    def recall(self, cue):
        idx, _, _ = self.pm.recall(text_to_pattern(cue, self.grid))
        return self.full_texts[idx] if 0 <= idx < len(self.full_texts) else ""


def eval_on(model, full_chunks, eval_idx, cue_len, noise_chars, seed):
    rng = random.Random(seed + 7)
    exact = 0; pc = 0; tc = 0
    for i in eval_idx:
        full = full_chunks[i]
        cue = list(full[:cue_len])
        for _ in range(noise_chars):
            j = rng.randrange(len(cue))
            cue[j] = chr(ord('a') + rng.randrange(26))
        rec = model.recall("".join(cue))
        exact += int(rec == full)
        suf_t = full[cue_len:]
        suf_p = rec[cue_len:cue_len + len(suf_t)]
        for a, b in zip(suf_t, suf_p):
            if a == b: pc += 1
        tc += len(suf_t)
    return exact / len(eval_idx), pc / max(1, tc)


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    print("=" * 78)
    print("EL TEXT MEMORY @ MASSIVE SCALE — 6 Gutenberg books concatenated")
    print("=" * 78)
    text = load_books(data_dir)
    print(f"[corpus] {len(text):,} chars from real Project Gutenberg books")

    win = 120
    cue_len = 80
    grid = 256
    sweep = [2000, 5000, 10000, 20000]
    n_eval = 200
    print(f"[setup] window={win}, cue={cue_len}, grid={grid}x{grid}={grid*grid} cells, "
          f"n_eval={n_eval}")

    rng = random.Random(123)

    print(f"\n{'N':>6} {'model':<22} {'fit_s':>7} {'eval_s':>7} "
          f"{'exact':>7} {'char%':>7} {'noisy_ex':>9} {'noisy_ch%':>10}")
    print("-" * 78)

    for N in sweep:
        cs = list(dict.fromkeys(chunks(text, win, N, seed=42)))
        N = len(cs)
        eval_idx = sorted(rng.sample(range(N), min(n_eval, N)))

        for cls, name in [(PrefixDict(cue_len), "prefix-dict"),
                          (TrigramKNN(),         "trigram-kNN"),
                          (SubstrateTextMemory(grid=grid), "substrate")]:
            t0 = time.time()
            for c in cs: cls.store(c)
            tfit = time.time() - t0

            t0 = time.time()
            ex0, ch0 = eval_on(cls, cs, eval_idx, cue_len, 0, 1)
            ex3, ch3 = eval_on(cls, cs, eval_idx, cue_len, 3, 2)
            tev = time.time() - t0

            print(f"{N:>6} {name:<22} {tfit:>7.1f} {tev:>7.1f} "
                  f"{ex0:>7.3f} {ch0:>7.3f} {ex3:>9.3f} {ch3:>10.3f}",
                  flush=True)
        print("-" * 78)


if __name__ == "__main__":
    main()
