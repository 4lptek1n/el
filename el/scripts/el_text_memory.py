"""el_text_memory — substrate as a LARGE-SCALE content-addressable memory
on REAL English text (Pride and Prejudice, Project Gutenberg).

This is the LLM-equivalent test: prompt completion via associative recall.
Store thousands of distinct text windows. Cue with the prefix. Recall the
complete window. Measure accuracy at scale.

No synthetic. No mocks. Real Gutenberg text. Honest numbers.
"""
from __future__ import annotations
import sys, time, hashlib, random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory


# ---------- corpus ----------
def load_corpus(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # strip Gutenberg header/footer roughly
    start = raw.find("*** START OF")
    end = raw.find("*** END OF")
    if start >= 0:
        start = raw.find("\n", start) + 1
    if end < 0:
        end = len(raw)
    body = raw[start:end]
    # collapse whitespace, keep printable
    body = " ".join(body.split())
    return body


def chunk(text: str, win: int, n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(0, len(text) - win), n))
    return [text[s:s + win] for s in starts]


# ---------- encoder: char-trigram hashing into a sparse cell pattern ----------
def text_to_pattern(s: str, grid: int, n_gram: int = 4, k_per: int = 4
                    ) -> list[tuple[int, int]]:
    """Char n-gram → k hashed cells per gram. No position info: position
    binning empirically destroyed convergence (cue & full hit different
    bins, no shared cells). Bag-of-ngrams works better for substrate."""
    cells = set()
    s2 = s.lower()
    for i in range(max(1, len(s2) - n_gram + 1)):
        gram = s2[i:i + n_gram]
        for r in range(k_per):
            h = hashlib.blake2b(f"{gram}|{r}".encode(),
                                digest_size=4).digest()
            v = int.from_bytes(h, "big") % (grid * grid)
            cells.add((v // grid, v % grid))
    return sorted(cells)


# ---------- baselines ----------
class TrigramKNN:
    """Brute-force nearest neighbor over trigram-set Jaccard similarity.
    This is an honest strong baseline — it has full access to the stored
    text, no compression. Substrate stores patterns in a fixed-size field."""
    def __init__(self): self.items: list[tuple[set[str], str]] = []
    def store(self, full: str):
        s2 = full.lower()
        tris = {s2[i:i + 3] for i in range(len(s2) - 2)}
        self.items.append((tris, full))
    def recall(self, cue: str) -> str:
        c2 = cue.lower()
        ctris = {c2[i:i + 3] for i in range(len(c2) - 2)}
        best, best_j = "", -1.0
        for tris, full in self.items:
            inter = len(ctris & tris)
            j = inter / max(1, len(ctris | tris))
            if j > best_j: best_j, best = j, full
        return best


class PrefixDict:
    """Exact-prefix lookup. Loses if cue has any noise."""
    def __init__(self, prefix_len: int): self.tbl = {}; self.pl = prefix_len
    def store(self, full: str):
        self.tbl[full[:self.pl]] = full
    def recall(self, cue: str) -> str:
        return self.tbl.get(cue[:self.pl], "")


# ---------- substrate ----------
class SubstrateTextMemory:
    def __init__(self, grid: int, max_store: int, seed: int = 0):
        self.grid = grid
        self.cfg = FieldConfig(rows=grid, cols=grid)
        self.pm = PatternMemory(
            cfg=self.cfg, seed=seed,
            write_lr=0.12, write_steps=20, write_decay=0.002,
            recall_steps=20,
        )
        self.full_texts: list[str] = []
        self.max_store = max_store

    def store(self, full: str):
        self.full_texts.append(full)
        self.pm.store(text_to_pattern(full, self.grid))

    def recall(self, cue: str) -> str:
        idx, _, _ = self.pm.recall(text_to_pattern(cue, self.grid))
        if 0 <= idx < len(self.full_texts):
            return self.full_texts[idx]
        return ""


# ---------- evaluation ----------
def eval_recall(name: str, model, chunks: list[str], cue_len: int,
                noise_chars: int = 0, seed: int = 0) -> dict:
    """Cue = first cue_len chars (with optional char-substitution noise)."""
    rng = random.Random(seed + 7)
    exact = 0; partial_chars = 0; total_chars = 0
    t0 = time.time()
    for full in chunks:
        cue = list(full[:cue_len])
        if noise_chars > 0:
            for _ in range(noise_chars):
                i = rng.randrange(len(cue))
                cue[i] = chr(ord('a') + rng.randrange(26))
        cue = "".join(cue)
        rec = model.recall(cue)
        exact += int(rec == full)
        # char-level overlap on the SUFFIX (the part we had to recall)
        suffix_true = full[cue_len:]
        suffix_pred = rec[cue_len:cue_len + len(suffix_true)]
        for a, b in zip(suffix_true, suffix_pred):
            if a == b: partial_chars += 1
        total_chars += len(suffix_true)
    dt = time.time() - t0
    return {"name": name, "exact": exact / len(chunks),
            "char_acc_suffix": partial_chars / max(1, total_chars),
            "n": len(chunks), "time": dt}


def main():
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "pride.txt"
    print("=" * 78)
    print("EL TEXT MEMORY — Pride and Prejudice (Project Gutenberg)")
    print("=" * 78)
    text = load_corpus(corpus_path)
    print(f"[corpus] {len(text):,} chars")

    win = 120
    cue_len = 80   # 2/3 of pattern as cue (LLM-style: long prompt -> completion)
    sweep = [25, 50, 100, 200, 500]   # capacity sweep — find substrate's limit
    grid = 128
    print(f"[setup] window={win} chars, cue={cue_len} chars, grid={grid}x{grid}")

    results = []
    for N in sweep:
        chunks = chunk(text, win=win, n=N, seed=42)
        # dedupe (rare but safe)
        chunks = list(dict.fromkeys(chunks))
        if len(chunks) < N:
            print(f"  [warn] dedup reduced N {N}->{len(chunks)}")
        N = len(chunks)
        print(f"\n--- capacity = {N} chunks ---")

        models = {
            "prefix-dict (exact)": PrefixDict(prefix_len=cue_len),
            "trigram-kNN (full text)": TrigramKNN(),
            "substrate (pattern memory)": SubstrateTextMemory(grid=grid,
                                                              max_store=N),
        }
        for name, m in models.items():
            t0 = time.time()
            for c in chunks: m.store(c)
            tfit = time.time() - t0
            r = eval_recall(name, m, chunks, cue_len=cue_len, noise_chars=0)
            r["fit"] = tfit; r["N"] = N; r["noise"] = 0
            results.append(r)
            print(f"  {name:<28} fit={tfit:5.1f}s  exact={r['exact']:.3f}  "
                  f"char-acc(suffix)={r['char_acc_suffix']:.3f}  "
                  f"recall={r['time']:.1f}s")

        # noisy cue test (3 char substitutions in cue)
        print(f"  [noisy cue: 3 random char substitutions in {cue_len}-char cue]")
        for name, m in models.items():
            r = eval_recall(name + " [noisy]", m, chunks,
                            cue_len=cue_len, noise_chars=3, seed=1)
            r["N"] = N; r["noise"] = 3
            results.append(r)
            print(f"  {name:<28}                     exact={r['exact']:.3f}  "
                  f"char-acc(suffix)={r['char_acc_suffix']:.3f}")

    # --- final table ---
    print("\n" + "=" * 78)
    print("CAPACITY SWEEP — exact recall rate")
    print("=" * 78)
    print(f"{'N':>6}  {'noise':>5}  {'prefix-dict':>12}  {'trigram-kNN':>12}  "
          f"{'substrate':>12}")
    for N in sweep:
        for noise in (0, 3):
            row = {r["name"].split(" [")[0]: r for r in results
                   if r["N"] == N and r["noise"] == noise}
            pd_ = row.get("prefix-dict (exact)", {}).get("exact", float('nan'))
            kn = row.get("trigram-kNN (full text)", {}).get("exact", float('nan'))
            sb = row.get("substrate (pattern memory)", {}).get("exact", float('nan'))
            print(f"{N:>6}  {noise:>5}  {pd_:>12.3f}  {kn:>12.3f}  {sb:>12.3f}")


if __name__ == "__main__":
    main()
