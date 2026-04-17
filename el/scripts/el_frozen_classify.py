"""el_frozen_classify — streaming TB-scale book classification with
the FrozenSubstrate + StreamingRidge architecture.

Architecture:
  text chunk
     │  text_to_pattern (deterministic char-ngram hashing → cell indices)
     ▼
  cue (sparse cell list)
     │  FrozenSubstrate.encode (relax-only forward pass)
     ▼
  feat ∈ R^D                       ← D fixed, e.g. 512
     │  StreamingRidge.partial_fit  ← O(D²) memory, INDEPENDENT OF N
     ▼
  ridge sufficient stats (A=DxD, B=DxK)
     │  solve at the end (Cholesky on D×D)
     ▼
  classifier W ∈ R^{D×K}

The substrate is imprinted with Hebbian writes on a SEED batch of
training paragraphs (unsupervised — the substrate is just learning
the texture of the data so that diffusion bounces interestingly).
Then it is frozen forever; downstream learning happens only in W.

Memory cost is bounded by D (e.g., 512). The corpus can be terabytes;
encoder is stateless and feeds the ridge accumulator chunk-by-chunk.
"""
from __future__ import annotations

import sys, time, hashlib, random, gc, resource
from pathlib import Path
from typing import Iterator
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory
from el.thermofield.frozen import FrozenSubstrate
from el.thermofield.readout import StreamingRidge


BOOKS = [
    ("pride.txt",        "Austen — Pride & Prejudice"),
    ("moby.txt",         "Melville — Moby Dick"),
    ("sherlock.txt",     "Doyle — Sherlock Holmes"),
    ("frankenstein.txt", "Shelley — Frankenstein"),
    ("alice.txt",        "Carroll — Alice in Wonderland"),
    ("tomsawyer.txt",    "Twain — Tom Sawyer"),
]


def load_book(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    s, e = raw.find("*** START OF"), raw.find("*** END OF")
    if s >= 0: s = raw.find("\n", s) + 1
    if e < 0: e = len(raw)
    return " ".join(raw[s:e].split())


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


def chunk_iter(text: str, win: int, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(0, max(1, len(text) - win)), min(n, len(text) - win)))
    return [text[s:s + win] for s in starts]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def mem_footprint(fr: FrozenSubstrate, ridge: StreamingRidge) -> dict:
    sub_bytes = (fr.C_right.nbytes + fr.C_down.nbytes
                 + fr.B_right.nbytes + fr.B_down.nbytes
                 + fr.read_positions.nbytes)
    ridge_bytes = ridge._A.nbytes + ridge._B.nbytes + ridge._feat_mean.nbytes
    return {
        "substrate_KB": sub_bytes / 1024,
        "ridge_KB":    ridge_bytes / 1024,
        "process_MB":  rss_mb(),
    }


# ---------------------------------------------------------------------------

def imprint_substrate(grid: int, seed_chunks: list[str], seed: int = 0) -> PatternMemory:
    """Hebbian-imprint the substrate with a representative batch of patterns."""
    cfg = FieldConfig(rows=grid, cols=grid)
    pm = PatternMemory(
        cfg=cfg, seed=seed,
        write_lr=0.10, write_steps=6, write_decay=0.001,
        recall_steps=1,  # never used
    )
    for c in seed_chunks:
        pm.store(text_to_pattern(c, grid))
    return pm


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    grid       = 192
    win        = 200
    n_readout  = 512
    relax      = 12
    n_imprint  = 600   # how many chunks to imprint substrate with (unsupervised)
    n_train    = 4000  # training chunks per book? — actually total across books
    n_test     = 600
    ridge_lambda = 1.0

    print("=" * 78)
    print("FROZEN SUBSTRATE + STREAMING RIDGE READOUT")
    print(f"  grid={grid}×{grid}  D={n_readout}  relax_steps={relax}")
    print(f"  imprint=N={n_imprint}  train≈{n_train}  test≈{n_test}")
    print("=" * 78)

    # -------- Load corpus
    train_chunks, train_y = [], []
    test_chunks,  test_y  = [], []
    imprint_pool = []
    n_books = len(BOOKS)
    per_book_train = n_train // n_books
    per_book_test  = n_test // n_books
    for yi, (fn, name) in enumerate(BOOKS):
        text = load_book(data_dir / fn)
        cut = int(0.70 * len(text))
        tr = chunk_iter(text[:cut], win, per_book_train, seed=10 + yi)
        te = chunk_iter(text[cut:], win, per_book_test,  seed=200 + yi)
        train_chunks += tr; train_y += [yi] * len(tr)
        test_chunks  += te; test_y  += [yi] * len(te)
        imprint_pool += chunk_iter(text[:cut], win, n_imprint // n_books, seed=900 + yi)
        print(f"  [{name:<32}] {len(tr)} train + {len(te)} test "
              f"from {len(text):,} char")
    print(f"[total] train={len(train_chunks)}  test={len(test_chunks)}  "
          f"classes={n_books}")

    # -------- Phase 1: imprint substrate (unsupervised Hebb)
    print(f"\n[phase 1] Hebbian imprint on {len(imprint_pool)} chunks (unsupervised)…")
    t0 = time.time()
    pm = imprint_substrate(grid, imprint_pool, seed=0)
    imprint_t = time.time() - t0
    print(f"  imprint done in {imprint_t:.1f}s")

    # -------- Phase 2: freeze
    print(f"\n[phase 2] freezing substrate (n_readout={n_readout})…")
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=n_readout,
                                             relax_steps=relax, seed=42)
    fp_init = fr.fingerprint()
    print(f"  fingerprint = {fp_init}")
    del pm; gc.collect()

    # -------- Phase 3: stream-encode + partial_fit ridge
    print(f"\n[phase 3] streaming train through frozen substrate "
          f"+ ridge (λ={ridge_lambda})…")
    ridge = StreamingRidge(n_features=n_readout, n_classes=n_books,
                           ridge_lambda=ridge_lambda)
    order = list(range(len(train_chunks)))
    random.Random(7).shuffle(order)
    BATCH = 64
    t0 = time.time()
    last_print = t0
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        feats = fr.encode_batch(text_to_pattern(train_chunks[i], grid) for i in idx)
        ys = np.array([train_y[i] for i in idx], dtype=np.int64)
        ridge.partial_fit(feats, ys)
        if time.time() - last_print > 20:
            mem = mem_footprint(fr, ridge)
            print(f"   processed {s + len(idx):>5}/{len(order)}  "
                  f"({(s + len(idx)) / (time.time() - t0):.1f} chunk/s)  "
                  f"sub={mem['substrate_KB']:.1f} KB  "
                  f"ridge={mem['ridge_KB']:.1f} KB  "
                  f"rss={mem['process_MB']:.1f} MB",
                  flush=True)
            last_print = time.time()
    train_t = time.time() - t0
    print(f"  train pass done in {train_t:.1f}s "
          f"({len(order) / train_t:.1f} chunk/s)")

    # -------- Phase 4: solve ridge
    print(f"\n[phase 4] solve ridge…")
    t0 = time.time()
    ridge.solve()
    print(f"  solved in {time.time() - t0:.2f}s")

    # -------- Phase 5: evaluate
    print(f"\n[phase 5] held-out test eval…")
    test_feats = fr.encode_batch(text_to_pattern(c, grid) for c in test_chunks)
    pred = ridge.predict(test_feats)
    truth = np.asarray(test_y)
    overall = float((pred == truth).mean())
    print(f"\n[results]")
    print(f"  classifier                  ", end="")
    for _, n in BOOKS: print(f"{n.split(' — ')[0][:8]:>9}", end="")
    print(f"{'OVERALL':>10}")
    line = f"  frozen-substr + ridge       "
    for k in range(n_books):
        m = truth == k
        acc_k = float((pred[m] == truth[m]).mean()) if m.any() else 0.0
        line += f"{acc_k:>9.3f}"
    line += f"{overall:>10.3f}"
    print(line)

    # -------- Phase 6: integrity check
    print(f"\n[phase 6] integrity check (substrate must be UNTOUCHED)")
    fp_after = fr.fingerprint()
    assert fp_init == fp_after, "substrate was modified during training!"
    print(f"  fingerprint unchanged ✔  ({fp_after})")

    # -------- Phase 7: memory footprint summary
    mem = mem_footprint(fr, ridge)
    print(f"\n[memory]  substrate={mem['substrate_KB']:.1f} KB  "
          f"ridge_stats={mem['ridge_KB']:.1f} KB  "
          f"process_rss={mem['process_MB']:.1f} MB")
    n_total = len(order) + len(test_chunks)
    print(f"  ratio bytes-per-chunk-seen:  "
          f"{(mem['substrate_KB'] + mem['ridge_KB']) * 1024 / n_total:.1f} bytes/chunk")
    print(f"  IMPORTANT: ridge_KB is INDEPENDENT of N. The classifier "
          f"would consume the SAME bytes after a trillion chunks.")


if __name__ == "__main__":
    main()
