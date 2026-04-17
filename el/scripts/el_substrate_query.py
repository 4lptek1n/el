"""el_substrate_query — pose a novel coding/security problem to the
trained-but-frozen substrate and show what it actually does.

What this proves:
  * The substrate, even after seeing 19,008 real GitHub issues, has
    NO generative or reasoning capability. It is an associative
    memory + a linear classifier.
  * It CAN return: (a) the nearest known issues by feature similarity,
    (b) the predicted repo distribution from the trained ridge.
  * It CANNOT return: a fix, an explanation, a patch, a CVE writeup.

This is the honest boundary of the architecture — not a flaw, just
its definition. We display the boundary explicitly.
"""
from __future__ import annotations
import sys, hashlib, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory
from el.thermofield.frozen import FrozenSubstrate
from el.thermofield.readout import StreamingRidge


def text_to_pattern(s: str, grid: int, n_gram: int = 4, k_per: int = 4):
    cells = set(); s2 = s.lower()
    for i in range(max(1, len(s2) - n_gram + 1)):
        gram = s2[i:i + n_gram]
        for r in range(k_per):
            h = hashlib.blake2b(f"{gram}|{r}".encode(), digest_size=4).digest()
            v = int.from_bytes(h, "big") % (grid * grid)
            cells.add((v // grid, v % grid))
    return sorted(cells)


# Three problems the substrate has NEVER seen — different difficulty
# bands; all currently-unsolved-by-the-substrate by definition.
PROBLEMS = [
    {
        "name": "Z1 — synthetic zero-day style memory corruption",
        "text": """
SEGFAULT in libfoo 4.2.0 when parsing a malformed PNG with chunk
length 0xFFFFFFFE. The bounds check at line 219 of png_chunk_read
uses signed int32 arithmetic and overflows before the heap allocation,
producing a 0-byte alloc followed by a 4-billion-byte memcpy from the
attacker-controlled IDAT stream. Affects all Linux distros shipping
libfoo. CVE not yet assigned. Reproducer: run `pngfuzz --crash-only
crashes/00001.png` against any libfoo-linked process.
""",
    },
    {
        "name": "Z2 — pandas timezone DST off-by-one",
        "text": """
DataFrame.resample('1H', closed='right', label='right') drops the
first hour after a spring-forward DST transition when the index is
tz-aware. Reproduce:

    idx = pd.date_range('2024-03-09', periods=48, freq='H',
                        tz='America/New_York')
    df  = pd.DataFrame({'x': range(48)}, index=idx)
    df.resample('1H', closed='right', label='right').sum()

Expected: 48 rows in result. Actual: 47 rows, 2024-03-10 03:00 row
missing. Bug introduced between pandas 2.1 and 2.2.
""",
    },
    {
        "name": "Z3 — torch grad NaN under autocast bf16",
        "text": """
torch.cuda.amp.autocast(dtype=torch.bfloat16) silently produces NaN
gradients in nn.LayerNorm when the input has more than 2^15 elements
on the last dim. Only triggers on Hopper GPUs with cublasLt path.
Loss appears finite forward-pass but loss.backward() fills .grad with
NaN. Workaround: force fp32 in the LayerNorm. Confirmed on torch
2.4.0+cu124, H100, CUDA 12.4.
""",
    },
]


def main():
    from datasets import load_from_disk
    arrow = Path(__file__).resolve().parents[1] / "data/swebench/train_arrow"
    ds = load_from_disk(str(arrow))

    grid, D, relax = 192, 1024, 12
    print("=" * 78)
    print("EL SUBSTRATE — interactive query (honest boundary demo)")
    print("=" * 78)
    print(f"corpus:  19,008 GitHub issues across 35 OSS projects")
    print(f"model:   FrozenSubstrate(grid={grid}², D={D}, relax={relax})")
    print(f"         + StreamingRidge(35-class, λ=1.0)")
    print()
    print("This script poses three NOVEL problems — none of them appear")
    print("verbatim in the corpus. We then ask the substrate two questions:")
    print("  Q1: which 5 known issues are most similar in your feature space?")
    print("  Q2: which repo would your trained ridge classify this under?")
    print("THE SUBSTRATE DOES NOT WRITE A PATCH. IT HAS NO GENERATIVE HEAD.")
    print("=" * 78)

    # Reload corpus features. We re-imprint + re-stream from scratch so
    # this script is self-contained — but the substrate state ends up
    # identical (deterministic seeds), and we encode all 19k corpus
    # docs so we can do nearest-neighbor retrieval on their features.
    repos = sorted(set(ds["repo"]))
    print(f"\n[setup] imprinting substrate on a 1500-doc unsupervised pool…")
    pm = PatternMemory(cfg=FieldConfig(rows=grid, cols=grid), seed=0,
                       write_lr=0.10, write_steps=6, write_decay=0.001,
                       recall_steps=1)
    import random
    pool = random.Random(0).sample(range(len(ds)), 1500)
    t0 = time.time()
    for i in pool:
        s = (ds[i]["problem_statement"] + "\n" + ds[i]["patch"])[:4000]
        pm.store(text_to_pattern(s, grid))
    print(f"  done in {time.time()-t0:.1f}s")

    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=D,
                                             relax_steps=relax, seed=42)
    print(f"  fingerprint: {fr.fingerprint()}")

    # We need the whole corpus encoded for retrieval. That's the slow part
    # (~10 min), so we cache to disk on first run.
    cache = Path("/tmp/swe_corpus_feats.npz")
    if cache.exists():
        z = np.load(cache)
        feats = z["feats"]
        print(f"[setup] loaded cached corpus features {feats.shape}")
    else:
        print(f"[setup] encoding {len(ds)} corpus docs (one-time, ~10 min)…")
        feats = np.empty((len(ds), D), dtype=np.float32)
        BATCH = 64
        t0 = time.time(); last = t0
        for s in range(0, len(ds), BATCH):
            chunk = list(range(s, min(s + BATCH, len(ds))))
            feats[chunk] = fr.encode_batch(
                text_to_pattern((ds[i]["problem_statement"] + "\n" + ds[i]["patch"])[:4000], grid)
                for i in chunk
            )
            if time.time() - last > 30:
                rate = (s + len(chunk)) / (time.time() - t0)
                eta = (len(ds) - s) / rate
                print(f"   {s+len(chunk):>5}/{len(ds)}  {rate:.1f} c/s  "
                      f"ETA {eta/60:.1f}min", flush=True)
                last = time.time()
        np.savez(cache, feats=feats)
        print(f"  cached → {cache}")

    # L2-normalize for cosine
    fnorm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

    # Train the ridge on the full corpus so Q2 has a real classifier
    print(f"\n[setup] training ridge on full 19,008 corpus…")
    repo_to_y = {r: i for i, r in enumerate(repos)}
    y = np.array([repo_to_y[r] for r in ds["repo"]], dtype=np.int64)
    ridge = StreamingRidge(n_features=D, n_classes=len(repos), ridge_lambda=1.0)
    ridge.partial_fit(feats, y)
    ridge.solve()
    print(f"  ridge solved (D={D}, K={len(repos)})")

    # ----- query each problem -----
    for prob in PROBLEMS:
        print("\n" + "─" * 78)
        print(prob["name"])
        print("─" * 78)
        print(prob["text"].strip()[:300] + ("…" if len(prob["text"]) > 300 else ""))

        q = fr.encode(text_to_pattern(prob["text"], grid))
        qn = q / (np.linalg.norm(q) + 1e-12)
        sims = fnorm @ qn  # cosine
        top5 = np.argsort(-sims)[:5]

        print(f"\n  Q1 — nearest 5 known issues by substrate feature cosine:")
        for rank, idx in enumerate(top5, 1):
            r = ds[int(idx)]
            ttl = r["problem_statement"].strip().splitlines()[0][:90]
            print(f"   {rank}. cos={sims[idx]:.3f}  [{r['repo']}]  {ttl}")

        scores = ridge.predict_scores(q.reshape(1, -1))[0]
        top3 = np.argsort(-scores)[:3]
        print(f"\n  Q2 — trained-ridge repo classification (top 3):")
        for rank, k in enumerate(top3, 1):
            print(f"   {rank}. {repos[k]:<40s}  score={scores[k]:+.3f}")

        # Honest verdict
        print(f"\n  [honest verdict]")
        print(f"   The substrate returned RETRIEVAL + CLASSIFICATION only.")
        print(f"   It did NOT propose a fix, locate the bug, or explain the")
        print(f"   crash. The architecture has no generative head and no")
        print(f"   causal-reasoning module. What you see above IS its full")
        print(f"   answer.")

    print("\n" + "=" * 78)
    print("END — substrate fingerprint after all queries:", fr.fingerprint())
    print("(should be identical to setup fingerprint — frozen weights)")
    print("=" * 78)


if __name__ == "__main__":
    main()
