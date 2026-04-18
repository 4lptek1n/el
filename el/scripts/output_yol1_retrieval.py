"""YOL 1 — RETRIEVAL-AS-OUTPUT.

Substrate retrieves the top-1 most similar known (issue, patch) and
returns the patch verbatim. NOT generation — copy-output. Honest.

Demo: 5 fresh queries, each gets:
  - top-1 cosine match in substrate feature space
  - the actual patch from that match (truncated)
  - cosine score (caller can threshold to refuse low-confidence)
"""
from __future__ import annotations
import sys, hashlib, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory
from el.thermofield.frozen import FrozenSubstrate


def text_to_pattern(s, grid, n_gram=4, k_per=4):
    cells = set(); s2 = s.lower()
    for i in range(max(1, len(s2) - n_gram + 1)):
        gram = s2[i:i + n_gram]
        for r in range(k_per):
            h = hashlib.blake2b(f"{gram}|{r}".encode(), digest_size=4).digest()
            v = int.from_bytes(h, "big") % (grid * grid)
            cells.add((v // grid, v % grid))
    return sorted(cells)


QUERIES = [
    "DataFrame.resample with closed='right' drops the first hour after a DST transition for tz-aware index",
    "BartTokenizer.add_tokens() does not update the embedding matrix when called after model load",
    "numpy boolean indexing of a structured array raises ValueError on Python 3.11",
    "pytest collects test files twice when conftest.py uses pytest_plugins inside a subdirectory",
    "scikit-learn StandardScaler.partial_fit overflows when feeding many small batches",
]


def main():
    GRID, D, RELAX = 192, 1024, 12
    IMPRINT = 1500
    MAX_DOC = 4000
    BATCH = 64
    K = 3

    from datasets import load_from_disk
    arrow = Path(__file__).resolve().parents[1] / "data/swebench/train_arrow"
    ds = load_from_disk(str(arrow))
    print("=" * 78)
    print("YOL 1 — RETRIEVAL-AS-OUTPUT (substrate copy-and-return)")
    print("=" * 78)
    print(f"corpus: {len(ds)} (issue, patch) pairs across "
          f"{len(set(ds['repo']))} repos")

    rng = np.random.default_rng(0)
    pool = rng.choice(len(ds), IMPRINT, replace=False)
    pm = PatternMemory(cfg=FieldConfig(rows=GRID, cols=GRID), seed=0,
                       write_lr=0.10, write_steps=6, write_decay=0.001,
                       recall_steps=1)
    print(f"\n[setup] imprint {IMPRINT}…")
    t0 = time.time()
    for i in pool:
        s = (ds[int(i)]["problem_statement"] + "\n" + ds[int(i)]["patch"])[:MAX_DOC]
        pm.store(text_to_pattern(s, GRID))
    print(f"  {time.time()-t0:.1f}s")
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=D,
                                             relax_steps=RELAX, seed=42)

    # Encode the whole corpus
    print(f"[encode] {len(ds)} docs (~{len(ds)/24/60:.1f} min)…")
    feats = np.empty((len(ds), D), dtype=np.float32)
    t0 = time.time(); last = t0
    for s in range(0, len(ds), BATCH):
        chunk = list(range(s, min(s + BATCH, len(ds))))
        feats[s:s+len(chunk)] = fr.encode_batch(
            text_to_pattern(ds[i]["problem_statement"][:MAX_DOC], GRID)
            for i in chunk)
        if time.time() - last > 30:
            r = (s + len(chunk)) / (time.time() - t0)
            print(f"   {s+len(chunk):>5}/{len(ds)} {r:.1f} c/s",
                  flush=True)
            last = time.time()
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    print(f"  done {time.time()-t0:.1f}s")

    # Run queries
    for qi, q in enumerate(QUERIES, 1):
        qf = fr.encode(text_to_pattern(q, GRID))
        qf = qf / (np.linalg.norm(qf) + 1e-12)
        sims = feats @ qf
        order = np.argsort(-sims)[:K]

        print(f"\n{'=' * 78}")
        print(f"QUERY {qi}: {q}")
        print("=" * 78)
        for rank, idx in enumerate(order, 1):
            row = ds[int(idx)]
            patch = row["patch"]
            patch_short = patch[:600] + ("…" if len(patch) > 600 else "")
            issue_short = row["problem_statement"][:200].replace("\n", " ")
            print(f"\n--- top-{rank}  cos={sims[idx]:.3f}  "
                  f"repo={row['repo']}  id={row['instance_id']}")
            print(f"  matched issue: {issue_short}…")
            print(f"  PATCH (verbatim, truncated):")
            for ln in patch_short.splitlines()[:18]:
                print(f"    {ln}")
        # honest confidence floor
        if sims[order[0]] < 0.55:
            print("\n  [refuse] top-1 cosine below 0.55 — substrate would not "
                  "answer this in production (no good neighbor found)")

    print("\n" + "=" * 78)
    print("VERDICT: YOL 1 returns ACTUAL patches verbatim from corpus.")
    print("It cannot answer novel questions — only echoes known ones.")
    print("Output quality = quality of nearest neighbor in feature space.")
    print("=" * 78)


if __name__ == "__main__":
    main()
