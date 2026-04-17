"""el_swebench_classify — frozen substrate + streaming ridge on SWE-bench.

Real software-engineering signal: 19,008 GitHub issues across 35 OSS
projects (pandas, numpy, qiskit, transformers, jax, ray, ...).
Issue text → repo prediction. Honest 35-way classification.

Why this is a real signal (not toy):
  * 37 MB problem-statement text + 294 MB unified-diff patch text
  * heavy class imbalance (pandas 27%, qiskit 7%, ...)  →  macro-F1
    is the honest metric; accuracy alone is gamed by majority guess
  * each issue is real engineering English (stack traces, bug repros,
    code blocks, links) — far harder than literary text classification
  * train/test split inside the SWE-bench train shard, stratified by
    repo (the official test split has disjoint repos so it cannot be
    used for closed-vocab repo prediction)

The architecture is unchanged from el_frozen_classify.py:
  text → text_to_pattern (char-ngram → cell hash)
       → FrozenSubstrate.encode (relax-only, write=False arrays)
       → StreamingRidge.partial_fit  (O(D²+D·K) sufficient stats)
       → Cholesky solve, analytic intercept
       → argmax(X·W + b)
"""
from __future__ import annotations
import sys, time, hashlib, random, gc, resource
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory
from el.thermofield.frozen import FrozenSubstrate
from el.thermofield.readout import StreamingRidge


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


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def stratified_split(labels: list[int], test_frac: float, seed: int):
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)
    rng = random.Random(seed)
    train_idx, test_idx = [], []
    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        n_te = max(1, int(round(len(idxs) * test_frac)))
        test_idx.extend(idxs[:n_te])
        train_idx.extend(idxs[n_te:])
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return train_idx, test_idx


def macro_f1(pred: np.ndarray, truth: np.ndarray, n_classes: int) -> float:
    f1s = []
    for k in range(n_classes):
        tp = int(((pred == k) & (truth == k)).sum())
        fp = int(((pred == k) & (truth != k)).sum())
        fn = int(((pred != k) & (truth == k)).sum())
        if tp == 0:
            f1s.append(0.0); continue
        prec = tp / (tp + fp); rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def trigram_centroid_baseline(train_docs, train_y, test_docs, test_y,
                               n_classes, dim=4096):
    """Cheap baseline: hashed trigram bag-of-words, per-class centroid,
    cosine-nearest. Uses the same hashing trick as the substrate input
    so the comparison is at the same vocabulary granularity."""
    def vec(s):
        v = np.zeros(dim, dtype=np.float32)
        s2 = s.lower()
        for i in range(max(1, len(s2) - 2)):
            h = hashlib.blake2b(s2[i:i + 3].encode(), digest_size=4).digest()
            v[int.from_bytes(h, "big") % dim] += 1.0
        n = np.linalg.norm(v)
        if n > 0: v /= n
        return v
    cents = np.zeros((n_classes, dim), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int64)
    for d, y in zip(train_docs, train_y):
        cents[y] += vec(d); counts[y] += 1
    for k in range(n_classes):
        if counts[k] > 0:
            cents[k] /= counts[k]
            n = np.linalg.norm(cents[k])
            if n > 0: cents[k] /= n
    pred = np.empty(len(test_docs), dtype=np.int64)
    for i, d in enumerate(test_docs):
        v = vec(d)
        pred[i] = int(np.argmax(cents @ v))
    return pred


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="Limit rows (0 = full 19008)")
    ap.add_argument("--grid", type=int, default=192)
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--relax", type=int, default=12)
    ap.add_argument("--imprint", type=int, default=1500)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--skip-trigram", action="store_true",
                    help="Skip the trigram baseline (slow on big N)")
    ap.add_argument("--max-doc", type=int, default=6000)
    args = ap.parse_args()

    from datasets import load_from_disk
    arrow = Path(__file__).resolve().parents[1] / "data/swebench/train_arrow"
    print("=" * 78)
    print("FROZEN SUBSTRATE + STREAMING RIDGE  on  SWE-bench (35 repos, 19k issues)")
    print("=" * 78)
    ds = load_from_disk(str(arrow))
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))
    print(f"loaded {len(ds)} rows from {arrow}")

    # ---- text + label
    repos = sorted(set(ds["repo"]))
    repo_to_y = {r: i for i, r in enumerate(repos)}
    docs = []
    for r in ds:
        # Concatenate problem statement + patch — patch is the loudest
        # repo signature (file paths, idioms). Truncate to keep encode
        # cost bounded per row.
        s = (r["problem_statement"] + "\n" + r["patch"])[:args.max_doc]
        docs.append(s)
    labels = [repo_to_y[r] for r in ds["repo"]]
    n_classes = len(repos)
    print(f"\n{n_classes} repos.  class distribution (top 5):")
    for r, n in Counter(ds["repo"]).most_common(5):
        print(f"  {r:40s} {n:>5}  ({n/len(ds):5.1%})")
    print(f"  ... tail repos: {Counter(ds['repo']).most_common()[-1]}")

    # ---- stratified 85/15 split
    tr, te = stratified_split(labels, test_frac=0.15, seed=0)
    print(f"\nsplit: train={len(tr)}  test={len(te)}  (stratified by repo)")

    # ---- baseline FIRST so user sees the bar
    print(f"\n[baseline] majority-class always = {Counter(labels).most_common(1)[0]}")
    maj = Counter(labels).most_common(1)[0][0]
    pred_maj = np.full(len(te), maj, dtype=np.int64)
    truth = np.array([labels[i] for i in te], dtype=np.int64)
    acc_maj = float((pred_maj == truth).mean())
    f1_maj = macro_f1(pred_maj, truth, n_classes)
    print(f"  majority baseline       acc={acc_maj:.3f}  macro-F1={f1_maj:.3f}")

    if not args.skip_trigram:
        print(f"\n[baseline] hashed-trigram centroid (dim=4096)…")
        t0 = time.time()
        pred_tri = trigram_centroid_baseline(
            [docs[i] for i in tr], [labels[i] for i in tr],
            [docs[i] for i in te], [labels[i] for i in te],
            n_classes, dim=4096,
        )
        acc_tri = float((pred_tri == truth).mean())
        f1_tri = macro_f1(pred_tri, truth, n_classes)
        print(f"  trigram centroid        acc={acc_tri:.3f}  macro-F1={f1_tri:.3f}  "
              f"({time.time()-t0:.1f}s)")
    else:
        acc_tri = f1_tri = float("nan")
        print(f"\n[baseline] trigram skipped (--skip-trigram)")

    # ---- substrate hyper-params (CLI-overridable)
    grid       = args.grid
    n_readout  = args.D
    relax      = args.relax
    n_imprint  = args.imprint
    ridge_lambda = args.lam
    print(f"\n[substrate] grid={grid}²  D={n_readout}  relax={relax}  "
          f"imprint_pool={n_imprint}  λ={ridge_lambda}")

    # ---- imprint substrate (unsupervised) on a sample of train
    rng = random.Random(0)
    pool_idx = rng.sample(tr, min(n_imprint, len(tr)))
    imprint_pool = [docs[i] for i in pool_idx]
    cfg = FieldConfig(rows=grid, cols=grid)
    pm = PatternMemory(cfg=cfg, seed=0,
                       write_lr=0.10, write_steps=6, write_decay=0.001,
                       recall_steps=1)
    print(f"[phase 1] imprinting {len(imprint_pool)} chunks (unsupervised)…")
    t0 = time.time()
    for c in imprint_pool:
        pm.store(text_to_pattern(c, grid))
    print(f"  imprint done in {time.time() - t0:.1f}s")

    print(f"\n[phase 2] freezing substrate…")
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=n_readout,
                                             relax_steps=relax, seed=42)
    fp_init = fr.fingerprint()
    print(f"  fingerprint = {fp_init}")
    del pm; gc.collect()

    # ---- stream encode + ridge
    print(f"\n[phase 3] streaming train through frozen+ridge "
          f"({len(tr)} chunks)…")
    ridge = StreamingRidge(n_features=n_readout, n_classes=n_classes,
                           ridge_lambda=ridge_lambda)
    BATCH = 64
    t0 = time.time(); last = t0
    for s in range(0, len(tr), BATCH):
        batch = tr[s:s + BATCH]
        feats = fr.encode_batch(text_to_pattern(docs[i], grid) for i in batch)
        ys = np.array([labels[i] for i in batch], dtype=np.int64)
        ridge.partial_fit(feats, ys)
        if time.time() - last > 30:
            done = s + len(batch); rate = done / (time.time() - t0)
            eta = (len(tr) - done) / rate
            print(f"   {done:>5}/{len(tr)}  {rate:.1f} c/s  ETA {eta/60:.1f}min  "
                  f"sub={(fr.C_right.nbytes+fr.C_down.nbytes+fr.B_right.nbytes+fr.B_down.nbytes)/1024:.0f}KB  "
                  f"ridge_A={ridge._A.nbytes/1024:.0f}KB  rss={rss_mb():.0f}MB",
                  flush=True)
            last = time.time()
    print(f"  train pass: {time.time()-t0:.1f}s")

    print(f"\n[phase 4] solve ridge…")
    t0 = time.time(); ridge.solve()
    print(f"  solved in {time.time()-t0:.2f}s")

    print(f"\n[phase 5] eval on {len(te)} held-out…")
    t0 = time.time()
    test_feats_chunks = []
    for s in range(0, len(te), BATCH):
        batch = te[s:s + BATCH]
        f = fr.encode_batch(text_to_pattern(docs[i], grid) for i in batch)
        test_feats_chunks.append(f)
    test_feats = np.vstack(test_feats_chunks)
    pred_sub = ridge.predict(test_feats)
    print(f"  eval encode in {time.time()-t0:.1f}s")
    acc_sub = float((pred_sub == truth).mean())
    f1_sub  = macro_f1(pred_sub, truth, n_classes)

    # ---- integrity check
    fp_after = fr.fingerprint()
    assert fp_init == fp_after, "substrate was modified during training!"

    # ---- summary table
    print(f"\n" + "=" * 78)
    print(f"RESULTS  ({len(te)} test issues, {n_classes} classes)")
    print(f"  {'classifier':45s} {'acc':>8s} {'macro-F1':>10s}")
    print(f"  {'-'*65}")
    print(f"  {'majority-class baseline':45s} {acc_maj:>8.3f} {f1_maj:>10.3f}")
    print(f"  {'hashed-trigram centroid (dim=4096)':45s} {acc_tri:>8.3f} {f1_tri:>10.3f}")
    print(f"  {'frozen substrate + streaming ridge':45s} {acc_sub:>8.3f} {f1_sub:>10.3f}")
    print(f"  {'random (1/35)':45s} {1/n_classes:>8.3f} {'~':>10s}")
    print(f"\nsubstrate fingerprint stable: {fp_init == fp_after} ({fp_after})")
    print(f"footprint: substrate={fr.C_right.nbytes*4/1024:.0f}KB  "
          f"ridge_state={(ridge._A.nbytes+ridge._B.nbytes)/1024/1024:.1f}MB  "
          f"rss={rss_mb():.0f}MB")
    print(f"  (ridge state is independent of N; would be the same after "
          f"streaming a billion issues)")
    print("=" * 78)


if __name__ == "__main__":
    main()
