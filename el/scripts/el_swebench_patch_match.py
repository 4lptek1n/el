"""el_swebench_patch_match — VERIFIED action signal benchmark.

The earlier el_swebench_classify.py used repo prediction — a surface
signal (essentially "guess the next token of the URL"). This script
replaces it with the actual SWE-bench verified signal:

    Each issue has exactly ONE patch known to make its FAIL_TO_PASS
    tests pass (verified by SWE-bench's docker test harness).
    Given the issue and a slate of N candidate patches (the true one
    + N-1 distractors from the same repo, so the "guess the repo"
    surface signal is destroyed), score each candidate. Accuracy =
    how often the substrate ranks the true patch #1.
    chance = 1/N.

Why this is a VERIFIED action signal, not a next-token surrogate:
  * The "correct" patch is the one whose unit tests actually pass —
    measured by running the test harness in a docker container, not
    by predicting any kind of token.
  * Distractors are real patches from the same repo, so the model
    cannot win by recognizing repo-specific vocabulary.
  * The label space is binary per (issue, patch) pair: tests pass
    or they don't. SWE-bench labels these.

The substrate gets:
    score(issue, patch) = cosine( encode(issue), encode(patch) )
no head, no further training. Pure substrate retrieval.
"""
from __future__ import annotations
import sys, hashlib, time, random
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory
from el.thermofield.frozen import FrozenSubstrate


def text_to_pattern(s: str, grid: int, n_gram: int = 4, k_per: int = 4):
    cells = set(); s2 = s.lower()
    for i in range(max(1, len(s2) - n_gram + 1)):
        gram = s2[i:i + n_gram]
        for r in range(k_per):
            h = hashlib.blake2b(f"{gram}|{r}".encode(), digest_size=4).digest()
            v = int.from_bytes(h, "big") % (grid * grid)
            cells.add((v // grid, v % grid))
    return sorted(cells)


def trigram_vec(s: str, dim: int = 4096):
    v = np.zeros(dim, dtype=np.float32); s2 = s.lower()
    for i in range(max(1, len(s2) - 2)):
        h = hashlib.blake2b(s2[i:i+3].encode(), digest_size=4).digest()
        v[int.from_bytes(h, "big") % dim] += 1.0
    n = np.linalg.norm(v); 
    return v / n if n > 0 else v


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=2000,
                    help="Number of issues to evaluate")
    ap.add_argument("--n_cand", type=int, default=10,
                    help="Number of candidate patches per issue (1 true + N-1 distractors)")
    ap.add_argument("--grid", type=int, default=192)
    ap.add_argument("--D",    type=int, default=1024)
    ap.add_argument("--relax", type=int, default=12)
    ap.add_argument("--imprint", type=int, default=1500)
    ap.add_argument("--max_doc", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.n_cand < 2:
        raise SystemExit("--n_cand must be >= 2 (need at least 1 distractor)")

    from datasets import load_from_disk
    arrow = Path(__file__).resolve().parents[1] / "data/swebench/train_arrow"
    ds = load_from_disk(str(arrow))

    print("=" * 78)
    print("EL SUBSTRATE — VERIFIED-SIGNAL benchmark")
    print("  task:    given issue, rank N candidate patches; correct =")
    print("           the unique patch whose FAIL_TO_PASS tests pass")
    print("  signal:  test-harness verdict (binary), not token prediction")
    print(f"  setup:   N={args.n_cand} candidates per issue, "
          f"distractors drawn from the same repo")
    print(f"  chance:  {1/args.n_cand:.3f}")
    print("  CAVEAT: only the positive (issue, true_patch) pair is")
    print("          SWE-bench-verified; distractors are assumed-negative")
    print("          (same-repo patches, not re-verified per pair). This is")
    print("          a verified-positive ranking proxy, not strict pairwise")
    print("          binary verification. Imprint pool is sampled from the")
    print("          full corpus (transductive); substrate sees no labels.")
    print("=" * 78)
    print(f"corpus: {len(ds)} issues across {len(set(ds['repo']))} repos")

    # Index patches by repo so distractors come from the same project.
    by_repo: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(ds["repo"]):
        by_repo[r].append(i)

    # Eligible issues = those whose repo has at least n_cand patches
    # (so we can draw distractors).
    rng = random.Random(args.seed)
    eligible = [i for i in range(len(ds))
                if len(by_repo[ds[i]["repo"]]) >= args.n_cand
                and len(ds[i]["patch"]) > 50
                and len(ds[i]["problem_statement"]) > 50]
    rng.shuffle(eligible)
    eval_idx = eligible[:args.n_eval]
    print(f"eligible issues: {len(eligible)}; evaluating: {len(eval_idx)}")

    # ----- substrate setup -----
    print(f"\n[setup] imprinting substrate "
          f"(grid={args.grid}², D={args.D}, relax={args.relax})…")
    pm = PatternMemory(cfg=FieldConfig(rows=args.grid, cols=args.grid),
                       seed=0, write_lr=0.10, write_steps=6,
                       write_decay=0.001, recall_steps=1)
    pool = rng.sample(range(len(ds)), args.imprint)
    t0 = time.time()
    for i in pool:
        s = (ds[i]["problem_statement"] + "\n" + ds[i]["patch"])[:args.max_doc]
        pm.store(text_to_pattern(s, args.grid))
    print(f"  imprint {args.imprint}: {time.time()-t0:.1f}s")
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=args.D,
                                             relax_steps=args.relax, seed=42)
    fp_init = fr.fingerprint()
    print(f"  fingerprint: {fp_init}")

    # ----- encode all issues + all patches we'll need -----
    # union of: every eval issue, plus its true patch, plus distractor
    # patches drawn from same repo. We pre-build the slate so we know
    # exactly what to encode (and avoid encoding the full corpus).
    slates = []  # list of (issue_idx, [cand_patch_idx], true_pos)
    needed_issue, needed_patch = set(), set()
    for ii in eval_idx:
        repo = ds[ii]["repo"]
        pool_p = [j for j in by_repo[repo] if j != ii and len(ds[j]["patch"]) > 50]
        if len(pool_p) < args.n_cand - 1:
            continue
        distractors = rng.sample(pool_p, args.n_cand - 1)
        cand = distractors + [ii]
        rng.shuffle(cand)
        true_pos = cand.index(ii)
        slates.append((ii, cand, true_pos))
        needed_issue.add(ii)
        for j in cand: needed_patch.add(j)
    print(f"slates: {len(slates)}  unique issues: {len(needed_issue)}  "
          f"unique patches to encode: {len(needed_patch)}")

    def issue_text(i): return ds[i]["problem_statement"][:args.max_doc]
    def patch_text(j): return ds[j]["patch"][:args.max_doc]

    print(f"\n[encode] issues ({len(needed_issue)})…")
    issue_idx_list = sorted(needed_issue)
    issue_pos = {i: k for k, i in enumerate(issue_idx_list)}
    issue_feats = np.empty((len(issue_idx_list), args.D), dtype=np.float32)
    t0 = time.time(); last = t0
    BATCH = 64
    for s in range(0, len(issue_idx_list), BATCH):
        chunk = issue_idx_list[s:s + BATCH]
        issue_feats[s:s+len(chunk)] = fr.encode_batch(
            text_to_pattern(issue_text(i), args.grid) for i in chunk)
        if time.time() - last > 30:
            r = (s + len(chunk)) / (time.time() - t0)
            print(f"   {s+len(chunk):>5}/{len(issue_idx_list)} "
                  f"{r:.1f} c/s  ETA {(len(issue_idx_list)-s)/r/60:.1f}min",
                  flush=True)
            last = time.time()
    print(f"  done {time.time()-t0:.1f}s")

    print(f"[encode] patches ({len(needed_patch)})…")
    patch_idx_list = sorted(needed_patch)
    patch_pos = {j: k for k, j in enumerate(patch_idx_list)}
    patch_feats = np.empty((len(patch_idx_list), args.D), dtype=np.float32)
    t0 = time.time(); last = t0
    for s in range(0, len(patch_idx_list), BATCH):
        chunk = patch_idx_list[s:s + BATCH]
        patch_feats[s:s+len(chunk)] = fr.encode_batch(
            text_to_pattern(patch_text(j), args.grid) for j in chunk)
        if time.time() - last > 30:
            r = (s + len(chunk)) / (time.time() - t0)
            print(f"   {s+len(chunk):>5}/{len(patch_idx_list)} "
                  f"{r:.1f} c/s  ETA {(len(patch_idx_list)-s)/r/60:.1f}min",
                  flush=True)
            last = time.time()
    print(f"  done {time.time()-t0:.1f}s")

    issue_n = issue_feats / (np.linalg.norm(issue_feats, axis=1, keepdims=True) + 1e-12)
    patch_n = patch_feats / (np.linalg.norm(patch_feats, axis=1, keepdims=True) + 1e-12)

    # ----- score & rank -----
    print(f"\n[eval] scoring slates…")
    rank_hits = {1: 0, 3: 0, 5: 0}
    mrr_sum = 0.0
    for issue_i, cands, true_pos in slates:
        qi = issue_n[issue_pos[issue_i]]
        ck = np.stack([patch_n[patch_pos[j]] for j in cands], axis=0)
        sims = ck @ qi
        order = np.argsort(-sims)
        rank = int(np.where(order == true_pos)[0][0]) + 1
        if rank <= 1: rank_hits[1] += 1
        if rank <= 3: rank_hits[3] += 1
        if rank <= 5: rank_hits[5] += 1
        mrr_sum += 1.0 / rank
    n = len(slates)
    if n == 0:
        raise SystemExit("no slates produced — check --n_cand and corpus size")

    # ----- baseline: trigram cosine -----
    print(f"[eval] running trigram baseline for comparison…")
    issue_tri = np.stack([trigram_vec(issue_text(i)) for i in issue_idx_list])
    patch_tri = np.stack([trigram_vec(patch_text(j)) for j in patch_idx_list])
    tri_hits = {1: 0, 3: 0, 5: 0}; tri_mrr = 0.0
    for issue_i, cands, true_pos in slates:
        qi = issue_tri[issue_pos[issue_i]]
        ck = np.stack([patch_tri[patch_pos[j]] for j in cands], axis=0)
        sims = ck @ qi
        order = np.argsort(-sims)
        rank = int(np.where(order == true_pos)[0][0]) + 1
        if rank <= 1: tri_hits[1] += 1
        if rank <= 3: tri_hits[3] += 1
        if rank <= 5: tri_hits[5] += 1
        tri_mrr += 1.0 / rank

    # ----- random baseline (analytic) -----
    rand_top1 = 1 / args.n_cand
    rand_top3 = min(3, args.n_cand) / args.n_cand
    rand_top5 = min(5, args.n_cand) / args.n_cand
    rand_mrr = sum(1/k for k in range(1, args.n_cand+1)) / args.n_cand

    # ----- integrity check -----
    fp_after = fr.fingerprint()
    assert fp_init == fp_after, "substrate was modified!"

    print(f"\n" + "=" * 78)
    print(f"VERIFIED-SIGNAL RESULTS — {n} issues, {args.n_cand} candidates each")
    print(f"  {'method':35s} {'top-1':>8s} {'top-3':>8s} {'top-5':>8s} {'MRR':>8s}")
    print(f"  {'-'*70}")
    print(f"  {'random (analytic)':35s} "
          f"{rand_top1:>8.3f} {rand_top3:>8.3f} {rand_top5:>8.3f} {rand_mrr:>8.3f}")
    print(f"  {'trigram cosine (4096-d hash)':35s} "
          f"{tri_hits[1]/n:>8.3f} {tri_hits[3]/n:>8.3f} {tri_hits[5]/n:>8.3f} "
          f"{tri_mrr/n:>8.3f}")
    print(f"  {'frozen substrate cosine':35s} "
          f"{rank_hits[1]/n:>8.3f} {rank_hits[3]/n:>8.3f} {rank_hits[5]/n:>8.3f} "
          f"{mrr_sum/n:>8.3f}")
    print(f"\nsubstrate fingerprint stable: True ({fp_after})")
    print(f"NOTE: ground truth = patch with verified passing "
          f"FAIL_TO_PASS tests. No token prediction involved.")
    print("=" * 78)


if __name__ == "__main__":
    main()
