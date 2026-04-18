"""YOL 4 — ATTRACTOR-BASED GENERATION (substrate-native, NO decoder).

The previous three OUTPUT paths each had a defect that mis-tested the
substrate's actual generative mechanism:

  Yol 1 (retrieval)     — copy verbatim, not generation.
  Yol 2 (hybrid LSTM)   — backprop'd decoder, substrate is just a feature
                          extractor; not a substrate-native generator.
  Yol 3 (recall→bytes)  — abused PatternMemory.recall (set-based Jaccard
                          top-N) which is structurally incapable of
                          recovering dense bit-exact byte sequences.

Yol 4 tests the PROPER substrate-native generative mechanism: Hopfield-
style associative completion via field dynamics.

PROTOCOL
========
1. Split the grid into two zones along the row axis:
     - top half rows  → ISSUE ZONE  (cue lives here)
     - bottom half rows → PATCH ZONE (completion lives here)

2. Encode each (issue_text, patch_text) into a joint sparse pattern:
     - issue → ~K_issue active cells in the issue zone (hash of n-grams)
     - patch → ~K_patch active cells in the patch zone
   Total joint pattern = K_issue + K_patch active cells.

3. Imprint K_train joint patterns via Hebbian write
   (`PatternMemory.store`). The Hebb rule strengthens C-edges between
   co-active cells, including cross-zone edges that bind issue cells to
   patch cells. THIS is the associative substrate of generation.

4. EVALUATE on K_eval HELD-OUT (issue, patch) pairs:
     - Inject ONLY the issue-zone cells of the eval cue (no clamp →
       initial condition only).
     - Relax the field for `recall_steps` steps with no plasticity.
     - Read the top-K_patch hottest cells WITHIN THE PATCH ZONE.
     - That set is the substrate's "generated" patch encoding.

5. METRICS (vs the eval pair's TRUE patch encoding):
     - precision  = |gen ∩ true| / |gen|
     - recall     = |gen ∩ true| / |true|
     - jaccard    = |gen ∩ true| / |gen ∪ true|
     - lift_vs_random  = jaccard / chance_jaccard
   Plus baselines:
     - random:   pick K_patch patch-zone cells uniformly
     - mean:     pick the K_patch most-frequently-hot cells across all
                 imprinted patch encodings (frequency prior)
     - oracle:   the true encoding itself (upper bound = 1.0)
     - retrieve: nearest-neighbour over imprinted issue encodings,
                 return that pair's patch encoding (memory upper bound
                 — substrate must beat random AND get close to retrieve)

If lift_vs_random >> 1 AND recall is meaningfully above the frequency-
prior baseline, the substrate is performing genuine non-LLM, no-decoder
associative completion. That is the cleanest possible evidence of
substrate-native generation.

CAVEAT (honest)
===============
This is bit-set completion in the patch ENCODING space, NOT byte-exact
patch text. The encoding is a sparse hash (lossy by construction) so we
can't decode it back to the original characters. What we CAN claim if
metrics pass: the substrate associates "this kind of issue" with "this
kind of patch encoding" — a real, falsifiable, substrate-native
generative signal that does not depend on any backprop'd component.
"""
from __future__ import annotations
import sys, hashlib, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------
def encode_interleaved(text: str, *, grid: int, k_active: int,
                       parity: int, n_gram: int = 4,
                       salt: str = "") -> list[tuple[int, int]]:
    """Hash text n-grams into ~k_active cells whose (row+col) % 2 == parity.

    Interleaving issue (parity=0) and patch (parity=1) cells means every
    issue cell has 4 neighboring patch cells (and vice-versa) — Hebb on
    nearest-neighbor C-edges can therefore form direct issue↔patch
    associations, which is impossible when the two are zonally separated.
    """
    cells: set[tuple[int, int]] = set()
    s = text.lower()
    if not s:
        return []
    n_target_cells = (grid * grid) // 2  # half the grid is reachable
    slots_per = max(1, k_active // max(1, len(s) - n_gram + 1) + 1)
    tries = 0
    for i in range(max(1, len(s) - n_gram + 1)):
        gram = s[i:i + n_gram]
        for r in range(slots_per * 4):  # extra rolls because parity halves space
            h = hashlib.blake2b(
                f"{salt}|{gram}|{r}".encode(), digest_size=4
            ).digest()
            v = int.from_bytes(h, "big") % (grid * grid)
            row, col = v // grid, v % grid
            if (row + col) % 2 == parity:
                cells.add((row, col))
            tries += 1
            if len(cells) >= k_active:
                break
        if len(cells) >= k_active:
            break
    return sorted(cells)


def encode_zone(text: str, *, n_rows: int, n_cols: int,
                row_offset: int, k_active: int,
                n_gram: int = 4, salt: str = "") -> list[tuple[int, int]]:
    """[Legacy zonal encoding — kept for reference.]
    Hash text n-grams into ~k_active cells in a sub-grid."""
    cells: set[tuple[int, int]] = set()
    s = text.lower()
    if not s:
        return []
    # Multiple hash slots per n-gram so length doesn't dominate sparsity.
    slots_per = max(1, k_active // max(1, len(s) - n_gram + 1) + 1)
    for i in range(max(1, len(s) - n_gram + 1)):
        gram = s[i:i + n_gram]
        for r in range(slots_per):
            h = hashlib.blake2b(
                f"{salt}|{gram}|{r}".encode(), digest_size=4
            ).digest()
            v = int.from_bytes(h, "big") % (n_rows * n_cols)
            cells.add((row_offset + v // n_cols, v % n_cols))
            if len(cells) >= k_active:
                break
        if len(cells) >= k_active:
            break
    return sorted(cells)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def metrics(gen: set, true: set) -> dict:
    inter = len(gen & true)
    return dict(
        precision=inter / max(1, len(gen)),
        recall=inter / max(1, len(true)),
        jaccard=inter / max(1, len(gen | true)),
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # ---- Hyperparams (kept small enough to run in <10 min) ----
    GRID = 64                        # 64×64 = 4096 cells
    K_ISSUE = 60                     # ~1.5% on parity-0 sublattice
    K_PATCH = 60                     # ~1.5% on parity-1 sublattice
    N_TRAIN = 400
    N_EVAL = 100
    MAX_TXT = 4000
    RELAX_STEPS = 18
    WRITE_STEPS = 6
    WRITE_LR = 0.08
    WRITE_DECAY = 0.008              # narrow sweet spot, found empirically
    WTA_K = 120                      # ≈ K_ISSUE + K_PATCH

    print("=" * 78)
    print("YOL 4 — ATTRACTOR-BASED ASSOCIATIVE GENERATION  (interleaved)")
    print("=" * 78)
    print(f"  grid={GRID}×{GRID}  issue=parity-0 sublattice  "
          f"patch=parity-1 sublattice")
    print(f"  k_issue={K_ISSUE}  k_patch={K_PATCH}  "
          f"sparsity≈{K_ISSUE/(GRID*GRID/2):.3f}")
    print(f"  N_train={N_TRAIN}  N_eval={N_EVAL}  "
          f"recall_steps={RELAX_STEPS}  wta_k={WTA_K}")

    # ---- Load corpus ----
    from datasets import load_from_disk
    arrow = Path(__file__).resolve().parents[1] / "data/swebench/train_arrow"
    ds = load_from_disk(str(arrow))
    print(f"  corpus: {len(ds)} pairs")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ds))
    train_idx = perm[:N_TRAIN]
    eval_idx = perm[N_TRAIN:N_TRAIN + N_EVAL]

    # ---- Encode ALL pairs (train + eval) ----
    def encode_pair(i):
        ex = ds[int(i)]
        issue = (ex["problem_statement"] or "")[:MAX_TXT]
        patch = (ex["patch"] or "")[:MAX_TXT]
        i_cells = encode_interleaved(
            issue, grid=GRID, k_active=K_ISSUE, parity=0, salt="issue"
        )
        p_cells = encode_interleaved(
            patch, grid=GRID, k_active=K_PATCH, parity=1, salt="patch"
        )
        return i_cells, p_cells

    print(f"\n[encode] {N_TRAIN + N_EVAL} pairs…")
    t0 = time.time()
    train_pairs = [encode_pair(i) for i in train_idx]
    eval_pairs = [encode_pair(i) for i in eval_idx]
    print(f"  {time.time()-t0:.1f}s")

    # Sanity: actual sparsity achieved
    avg_ki = np.mean([len(i) for i, _ in train_pairs])
    avg_kp = np.mean([len(p) for _, p in train_pairs])
    print(f"  avg active: issue={avg_ki:.1f}  patch={avg_kp:.1f}")

    # ---- Build substrate, imprint joint patterns ----
    pm = PatternMemory(
        cfg=FieldConfig(rows=GRID, cols=GRID),
        seed=0,
        write_steps=WRITE_STEPS,
        write_lr=WRITE_LR,
        write_decay=WRITE_DECAY,
        recall_steps=RELAX_STEPS,
        wta_k=WTA_K,
        wta_suppression=0.5,
    )
    print(f"\n[imprint] {N_TRAIN} joint (issue ⊕ patch) patterns…")
    t0 = time.time()
    for i_cells, p_cells in train_pairs:
        joint = sorted(set(i_cells) | set(p_cells))
        pm.store(joint)
    fp_train = (
        float(pm.field.C_right.mean()), float(pm.field.C_down.mean()),
        float(pm.field.C_right.std()), float(pm.field.C_down.std()),
    )
    print(f"  {time.time()-t0:.1f}s   "
          f"C_right_mean={fp_train[0]:.3f}  C_down_mean={fp_train[1]:.3f}")

    # ---- Build "frequency prior" baseline (parity-1 sublattice only) ----
    freq = np.zeros((GRID, GRID), dtype=np.int32)
    for _, p_cells in train_pairs:
        for r, c in p_cells:
            freq[r, c] += 1
    # Mask out parity-0 cells (which can never be patch)
    parity_grid = (np.add.outer(np.arange(GRID), np.arange(GRID)) % 2)
    freq_masked = np.where(parity_grid == 1, freq, -1)
    top_freq_idx = np.argpartition(freq_masked.ravel(), -K_PATCH)[-K_PATCH:]
    freq_baseline_set = set(
        (int(i // GRID), int(i % GRID)) for i in top_freq_idx
    )

    # ---- Eval ----
    print(f"\n[eval] attractor completion on {N_EVAL} held-out pairs…")
    rows = []
    f = pm.field
    # Patch lives on parity-1 sublattice (≈ half the grid)
    parity_grid = (np.add.outer(np.arange(GRID), np.arange(GRID)) % 2)
    pz_mask_2d = (parity_grid == 1)
    pz_indices = np.flatnonzero(pz_mask_2d.ravel())
    n_pz_cells = len(pz_indices)

    # Pre-compute random baseline expected jaccard
    chance = K_PATCH / n_pz_cells  # density of random pick in patch zone
    chance_jaccard = chance / (2 - chance)  # E[|A∩B|/|A∪B|] for two K-of-N sets

    t0 = time.time()
    for ei, (i_cells, true_p_cells) in enumerate(eval_pairs):
        # ---- Substrate completion ----
        f.reset_temp()
        # NOTE: do NOT clamp the cue. Inject as initial condition; let
        # the field's stored Hebb edges propagate heat into the patch
        # zone. Clamping would just keep the issue zone hot indefinitely
        # without forcing the network to use its associative weights.
        f.inject(i_cells, [1.0] * len(i_cells), clamp=False)
        for _ in range(RELAX_STEPS):
            f.step()

        # Read top-K_patch hottest cells RESTRICTED to parity-1 sublattice
        T_full = f.T.copy()
        T_pz_only = np.where(pz_mask_2d, T_full, -1.0).ravel()
        if T_pz_only.max() <= 0:
            gen_set: set[tuple[int, int]] = set()
        else:
            top_idx = np.argpartition(T_pz_only, -K_PATCH)[-K_PATCH:]
            gen_set = set(
                (int(i // GRID), int(i % GRID)) for i in top_idx
            )

        # ---- Random baseline: K_patch random parity-1 cells ----
        rng_e = np.random.default_rng(1000 + ei)
        chosen = rng_e.choice(pz_indices, K_PATCH, replace=False)
        rand_set = set(
            (int(i // GRID), int(i % GRID)) for i in chosen
        )

        # ---- Retrieval upper bound: nearest train-pair issue → its patch ----
        true_i_set = set(i_cells)
        best_j, best_score = -1, -1.0
        for j, (i_cells_j, _) in enumerate(train_pairs):
            sc = jaccard(true_i_set, set(i_cells_j))
            if sc > best_score:
                best_score = sc
                best_j = j
        retr_set = set(train_pairs[best_j][1])

        true_set = set(true_p_cells)
        rows.append({
            "substrate": metrics(gen_set, true_set),
            "random":    metrics(rand_set, true_set),
            "freq":      metrics(freq_baseline_set, true_set),
            "retrieve":  metrics(retr_set, true_set),
        })

    f.reset_temp()
    fp_after = (
        float(pm.field.C_right.mean()), float(pm.field.C_down.mean()),
        float(pm.field.C_right.std()), float(pm.field.C_down.std()),
    )
    print(f"  {time.time()-t0:.1f}s")
    print(f"  fingerprint stable: {fp_train == fp_after}  (no eval mutation)")

    # ---- Aggregate ----
    def agg(key, metric):
        return float(np.mean([r[key][metric] for r in rows]))

    print("\n" + "=" * 78)
    print("YOL 4 RESULTS  (mean over {} held-out pairs)".format(N_EVAL))
    print("=" * 78)
    header = f"  {'method':<12}  {'precision':>9}  {'recall':>9}  {'jaccard':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in ("random", "freq", "substrate", "retrieve"):
        print(f"  {key:<12}  "
              f"{agg(key, 'precision'):>9.3f}  "
              f"{agg(key, 'recall'):>9.3f}  "
              f"{agg(key, 'jaccard'):>9.3f}")

    sub_j = agg("substrate", "jaccard")
    rand_j = agg("random", "jaccard")
    freq_j = agg("freq", "jaccard")
    retr_j = agg("retrieve", "jaccard")
    print(f"\n  chance_jaccard (closed-form): {chance_jaccard:.4f}")
    print(f"  lift vs random:  {sub_j / max(1e-9, rand_j):.2f}×")
    print(f"  lift vs freq:    {sub_j / max(1e-9, freq_j):.2f}×")
    print(f"  fraction of retrieval upper bound: "
          f"{sub_j / max(1e-9, retr_j):.2%}")

    # ---- Verdict ----
    print("\n" + "=" * 78)
    if sub_j > 2.0 * rand_j and sub_j > freq_j:
        print("VERDICT: substrate performs GENUINE associative completion.")
        print("It beats both random and frequency-prior baselines using only")
        print("Hebbian writes + field dynamics — no backprop, no decoder, no")
        print("autoregression. This is the substrate-native, non-LLM")
        print("generation signal we were probing for.")
    elif sub_j > 1.2 * rand_j:
        print("VERDICT: substrate shows WEAK associative completion.")
        print("Above random but inconclusive vs frequency prior. Honest")
        print("partial result — the mechanism produces signal but is not")
        print("clearly above the simplest non-substrate baseline.")
    else:
        print("VERDICT: substrate FAILS to perform associative completion.")
        print("Honest negative — joint Hebb imprinting at this density does")
        print("not establish issue→patch attractor binding strong enough to")
        print("beat random. The substrate is a memory + classifier, not a")
        print("non-LLM generator.")
    print("=" * 78)


if __name__ == "__main__":
    main()
