"""el_use_classify_v2 — substrate USES knowledge via top-K weighted vote.

Fix from v1: instead of single nearest-neighbor argmax, query substrate's
field with the test cue, take the resulting hot_set, then compute
Jaccard overlap with ALL stored patterns and AGGREGATE top-K by
similarity-weighted majority vote over labels.

Also tries a 'one-substrate-per-class' variant where each class has
its OWN field; classify by which field's recall has highest hot_set
overlap with the class's combined attractor.
"""
from __future__ import annotations
import sys, time, random
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory
from el_text_massive import text_to_pattern
from el_use_classify import (
    BOOKS, load_book, split_chunks, CentroidClassifier, TrigramKNN,
)


class SubstrateTopK:
    """Single shared substrate; classify by top-K Jaccard vote on hot_set."""
    def __init__(self, grid: int, top_k: int = 25, seed: int = 0):
        self.grid, self.top_k = grid, top_k
        self.cfg = FieldConfig(rows=grid, cols=grid)
        self.pm = PatternMemory(
            cfg=self.cfg, seed=seed,
            write_lr=0.12, write_steps=20, write_decay=0.002,
            recall_steps=20,
        )
        self.labels: list[int] = []

    def fit(self, chunks, labels):
        for c, y in zip(chunks, labels):
            self.pm.store(text_to_pattern(c, self.grid))
            self.labels.append(y)

    def predict(self, chunk):
        idx, _, hot = self.pm.recall(text_to_pattern(chunk, self.grid))
        hot_set = {tuple(x) for x in hot.tolist()}
        # similarity of hot_set with every stored pattern
        scores = []
        for i, p in enumerate(self.pm.patterns):
            ps = set(p)
            inter = len(hot_set & ps)
            union = len(hot_set | ps)
            scores.append(inter / union if union else 0.0)
        order = np.argsort(scores)[::-1][:self.top_k]
        # weighted majority vote
        bucket = Counter()
        for i in order:
            bucket[self.labels[i]] += scores[i]
        return bucket.most_common(1)[0][0]


class SubstratePerClass:
    """One field per class. Classify by which class-substrate has the
    highest overlap of (test cue's hot_set) ∩ (the class's combined
    stored cells)."""
    def __init__(self, n_classes: int, grid: int, seed: int = 0):
        self.n_classes = n_classes
        self.grid = grid
        self.cfgs = [FieldConfig(rows=grid, cols=grid) for _ in range(n_classes)]
        self.pms = [PatternMemory(
            cfg=self.cfgs[k], seed=seed + k,
            write_lr=0.12, write_steps=20, write_decay=0.002,
            recall_steps=20,
        ) for k in range(n_classes)]
        self.cell_union: list[set] = [set() for _ in range(n_classes)]

    def fit(self, chunks, labels):
        for c, y in zip(chunks, labels):
            p = text_to_pattern(c, self.grid)
            self.pms[y].store(p)
            self.cell_union[y].update(p)

    def predict(self, chunk):
        # Probe each class-substrate with the same cue, compare hot_set
        cue = text_to_pattern(chunk, self.grid)
        cue_set = set(cue)
        best, by = -1.0, 0
        for k in range(self.n_classes):
            _, _, hot = self.pms[k].recall(cue)
            hot_set = {tuple(x) for x in hot.tolist()}
            cu = self.cell_union[k]
            # how strongly does the class-k field reproduce the cue
            # in its hot_set, weighted by overlap with the class union
            attractor_match = len(hot_set & cu) / max(1, len(hot_set | cu))
            cue_match = len(hot_set & cue_set) / max(1, len(hot_set | cue_set))
            score = 0.5 * attractor_match + 0.5 * cue_match
            if score > best: best, by = score, k
        return by


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    print("=" * 78)
    print("EL USE v2 — substrate classifier with top-K aggregation")
    print("=" * 78)
    win, per_book_train, per_book_test, grid = 200, 400, 100, 192

    train_chunks, train_y = [], []
    test_chunks, test_y = [], []
    for yi, (fn, name) in enumerate(BOOKS):
        text = load_book(data_dir / fn)
        cut = int(0.70 * len(text))
        tr = split_chunks(text[:cut], win, per_book_train, seed=10 + yi)
        te = split_chunks(text[cut:], win, per_book_test, seed=200 + yi)
        train_chunks += tr; train_y += [yi] * len(tr)
        test_chunks += te; test_y += [yi] * len(te)
        print(f"  [{name:<32}] {len(tr)} train + {len(te)} test from {len(text):,} char")
    print(f"[total] train={len(train_chunks)}, test={len(test_chunks)}, "
          f"5 classes, win={win}, grid={grid}x{grid}")

    classifiers = {
        "centroid (trigram)":     CentroidClassifier(),
        "kNN-1 (trigram brute)":  TrigramKNN(),
        "substrate top-25 vote":  SubstrateTopK(grid=grid, top_k=25),
        "substrate per-class":    SubstratePerClass(n_classes=len(BOOKS), grid=grid),
    }

    print("\n[fit]")
    for name, m in classifiers.items():
        t0 = time.time()
        m.fit(train_chunks, train_y)
        print(f"  {name:<28} fit={time.time()-t0:.1f}s")

    print("\n[per-class accuracy on held-out test]")
    print(f"  {'classifier':<28} ", end="")
    for _, name in BOOKS:
        print(f"{name.split(' — ')[0][:8]:>9}", end="")
    print(f"{'OVERALL':>10}")

    for name, m in classifiers.items():
        per = [0] * len(BOOKS); cnt = [0] * len(BOOKS)
        t0 = time.time()
        for c, y in zip(test_chunks, test_y):
            p = m.predict(c)
            cnt[y] += 1
            if p == y: per[y] += 1
        print(f"  {name:<28} ", end="")
        for k in range(len(BOOKS)):
            print(f"{(per[k]/cnt[k] if cnt[k] else 0):>9.3f}", end="")
        print(f"{sum(per)/sum(cnt):>10.3f}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
