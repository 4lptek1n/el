"""el_use_classify — substrate USES stored knowledge for classification.

Real downstream task: given an unseen paragraph, classify which book
it came from (5-way: Pride/Moby/Sherlock/Frankenstein/Tom Sawyer).

Substrate stores TRAINING paragraphs labeled by book. Recall returns
nearest stored pattern -> emit its book label. The classifier is the
substrate's own associative recall, USED, not just measured.

Baselines:
  - random (20%)
  - bag-of-trigram nearest centroid (real text classifier)
  - bag-of-trigram kNN-1 (uncompressed, brute force)
"""
from __future__ import annotations
import sys, time, random, hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory
from el_text_massive import text_to_pattern


BOOKS = [
    ("pride.txt",        "Austen — Pride & Prejudice"),
    ("moby.txt",         "Melville — Moby Dick"),
    ("sherlock.txt",     "Doyle — Sherlock Holmes"),
    ("frankenstein.txt", "Shelley — Frankenstein"),
    ("tomsawyer.txt",    "Twain — Tom Sawyer"),
]


def load_book(p: Path) -> str:
    raw = p.read_text(encoding="utf-8", errors="ignore")
    s, e = raw.find("*** START OF"), raw.find("*** END OF")
    if s >= 0: s = raw.find("\n", s) + 1
    if e < 0: e = len(raw)
    return " ".join(raw[s:e].split())


def split_chunks(text: str, win: int, n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(0, len(text) - win), n))
    return [text[s:s + win] for s in starts]


# ---------- substrate classifier ----------
class SubstrateClassifier:
    def __init__(self, grid: int, seed: int = 0):
        self.grid = grid
        self.cfg = FieldConfig(rows=grid, cols=grid)
        self.pm = PatternMemory(
            cfg=self.cfg, seed=seed,
            write_lr=0.12, write_steps=20, write_decay=0.002,
            recall_steps=20,
        )
        self.labels: list[int] = []
    def fit(self, chunks: list[str], labels: list[int]):
        for c, y in zip(chunks, labels):
            self.pm.store(text_to_pattern(c, self.grid))
            self.labels.append(y)
    def predict(self, chunk: str) -> int:
        idx, _, _ = self.pm.recall(text_to_pattern(chunk, self.grid))
        return self.labels[idx] if 0 <= idx < len(self.labels) else 0


# ---------- baselines ----------
def trigrams(s: str) -> set[str]:
    s = s.lower()
    return {s[i:i + 3] for i in range(len(s) - 2)}


class TrigramKNN:
    def __init__(self): self.items = []
    def fit(self, chunks, labels):
        for c, y in zip(chunks, labels):
            self.items.append((trigrams(c), y))
    def predict(self, chunk):
        ct = trigrams(chunk)
        best, by = -1.0, 0
        for t, y in self.items:
            j = len(ct & t) / max(1, len(ct | t))
            if j > best: best, by = j, y
        return by


class CentroidClassifier:
    """Per-class trigram count centroid, predict by max cosine."""
    def __init__(self): self.cents = {}
    def fit(self, chunks, labels):
        accum = defaultdict(Counter)
        for c, y in zip(chunks, labels):
            accum[y].update(trigrams(c))
        for y, cnt in accum.items():
            tot = sum(cnt.values())
            self.cents[y] = {k: v / tot for k, v in cnt.items()}
    def predict(self, chunk):
        ct = trigrams(chunk)
        best, by = -1.0, 0
        for y, c in self.cents.items():
            score = sum(c.get(t, 0) for t in ct)
            if score > best: best, by = score, y
        return by


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    print("=" * 78)
    print("EL USE — substrate classifies which book a paragraph is from")
    print("=" * 78)

    win = 200
    per_book_train = 400
    per_book_test = 100
    grid = 192

    train_chunks, train_y = [], []
    test_chunks, test_y = [], []
    for yi, (fn, name) in enumerate(BOOKS):
        text = load_book(data_dir / fn)
        # strict split: train from first 70%, test from last 30%
        cut = int(0.70 * len(text))
        tr = split_chunks(text[:cut], win, per_book_train, seed=10 + yi)
        te = split_chunks(text[cut:], win, per_book_test, seed=200 + yi)
        train_chunks += tr; train_y += [yi] * len(tr)
        test_chunks += te; test_y += [yi] * len(te)
        print(f"  [{name:<32}] {len(tr)} train + {len(te)} test from {len(text):,} char")
    print(f"[total] train={len(train_chunks)}, test={len(test_chunks)}, "
          f"5 classes, win={win}, grid={grid}x{grid}")

    classifiers = {
        "centroid (trigram)":  CentroidClassifier(),
        "kNN-1 (trigram brute)": TrigramKNN(),
        "substrate (recall)":  SubstrateClassifier(grid=grid),
    }

    print("\n[fit]")
    for name, m in classifiers.items():
        t0 = time.time()
        m.fit(train_chunks, train_y)
        print(f"  {name:<25} fit={time.time()-t0:.1f}s")

    print("\n[per-class accuracy on held-out test set]")
    print(f"  {'classifier':<25}  ", end="")
    for _, name in BOOKS:
        print(f"{name.split(' — ')[0][:8]:>9}", end="")
    print(f"{'OVERALL':>10}")

    rows = {}
    for name, m in classifiers.items():
        per = [0] * len(BOOKS); cnt = [0] * len(BOOKS)
        t0 = time.time()
        for c, y in zip(test_chunks, test_y):
            p = m.predict(c)
            cnt[y] += 1
            if p == y: per[y] += 1
        rows[name] = (per, cnt, time.time() - t0)
        print(f"  {name:<25}  ", end="")
        for k in range(len(BOOKS)):
            acc = per[k] / cnt[k] if cnt[k] else 0
            print(f"{acc:>9.3f}", end="")
        ov = sum(per) / sum(cnt)
        print(f"{ov:>10.3f}  ({rows[name][2]:.1f}s)")

    print("\n[confusion matrix — substrate]")
    confmat = np.zeros((len(BOOKS), len(BOOKS)), dtype=int)
    sub = classifiers["substrate (recall)"]
    for c, y in zip(test_chunks, test_y):
        p = sub.predict(c)
        confmat[y, p] += 1
    label = "true/pred"
    print(f"  {label:<10}", end="")
    for _, n in BOOKS: print(f"{n.split(' — ')[0][:8]:>10}", end="")
    print()
    for i, (_, n) in enumerate(BOOKS):
        print(f"  {n.split(' — ')[0][:8]:<10}", end="")
        for j in range(len(BOOKS)):
            print(f"{confmat[i, j]:>10d}", end="")
        print()


if __name__ == "__main__":
    main()
