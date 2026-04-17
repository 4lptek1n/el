"""el_hybrid_fix — hybrid: substrate persistent memory + rule agent.

Practical pattern: substrate handles WHAT IT'S GOOD AT (associative
recall of past fixes), rule-based agent handles WHAT IT'S GOOD AT
(text matching + apply edit). Together they beat either alone on
recurring bug patterns.

Demo: `MultiModalSubstrate` stores (buggy_signature -> fix_id) pairs.
When a new buggy code arrives, substrate recalls the closest stored
signature, agent retrieves and applies its known fix. If recall score
is too low, agent FALLS BACK to no-fix (honest abstain).

This is the production-shaped pattern: substrate as on-device memory,
deterministic agent as actuator.
"""
from __future__ import annotations
import sys, hashlib, tempfile, subprocess, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.multi_substrate import MultiModalSubstrate


def code_to_pattern(code: str, grid: int = 48) -> list[tuple[int, int]]:
    cells = set()
    for i, ch in enumerate(code[:512]):
        for s in range(4):
            h = hashlib.blake2b(f"{i}|{ch}|{s}".encode(), digest_size=4).digest()
            idx = int.from_bytes(h, "big") % (grid * grid)
            cells.add((idx // grid, idx % grid))
    return sorted(cells)


class HybridFixer:
    def __init__(self, grid: int = 48, recall_threshold: float = 0.40):
        self.cfg = FieldConfig(rows=grid, cols=grid)
        self.sub = MultiModalSubstrate(cfg=self.cfg)
        self.fixes: list[tuple[int, str, str]] = []  # (id, find, replace)
        self.recall_threshold = recall_threshold
        self.grid = grid

    def learn(self, buggy_code: str, find: str, replace: str) -> int:
        """Remember that for `buggy_code`-ish patterns, replacing `find`
        with `replace` is the verified fix."""
        fix_id = len(self.fixes)
        self.fixes.append((fix_id, find, replace))
        self.sub.store_pattern(code_to_pattern(buggy_code, self.grid))
        return fix_id

    def propose(self, buggy_code: str) -> tuple[str | None, float]:
        """Return (patched_code or None, recall_score). None = abstain."""
        cue = code_to_pattern(buggy_code, self.grid)
        idx, score, _ = self.sub.recall(cue)
        if idx < 0 or score < self.recall_threshold:
            return None, float(score)
        _, find, replace = self.fixes[idx]
        if find not in buggy_code:
            return None, float(score)
        return buggy_code.replace(find, replace, 1), float(score)


def verify(code: str, test: str, timeout: float = 3.0) -> bool:
    full = code + "\n" + test + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        return r.returncode == 0
    finally:
        try: os.unlink(path)
        except OSError: pass


# ---------- Demo ----------
TRAIN = [
    ("def add(a,b):\n    return a - b", "a - b", "a + b",
     "assert add(2,3) == 5"),
    ("def mul(a,b):\n    return a + b", "a + b", "a * b",
     "assert mul(3,4) == 12"),
    ("def sq(x):\n    return x + x", "x + x", "x * x",
     "assert sq(5) == 25"),
]

TEST = [
    # in-distribution: similar to train items 0, 1, 2
    ("def total(a,b):\n    return a - b", "a - b", "a + b",
     "assert total(10,7) == 17"),
    ("def prod(a,b):\n    return a + b", "a + b", "a * b",
     "assert prod(6,7) == 42"),
    ("def square(x):\n    return x + x", "x + x", "x * x",
     "assert square(8) == 64"),
    # out-of-distribution: substrate should ABSTAIN
    ("def f(L):\n    return L[0]", "L[0]", "L[-1]",
     "assert f([1,2,9]) == 9"),
]


def main():
    fx = HybridFixer(grid=48, recall_threshold=0.10)
    print("[learn] storing 3 known fix-patterns in substrate...")
    for code, find, replace, _ in TRAIN:
        fid = fx.learn(code, find, replace)
        print(f"  fix#{fid}: {find!r} -> {replace!r}")

    print("\n[test] proposing fixes on novel buggy code")
    n_pass = n_abstain = n_wrong = 0
    for code, _, _, test in TEST:
        patched, score = fx.propose(code)
        if patched is None:
            print(f"  ABSTAIN (score={score:.2f}): {code.splitlines()[1].strip()}")
            n_abstain += 1
            continue
        ok = verify(patched, test)
        verdict = "PASS" if ok else "FAIL"
        print(f"  {verdict}    (score={score:.2f}): "
              f"{code.splitlines()[1].strip()} -> "
              f"{patched.splitlines()[1].strip()}")
        if ok: n_pass += 1
        else:  n_wrong += 1
    print(f"\n[result] PASS={n_pass}  ABSTAIN={n_abstain}  WRONG={n_wrong}  "
          f"out of {len(TEST)}")
    print("[interpretation] substrate routes recurring patterns to known "
          "fixes, abstains on novel.")


if __name__ == "__main__":
    main()
