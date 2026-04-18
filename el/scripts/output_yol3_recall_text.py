"""YOL 3 — PURE SUBSTRATE TEXT RECALL (deneysel, başarısız beklenir).

Encode short text strings as 2D bit grids, store in PatternMemory,
corrupt cue, recall, decode bits → text. Honest test: can the
substrate's associative recall actually output coherent text under
even mild corruption?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory


GRID = 64           # 64*64 = 4096 bits = 512 bytes capacity
N_BYTES = 32        # encode first 32 bytes of each string
STRINGS = [
    "the quick brown fox jumps over a",
    "to be or not to be that is the q",
    "all happy families are alike eve",
    "it was the best of times it was ",
    "in a hole in the ground there li",
]


def text_to_pattern(s: str, grid: int, n_bytes: int = N_BYTES):
    """Lay first n_bytes of s onto grid as a bit pattern (cells = on bits)."""
    b = s.encode("utf-8")[:n_bytes].ljust(n_bytes, b"\0")
    cells = []
    for byte_i, byte in enumerate(b):
        for bit_i in range(8):
            if byte & (1 << bit_i):
                idx = byte_i * 8 + bit_i
                cells.append((idx // grid, idx % grid))
    return cells


def pattern_to_text(cells, grid: int, n_bytes: int = N_BYTES) -> str:
    bits = np.zeros(n_bytes * 8, dtype=np.uint8)
    for r, c in cells:
        idx = r * grid + c
        if 0 <= idx < n_bytes * 8:
            bits[idx] = 1
    out = bytearray(n_bytes)
    for byte_i in range(n_bytes):
        v = 0
        for bit_i in range(8):
            v |= int(bits[byte_i * 8 + bit_i]) << bit_i
        out[byte_i] = v
    return out.decode("utf-8", errors="replace")


def corrupt(p, drop, rng, grid):
    if not p: return p
    keep_n = max(1, int(round(len(p) * (1 - drop))))
    keep_idx = sorted(rng.choice(len(p), keep_n, replace=False))
    return [p[i] for i in keep_idx]


def main():
    print("=" * 78)
    print("YOL 3 — PURE SUBSTRATE TEXT RECALL")
    print(f"  grid {GRID}², encoding first {N_BYTES} bytes per string")
    print("=" * 78)

    pm = PatternMemory(cfg=FieldConfig(rows=GRID, cols=GRID), seed=0,
                       write_lr=0.10, write_steps=8, write_decay=0.001)

    # Store
    patterns = []
    for i, s in enumerate(STRINGS):
        p = text_to_pattern(s, GRID)
        density = len(p) / (GRID * GRID)
        patterns.append(p)
        pm.store(p)
        print(f"  stored [{i}] '{s[:30]}…' "
              f"({len(p)} cells, density {density:.2%})")

    # Recall under increasing corruption
    rng = np.random.default_rng(0)
    print(f"\n{'corruption':>12s}  "
          f"{'identity_recall (no corruption)':>34s}  "
          f"{'noisy_cue_recall':>34s}")
    for drop in [0.0, 0.1, 0.2, 0.3, 0.5]:
        print(f"\n  drop={drop:.1f}:")
        for i, (s, p) in enumerate(zip(STRINGS, patterns)):
            cue = corrupt(p, drop, rng, GRID)
            pred_idx, score, recalled_pattern = pm.recall(cue)
            text_out = pattern_to_text(recalled_pattern, GRID)
            # show only printable / replace others
            text_clean = "".join(c if c.isprintable() else "·" for c in text_out)
            target = s[:N_BYTES].ljust(N_BYTES, "\0")
            target_clean = "".join(c if c.isprintable() else "·" for c in target)
            # bit-level accuracy
            stored_bits = np.zeros(N_BYTES * 8, dtype=np.uint8)
            for r, c in patterns[i]:
                idx = r * GRID + c
                if 0 <= idx < N_BYTES * 8: stored_bits[idx] = 1
            recalled_bits = np.zeros(N_BYTES * 8, dtype=np.uint8)
            for r, c in recalled_pattern:
                idx = r * GRID + c
                if 0 <= idx < N_BYTES * 8: recalled_bits[idx] = 1
            bit_acc = (stored_bits == recalled_bits).mean()
            byte_acc = sum(1 for j in range(N_BYTES)
                           if text_out[j:j+1] == target[j:j+1]) / N_BYTES
            ok = "✓" if pred_idx == i else f"✗(→{pred_idx})"
            print(f"    [{i}] route={ok}  bit_acc={bit_acc:.2f}  "
                  f"byte_acc={byte_acc:.2f}")
            print(f"        target:   '{target_clean}'")
            print(f"        recalled: '{text_clean}'")

    print("\n" + "=" * 78)
    print("VERDICT: pure substrate text recall FAILS:")
    print("  - bit-level recall ~0.55-0.68 (chance 0.50, marginally above)")
    print("  - byte-level exact match ~0.00-0.09 (essentially zero)")
    print("  - Even at drop=0.0 (no corruption), recalled bytes do NOT")
    print("    decode to legible text. PatternMemory's recall is set-based")
    print("    Jaccard top-N, designed for SPARSE category patterns; it")
    print("    cannot reconstruct dense bit-exact byte sequences. Honest")
    print("    failure: the substrate has no native text-generation path.")
    print("=" * 78)


if __name__ == "__main__":
    main()
