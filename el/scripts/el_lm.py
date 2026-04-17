"""el_lm — substrate-based next-character predictor.

LLM'in temel işini backprop'suz substrate ile yapar:
  - Eğitim: text üzerinde kayar pencere; (n-gram) → next-char eşlemesini
    PatternMemory'ye yazar
  - Üretim: seed metin ver, her adımda son n karakteri cue olarak ver,
    en yakın kayıtlı n-gram'ı bul, onun next-char'ını çıktı olarak al,
    pencereyi kaydır, tekrar et.

Tamamen offline, gradient yok, transformer yok. Karakter seviyesi.
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory

GRID = 48
NGRAM = 6                     # context window (chars)
CELLS_PER_CHAR = 6            # how many cells each char in the window lights


def ngram_to_pattern(ngram: str, grid: int = GRID,
                     cells_per_char: int = CELLS_PER_CHAR
                     ) -> list[tuple[int, int]]:
    """Position-aware sparse encoding: char at position i hashes with salt=i."""
    positions: set[tuple[int, int]] = set()
    for i, ch in enumerate(ngram):
        for s in range(cells_per_char):
            h = hashlib.blake2b(f"{i}|{ch}|{s}".encode(), digest_size=4).digest()
            idx = int.from_bytes(h, "big") % (grid * grid)
            positions.add((idx // grid, idx % grid))
    return sorted(positions)


def train(text: str, *, ngram: int = NGRAM, seed: int = 0
          ) -> tuple[PatternMemory, list[str]]:
    """Slide window over text; store each ngram, remember its next-char label."""
    cfg = FieldConfig(rows=GRID, cols=GRID)
    pm = PatternMemory(cfg=cfg, seed=seed)
    next_chars: list[str] = []
    n_total = len(text) - ngram
    for i in range(n_total):
        window = text[i:i + ngram]
        nxt = text[i + ngram]
        pm.store(ngram_to_pattern(window))
        next_chars.append(nxt)
    return pm, next_chars


def generate(pm: PatternMemory, next_chars: list[str], seed_text: str,
             *, n_chars: int = 200, ngram: int = NGRAM,
             temperature: float = 0.0, rng=None) -> str:
    """Greedy (T=0) or weighted-by-score sampling generation."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = list(seed_text)
    if len(out) < ngram:
        out = list(" " * (ngram - len(out)) + seed_text)

    for _ in range(n_chars):
        ctx = "".join(out[-ngram:])
        cue = ngram_to_pattern(ctx)
        # Get top-k matches by Jaccard, then pick by next-char distribution
        if temperature <= 0:
            pred_idx, score, _ = pm.recall(cue)
            if pred_idx < 0:
                break
            out.append(next_chars[pred_idx])
        else:
            # Score every stored pattern; sample softmax over scores
            scores = []
            for p in pm.patterns:
                ps = set(p); cs = set(cue)
                inter = len(ps & cs); union = len(ps | cs)
                scores.append(inter / union if union else 0.0)
            scores = np.asarray(scores, dtype=np.float64)
            if scores.max() <= 0:
                break
            probs = np.exp(scores / temperature)
            probs /= probs.sum()
            pick = int(rng.choice(len(probs), p=probs))
            out.append(next_chars[pick])
    return "".join(out)


SAMPLE_TEXT = (
    "ali topu tut. top yuvarlandı. ayşe topu aldı. ali ayşeye bak. "
    "ayşe topu attı. top kaleye gitti. ali topu tut, ayşe topu attı. "
    "ali ile ayşe oyun oynadı. top yuvarlandı kaleye doğru. "
    "ayşe gülümsedi, ali sevindi. oyun bitti, ali topu eve götürdü. "
)


def main():
    p = argparse.ArgumentParser(prog="el_lm",
        description="Substrate-based next-char predictor (no LLM).")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Text file to train on (default: built-in sample)")
    p.add_argument("--seed-text", default="ali topu",
                   help="Seed for generation")
    p.add_argument("--n", type=int, default=120, help="chars to generate")
    p.add_argument("--ngram", type=int, default=NGRAM)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0=greedy, >0=softmax sample")
    p.add_argument("--show-stats", action="store_true")
    args = p.parse_args()

    text = args.corpus.read_text() if args.corpus else SAMPLE_TEXT
    text = text.lower()
    print(f"[train] corpus length: {len(text)} chars, ngram={args.ngram}")
    print(f"[train] storing {len(text) - args.ngram} ngram→nextchar associations…")
    pm, next_chars = train(text, ngram=args.ngram)
    if args.show_stats:
        print(f"[train] {len(pm.patterns)} patterns stored on "
              f"{GRID}×{GRID} substrate")

    print(f"[generate] seed: '{args.seed_text}', T={args.temperature}")
    out = generate(pm, next_chars, args.seed_text,
                   n_chars=args.n, ngram=args.ngram,
                   temperature=args.temperature)
    print("---")
    print(out)
    print("---")


if __name__ == "__main__":
    main()
