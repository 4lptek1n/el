"""el_lm_v2 — substrate-NATIVE next-char predictor on LayeredField + RRAM crossbar.

What changed vs el_lm:
  - Real LayeredField: 3 stacked 64×64 fields with vertical C+B and
    if-then rising-edge spike gating between layers (cortical column motif).
  - SparseCrossbar overlay: K=8 RRAM-style non-local edges per cell on L0
    so far-apart context tokens can still reach the output region.
  - Output is read from L2 (abstraction layer), not from PatternMemory.
  - Heat actually propagates: input heats L0 → diffuses + spikes to L1 →
    spikes to L2 → readout region with hottest char-block wins.
  - Hebbian training on intra-layer C, vertical C_v, and crossbar C
    simultaneously (substrate-wide plasticity).

NOT an n-gram lookup. The substrate physically encodes context-to-char
associations in its conductivity matrix.
"""
from __future__ import annotations
import argparse, hashlib, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.layered import LayeredField
from el.thermofield.crossbar import SparseCrossbar


# ---------- Encoding ----------
def ngram_to_cells(ngram: str, grid: int, cells_per_char: int = 5
                   ) -> list[tuple[int, int]]:
    """Position-aware sparse encoding of a context window into L0 cells."""
    out: set[tuple[int, int]] = set()
    for i, ch in enumerate(ngram):
        for s in range(cells_per_char):
            h = hashlib.blake2b(f"L0|{i}|{ch}|{s}".encode(), digest_size=4).digest()
            idx = int.from_bytes(h, "big") % (grid * grid)
            out.add((idx // grid, idx % grid))
    return sorted(out)


def char_to_block(ch: str, vocab: list[str], grid: int,
                  block: int = 4) -> list[tuple[int, int]]:
    """Map char → contiguous block of `block×block` cells in L2 output region."""
    if ch not in vocab:
        return []
    idx = vocab.index(ch)
    cols_per_row = grid // block
    br, bc = divmod(idx, cols_per_row)
    r0, c0 = br * block, bc * block
    return [(r0 + dr, c0 + dc) for dr in range(block) for dc in range(block)]


# ---------- Substrate ----------
class SubstrateLM:
    def __init__(self, *, grid: int = 64, n_layers: int = 3,
                 crossbar_k: int = 8, ngram: int = 7, seed: int = 0,
                 n_relax: int = 8, hebb_lr: float = 0.04,
                 vocab: list[str] | None = None):
        cfg = FieldConfig(rows=grid, cols=grid, diffusion_rate=0.30,
                          decay=0.005, n_steps=n_relax, nonlinear_alpha=2.5)
        self.layered = LayeredField(n_layers=n_layers, cfg=cfg, seed=seed,
                                    vertical_rate=0.30,
                                    firing_threshold=0.25,
                                    spike_amplitude=0.25)
        # Crossbar overlay on L0 only (cheap, decisive for long-range cues)
        self.crossbar = SparseCrossbar(n_cells=grid * grid, k=crossbar_k,
                                       seed=seed + 99, flux_rate=0.08)
        self.grid = grid
        self.n_relax = n_relax
        self.hebb_lr = hebb_lr
        self.ngram = ngram
        self.vocab = vocab or []

    def _step_full(self) -> None:
        """One global step: layered (intra+vertical+spike) + crossbar on L0."""
        self.layered.step()
        T0 = self.layered.layers[0].T
        flat = T0.reshape(-1)
        self.crossbar.step(flat)
        self.layered.layers[0].T = flat.reshape(self.grid, self.grid)
        # Re-apply L0 clamps (crossbar.step may have nudged clamped cells)
        if hasattr(self.layered.layers[0], "_clamp_positions"):
            self.layered.layers[0]._apply_clamps()

    def train_pair(self, ngram: str, target_char: str) -> None:
        """One supervised episode: clamp L0=ngram + L2=target_char, relax,
        Hebbian on co-activity in each layer + crossbar."""
        self.layered.reset()
        in_cells = ngram_to_cells(ngram, self.grid)
        out_cells = char_to_block(target_char, self.vocab, self.grid)
        if not out_cells:
            return
        self.layered.inject(0, in_cells, [1.0] * len(in_cells))
        # Clamp target on L2 (top layer = abstraction)
        self.layered.inject(self.layered.n_layers - 1, out_cells,
                            [1.0] * len(out_cells))
        for _ in range(self.n_relax):
            self._step_full()
        # Hebbian: intra-layer C_right/C_down on every layer
        for layer in self.layered.layers:
            T = layer.T
            avg_h = 0.5 * (T[:, :-1] + T[:, 1:])
            avg_v = 0.5 * (T[:-1, :] + T[1:, :])
            layer.C_right += self.hebb_lr * avg_h
            layer.C_down += self.hebb_lr * avg_v
            np.clip(layer.C_right, 0.05, 1.5, out=layer.C_right)
            np.clip(layer.C_down, 0.05, 1.5, out=layer.C_down)
        # Hebbian on vertical C_v between layers
        for l in range(self.layered.n_layers - 1):
            co = 0.5 * (self.layered.layers[l].T + self.layered.layers[l + 1].T)
            self.layered.C_v[l] += self.hebb_lr * co
            np.clip(self.layered.C_v[l], 0.05, 1.5, out=self.layered.C_v[l])
        # Hebbian on crossbar C (long-range RRAM update)
        T0_flat = self.layered.layers[0].T.reshape(-1)
        co_xb = T0_flat[self.crossbar.src] * T0_flat[self.crossbar.dst]
        new_C = self.crossbar.C + self.hebb_lr * 0.5 * co_xb
        np.clip(new_C, 0.05, 1.5, out=new_C)
        self.crossbar.C = new_C
        # Anti-Hebbian on L2: weaken edges leading INTO non-target char blocks
        # (winner-take-all style competition between output regions)
        T_top = self.layered.layers[-1].T
        for ch in self.vocab:
            if ch == target_char:
                continue
            cells = char_to_block(ch, self.vocab, self.grid)
            if not cells:
                continue
            # Decay any heat that reached this non-target block by lowering
            # incoming intra-layer C around it
            for (r, c) in cells:
                if c > 0:
                    self.layered.layers[-1].C_right[r, c - 1] *= (1.0 - self.hebb_lr * 0.5)
                if c < self.grid - 1:
                    self.layered.layers[-1].C_right[r, c] *= (1.0 - self.hebb_lr * 0.5)
                if r > 0:
                    self.layered.layers[-1].C_down[r - 1, c] *= (1.0 - self.hebb_lr * 0.5)
                if r < self.grid - 1:
                    self.layered.layers[-1].C_down[r, c] *= (1.0 - self.hebb_lr * 0.5)
            np.clip(self.layered.layers[-1].C_right, 0.05, 1.5,
                    out=self.layered.layers[-1].C_right)
            np.clip(self.layered.layers[-1].C_down, 0.05, 1.5,
                    out=self.layered.layers[-1].C_down)

    def predict_char(self, ngram: str) -> tuple[str, np.ndarray]:
        """Inference: clamp L0=ngram, relax, read all L2 char-blocks, return
        argmax char + score-vector for diagnostics."""
        self.layered.reset()
        in_cells = ngram_to_cells(ngram, self.grid)
        self.layered.inject(0, in_cells, [1.0] * len(in_cells))
        for _ in range(self.n_relax):
            self._step_full()
        T2 = self.layered.layers[-1].T
        scores = np.zeros(len(self.vocab), dtype=np.float32)
        for i, ch in enumerate(self.vocab):
            cells = char_to_block(ch, self.vocab, self.grid)
            if cells:
                scores[i] = float(np.mean([T2[r, c] for r, c in cells]))
        return self.vocab[int(scores.argmax())], scores


# ---------- Driver ----------
def build_vocab(text: str) -> list[str]:
    chars = sorted(set(text))
    return chars


def train(text: str, model: SubstrateLM, *, epochs: int = 1,
          report_every: int = 200) -> None:
    n = len(text) - model.ngram
    t0 = time.time()
    for ep in range(epochs):
        for i in range(n):
            ng = text[i:i + model.ngram]
            tgt = text[i + model.ngram]
            model.train_pair(ng, tgt)
            if (i + 1) % report_every == 0:
                el = time.time() - t0
                print(f"[ep {ep+1} {i+1}/{n}] {el:.1f}s "
                      f"({(i+1)/el:.1f} pairs/s)")


def generate(model: SubstrateLM, seed_text: str, n_chars: int = 120) -> str:
    out = list(seed_text)
    if len(out) < model.ngram:
        out = list(" " * (model.ngram - len(out)) + seed_text)
    for _ in range(n_chars):
        ctx = "".join(out[-model.ngram:])
        ch, _ = model.predict_char(ctx)
        out.append(ch)
    return "".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--seed-text", default="ali sabah")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--ngram", type=int, default=7)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--crossbar-k", type=int, default=8)
    p.add_argument("--n-relax", type=int, default=8)
    p.add_argument("--hebb-lr", type=float, default=0.04)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-chars", type=int, default=0,
                   help="truncate corpus for speed (0=full)")
    args = p.parse_args()

    text = args.corpus.read_text().lower()
    if args.max_chars > 0:
        text = text[: args.max_chars]
    vocab = build_vocab(text)
    print(f"[corpus] {len(text)} chars, vocab={len(vocab)} "
          f"({''.join(vocab)!r})")
    print(f"[substrate] {args.n_layers}× {args.grid}×{args.grid} layers "
          f"+ crossbar k={args.crossbar_k}, ngram={args.ngram}, "
          f"relax={args.n_relax}")
    model = SubstrateLM(grid=args.grid, n_layers=args.n_layers,
                        crossbar_k=args.crossbar_k, ngram=args.ngram,
                        n_relax=args.n_relax, hebb_lr=args.hebb_lr,
                        vocab=vocab)
    print(f"[train] {len(text)-args.ngram} pairs × {args.epochs} epochs")
    train(text, model, epochs=args.epochs)

    print(f"\n[generate] seed: {args.seed_text!r}")
    out = generate(model, args.seed_text, n_chars=args.n)
    print("---")
    print(out)
    print("---")


if __name__ == "__main__":
    main()
