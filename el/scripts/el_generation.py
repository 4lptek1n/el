"""el_generation — substrate as a TEXT GENERATOR (LLM-style autocomplete).

Given a real text prompt (40 chars), generate the next 80 chars by
iteratively recalling the nearest stored chunk and emitting its
continuation, then sliding the cue window forward.

Compared against:
  - random char generator
  - bigram count-based LM (real n-gram model trained on corpus)
  - trigram count-based LM
"""
from __future__ import annotations
import sys, time, random
from collections import defaultdict, Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from el_text_massive import load_books, chunks, SubstrateTextMemory


# ---------- baselines ----------
class NgramLM:
    def __init__(self, n: int):
        self.n = n
        self.tbl = defaultdict(Counter)
    def fit(self, text: str):
        for i in range(len(text) - self.n):
            ctx = text[i:i + self.n - 1]
            nxt = text[i + self.n - 1]
            self.tbl[ctx][nxt] += 1
    def gen_next(self, ctx: str) -> str:
        ctx = ctx[-(self.n - 1):]
        c = self.tbl.get(ctx)
        if not c: return " "
        return c.most_common(1)[0][0]
    def generate(self, prompt: str, m: int) -> str:
        out = list(prompt)
        for _ in range(m):
            out.append(self.gen_next("".join(out)))
        return "".join(out)[len(prompt):]


class RandomLM:
    def fit(self, text): self.alpha = sorted(set(text)); self.r = random.Random(7)
    def generate(self, prompt, m): return "".join(self.r.choice(self.alpha) for _ in range(m))


class SubstrateLM:
    """Generation by iterative content-addressable recall."""
    def __init__(self, sub: SubstrateTextMemory, win: int, cue_len: int):
        self.sub = sub; self.win = win; self.cue_len = cue_len
    def generate(self, prompt: str, m: int) -> str:
        out = prompt
        gen = ""
        while len(gen) < m:
            cue = out[-self.cue_len:]
            chunk = self.sub.recall(cue)
            # take the continuation that comes AFTER what matches the cue
            # find best alignment of cue inside chunk
            i = chunk.find(cue[-20:])     # try to align last 20 chars of cue
            if i < 0: i = 0
            tail = chunk[i + 20:] if i >= 0 else chunk[self.cue_len:]
            if not tail: tail = chunk[self.cue_len:]
            take = tail[:max(1, self.win - self.cue_len)]
            if not take:
                take = " "
            gen += take
            out += take
        return gen[:m]


def char_acc(pred: str, true: str) -> float:
    return sum(a == b for a, b in zip(pred, true)) / max(1, len(true))


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    print("=" * 78)
    print("EL GENERATION — substrate as autocompleter on real Gutenberg text")
    print("=" * 78)
    text = load_books(data_dir)
    print(f"[corpus] {len(text):,} chars")

    win, cue_len, gen_len = 120, 40, 80
    n_chunks = 3000
    grid = 192
    n_eval = 50

    cs = list(dict.fromkeys(chunks(text, win, n_chunks, seed=42)))[:n_chunks]
    train_text = " ".join(cs)
    print(f"[train] {len(cs)} chunks, {len(train_text):,} train chars; "
          f"grid={grid}x{grid}; gen_len={gen_len} chars after {cue_len}-char prompt")

    # evaluation prompts: REAL excerpts from corpus that are NOT in train chunks
    rng = random.Random(99)
    eval_starts = rng.sample(range(len(text) - win - cue_len), n_eval * 5)
    eval_pairs = []
    train_set = set(cs)
    for s in eval_starts:
        ex = text[s:s + cue_len + gen_len]
        if not any(c in ex for c in train_set):
            eval_pairs.append((text[s:s + cue_len], text[s + cue_len:s + cue_len + gen_len]))
        if len(eval_pairs) >= n_eval: break
    print(f"[eval] {len(eval_pairs)} held-out (prompt, continuation) pairs")

    print("\n[fit]")
    rnd = RandomLM(); rnd.fit(train_text)
    bg = NgramLM(2); bg.fit(train_text); print(f"  bigram contexts: {len(bg.tbl)}")
    tg = NgramLM(3); tg.fit(train_text); print(f"  trigram contexts: {len(tg.tbl)}")
    fg = NgramLM(4); fg.fit(train_text); print(f"  4-gram contexts: {len(fg.tbl)}")

    sub = SubstrateTextMemory(grid=grid)
    t0 = time.time()
    for c in cs: sub.store(c)
    print(f"  substrate fit: {time.time()-t0:.1f}s ({n_chunks} chunks)")
    sublm = SubstrateLM(sub, win=win, cue_len=cue_len)

    print("\n[generation char-accuracy on held-out prompts]")
    print(f"{'model':<18} {'mean_acc':>10} {'max_acc':>9}  example pred (first 60 chars)")
    print("-" * 100)
    sample = eval_pairs[0]
    for name, m in [("random",   rnd),
                    ("bigram LM", bg),
                    ("trigram LM", tg),
                    ("4-gram LM", fg),
                    ("substrate", sublm)]:
        accs = []
        for prompt, true in eval_pairs:
            pred = m.generate(prompt, gen_len)
            accs.append(char_acc(pred, true))
        mean = sum(accs) / len(accs)
        mx = max(accs)
        ex_pred = m.generate(sample[0], gen_len)[:60]
        print(f"{name:<18} {mean:>10.3f} {mx:>9.3f}  {ex_pred!r}")

    print("\nground truth example continuation (first 60 chars):")
    print(f"  prompt: {sample[0]!r}")
    print(f"  true:   {sample[1][:60]!r}")


if __name__ == "__main__":
    main()
