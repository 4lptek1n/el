"""Training loop for the Action Transformer.

Runs three configurations:

    python -m el.transformer.train --preset tiny --steps 50           # CPU smoke
    python -m el.transformer.train --preset small --steps 5000        # single GPU
    python -m el.transformer.train --preset h200 --steps 200000       # Modal H200

Modal entrypoint: see `scripts/modal_train.py`.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from .dataset import build_training_examples
from .model import ModelConfig, TORCH_AVAILABLE, build_model
from .tokenizer import ActionTokenizer


def collect_text_corpus(examples) -> list[str]:
    texts: list[str] = []
    for ex in examples:
        texts.append(ex.intent.raw or ex.intent.verb)
        for a in ex.actions:
            for _, v in a.kwargs:
                texts.append(str(v))
    return texts


def build_tokenizer_and_rows(examples, max_len: int = 512) -> tuple[ActionTokenizer, list[list[int]]]:
    corpus = collect_text_corpus(examples)
    tok = ActionTokenizer.build(corpus, word_cap=4096, max_len=max_len)
    rows = []
    for ex in examples:
        ids = tok.encode_row(ex.intent.to_dict(), [a.to_dict() for a in ex.actions], ex.reward)
        rows.append(ids)
    return tok, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="train the el Action Transformer")
    parser.add_argument("--preset", choices=["tiny", "small", "h200"], default="tiny")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default="checkpoints/el-action-transformer")
    parser.add_argument("--extra", type=str, default=None, help="extra JSONL corpus path")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("error: PyTorch not installed. pip install 'el[train]'")
        return 2

    import torch
    from torch.utils.data import DataLoader, Dataset

    extra_paths = [Path(args.extra)] if args.extra else []
    examples = build_training_examples(extra_paths=extra_paths)
    if not examples:
        print("error: no training examples found. Run `el seed` or pass --extra.")
        return 2

    tokenizer, rows = build_tokenizer_and_rows(examples)
    cfg = ModelConfig.preset(args.preset, vocab_size=tokenizer.vocab_size)
    print(f"[train] examples={len(examples)} vocab={tokenizer.vocab_size} params~{cfg.approx_params():,}")

    class RowDS(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            ids = self.data[i][: cfg.max_len]
            if len(ids) < 8:
                ids = ids + [tokenizer.tid("<pad>")] * (8 - len(ids))
            return torch.tensor(ids, dtype=torch.long)

    def collate(batch):
        max_t = max(len(x) for x in batch)
        pad_id = tokenizer.tid("<pad>")
        out = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
        for i, x in enumerate(batch):
            out[i, : len(x)] = x
        return out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.tid("<pad>"))

    loader = DataLoader(RowDS(rows), batch_size=args.batch, shuffle=True, collate_fn=collate)

    it = iter(loader)
    step = 0
    running = 0.0
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(device)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        logits = model(inp)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        running = 0.98 * running + 0.02 * float(loss.item()) if step else float(loss.item())
        step += 1
        if step % max(1, args.steps // 20) == 0 or step == args.steps:
            print(f"[train] step={step:>6d}/{args.steps} loss={running:.4f} ppl={math.exp(min(20, running)):.2f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"[train] saved checkpoint to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
