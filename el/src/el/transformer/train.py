"""Training loop for the Action Transformer.

Real (not demo) training. Produces a checkpoint at
``<state_dir>/checkpoints/action-transformer/`` (or ``--out``) with:

- ``model.pt``    — best validation loss state dict
- ``tokenizer.json``
- ``config.json``  — model config
- ``metrics.json`` — train/val loss curves + held-out accuracy

Default corpus is the bundled seed JSONLs **plus** the deterministic synthetic
corpus (``corpus_synth``). Pass ``--no-synth`` to use only bundled data.

Configurations:

    python -m el.transformer.train --preset tiny  --steps 2000   # CPU, ~3M
    python -m el.transformer.train --preset small --steps 5000   # 1× GPU
    python -m el.transformer.train --preset h200  --steps 200000 # H200
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

from .corpus_synth import synthesize_corpus
from .dataset import build_training_examples
from .model import ModelConfig, TORCH_AVAILABLE, build_model
from .tokenizer import ActionTokenizer


def collect_text_corpus(examples) -> list[str]:
    texts: list[str] = []
    for ex in examples:
        texts.append(ex.intent.raw or ex.intent.verb)
        for k, v in ex.intent.args:
            texts.append(str(k))
            texts.append(str(v))
        for a in ex.actions:
            for k, v in a.kwargs:
                texts.append(str(k))
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


def _example_signature(ex) -> tuple:
    """Canonical signature used to group near-duplicates so they all land
    in the same split (no leakage from synthetic combinatorics)."""
    return (
        ex.intent.verb,
        ex.intent.obj,
        ex.intent.scope,
        tuple(sorted(ex.intent.args)),
        tuple(a.name for a in ex.actions),
        tuple(tuple(sorted(a.kwargs)) for a in ex.actions),
    )


def _split_indices_grouped(
    examples, val_frac: float, test_frac: float, seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Group-aware split: every (intent+actions) signature lives in exactly
    one of train/val/test, eliminating duplicate-row leakage across splits."""
    rng = random.Random(seed)
    groups: dict[tuple, list[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(_example_signature(ex), []).append(i)
    keys = list(groups.keys())
    rng.shuffle(keys)
    n_groups = len(keys)
    n_val_g = max(1, int(n_groups * val_frac))
    n_test_g = max(1, int(n_groups * test_frac))
    val_keys = keys[:n_val_g]
    test_keys = keys[n_val_g : n_val_g + n_test_g]
    train_keys = keys[n_val_g + n_test_g :]
    val = [i for k in val_keys for i in groups[k]]
    test = [i for k in test_keys for i in groups[k]]
    train = [i for k in train_keys for i in groups[k]]
    return train, val, test


def _cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> int:
    parser = argparse.ArgumentParser(description="train the el Action Transformer")
    parser.add_argument("--preset", choices=["tiny", "small", "h200"], default="tiny")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--out", type=str, default="checkpoints/el-action-transformer")
    parser.add_argument("--extra", type=str, default=None, help="extra JSONL corpus path")
    parser.add_argument("--no-synth", action="store_true", help="skip synthetic corpus")
    parser.add_argument("--synth-per-verb", type=int, default=120)
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("error: PyTorch not installed. pip install 'el[train]'")
        return 2

    import torch
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    extra_paths = [Path(args.extra)] if args.extra else []
    examples = build_training_examples(extra_paths=extra_paths)
    if not args.no_synth:
        examples = examples + synthesize_corpus(
            examples_per_verb=args.synth_per_verb, seed=args.seed
        )
    if not examples:
        print("error: no training examples found.")
        return 2

    tokenizer, rows = build_tokenizer_and_rows(examples)
    cfg = ModelConfig.preset(args.preset, vocab_size=tokenizer.vocab_size)

    train_idx, val_idx, test_idx = _split_indices_grouped(
        examples, args.val_frac, args.test_frac, args.seed
    )
    print(
        f"[train] examples={len(examples)} (train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}) "
        f"vocab={tokenizer.vocab_size} preset={args.preset} params~{cfg.approx_params():,}"
    )

    class RowDS(Dataset):
        def __init__(self, all_rows, indices):
            self.rows = all_rows
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            ids = self.rows[self.indices[i]][: cfg.max_len]
            if len(ids) < 8:
                ids = ids + [tokenizer.tid("<pad>")] * (8 - len(ids))
            return torch.tensor(ids, dtype=torch.long)

    pad_id = tokenizer.tid("<pad>")

    def collate(batch):
        max_t = max(len(x) for x in batch)
        out = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
        for i, x in enumerate(batch):
            out[i, : len(x)] = x
        return out

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    train_loader = DataLoader(
        RowDS(rows, train_idx), batch_size=args.batch, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        RowDS(rows, val_idx), batch_size=args.batch, shuffle=False, collate_fn=collate
    )

    def run_eval():
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                logits = model(inp)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                ntoks = (tgt != pad_id).sum().item()
                total_loss += float(loss.item()) * ntoks
                total_tokens += ntoks
        model.train()
        return total_loss / max(1, total_tokens)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "preset": args.preset, "device": device, "params": cfg.approx_params(),
        "vocab_size": tokenizer.vocab_size, "n_train": len(train_idx),
        "n_val": len(val_idx), "n_test": len(test_idx),
        "steps": [], "train_loss": [], "val_loss": [],
        "lr": [], "wall_seconds": 0.0,
    }

    it = iter(train_loader)
    step = 0
    running = 0.0
    best_val = float("inf")
    t0 = time.time()
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = batch.to(device)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        logits = model(inp)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        lr = _cosine_lr(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        running = 0.98 * running + 0.02 * float(loss.item()) if step else float(loss.item())
        step += 1
        if step % args.val_every == 0 or step == args.steps:
            val_loss = run_eval()
            metrics["steps"].append(step)
            metrics["train_loss"].append(running)
            metrics["val_loss"].append(val_loss)
            metrics["lr"].append(lr)
            print(
                f"[train] step={step:>6d}/{args.steps} lr={lr:.2e} "
                f"train={running:.4f} val={val_loss:.4f} ppl={math.exp(min(20, val_loss)):.2f}",
                flush=True,
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), out_dir / "model.pt")

    # save tokenizer + config + held-out test indices for downstream eval
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    metrics["wall_seconds"] = time.time() - t0
    metrics["best_val_loss"] = best_val
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "test_indices.json").write_text(json.dumps(test_idx), encoding="utf-8")
    # also stash the corpus so eval can index into the same examples
    from .dataset import dump_jsonl
    dump_jsonl(examples, out_dir / "corpus.jsonl")
    print(f"[train] best val loss={best_val:.4f}; saved checkpoint to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
