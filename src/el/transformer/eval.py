"""Held-out evaluation for the Action Transformer.

Computes per-example:
  - first-action-name accuracy (does the model emit the right primitive first?)
  - exact-action-sequence accuracy (do all primitive names match in order?)
  - average action-name overlap (Jaccard over primitive names)

Usage:
    python -m el.transformer.eval \
        --checkpoint checkpoints/el-action-transformer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import _decode_actions
from .dataset import load_jsonl
from .model import ModelConfig, TORCH_AVAILABLE, build_model
from .tokenizer import ActionTokenizer


def main() -> int:
    p = argparse.ArgumentParser(description="evaluate a trained Action Transformer")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-eval", type=int, default=300)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--report", default=None, help="path to write JSON report")
    args = p.parse_args()

    if not TORCH_AVAILABLE:
        print("error: torch not installed")
        return 2

    import torch

    ckpt = Path(args.checkpoint)
    cfg = ModelConfig(**json.loads((ckpt / "config.json").read_text()))
    tok = ActionTokenizer.load(ckpt / "tokenizer.json")
    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu"))
    model.eval()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    model = model.to(device)

    test_indices = json.loads((ckpt / "test_indices.json").read_text())
    examples = load_jsonl(ckpt / "corpus.jsonl")
    test_examples = [examples[i] for i in test_indices][: args.max_eval]
    print(f"[eval] evaluating {len(test_examples)} held-out examples on {device}")

    n = 0
    n_first = 0
    n_seq = 0
    jacc_sum = 0.0
    misses: list[dict] = []
    eos = tok.tid("<eos>")
    with torch.no_grad():
        for ex in test_examples:
            prefix = tok.encode_command(ex.intent.to_dict())
            ids = torch.tensor([prefix], dtype=torch.long, device=device)
            out = model.generate(ids, max_new=64, eos_id=eos)
            tokens = tok.decode(out[0].tolist())
            pred = _decode_actions(tokens)
            gold_names = [a.name for a in ex.actions]
            pred_names = [a.name for a in pred]
            n += 1
            if pred_names and gold_names and pred_names[0] == gold_names[0]:
                n_first += 1
            if pred_names == gold_names:
                n_seq += 1
            inter = len(set(pred_names) & set(gold_names))
            uni = max(1, len(set(pred_names) | set(gold_names)))
            jacc_sum += inter / uni
            if pred_names != gold_names and len(misses) < 20:
                misses.append({
                    "raw": ex.intent.raw, "verb": ex.intent.verb,
                    "gold": gold_names, "pred": pred_names,
                })

    report = {
        "n_eval": n,
        "first_action_accuracy": n_first / max(1, n),
        "sequence_accuracy": n_seq / max(1, n),
        "name_jaccard": jacc_sum / max(1, n),
        "sample_misses": misses,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "sample_misses"}, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[eval] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
