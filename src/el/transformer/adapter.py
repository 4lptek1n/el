"""Inference adapter — bridges the executor and an optional trained model.

The adapter loads a checkpoint if one exists at the canonical location,
and exposes `propose_plan(intent) -> list[Action]`. If PyTorch or the
checkpoint is missing, the adapter returns None, and the executor falls
through to self-play.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import inspect

from ..config import Config
from ..intent import Intent
from ..primitives import Action, PRIMITIVES
from .tokenizer import ActionTokenizer


def _validate_actions(actions: list[Action]) -> list[Action]:
    """Drop actions whose primitive is unknown or whose required kwargs
    are missing. Returns the surviving prefix (stop at first invalid)."""
    out: list[Action] = []
    for a in actions:
        fn = PRIMITIVES.get(a.name)
        if fn is None:
            break
        sig = inspect.signature(fn)
        provided = {k for k, _ in a.kwargs}
        required = [
            p.name for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        if any(r not in provided for r in required):
            break
        out.append(a)
    return out


@dataclass
class TransformerAdapter:
    checkpoint_dir: Path
    tokenizer: ActionTokenizer
    model: object | None = None

    @classmethod
    def try_load(cls, config: Config) -> Optional["TransformerAdapter"]:
        # First look in the user's state dir; fall back to the bundled
        # checkpoint shipped inside the repo (artifacts/el/checkpoints/...).
        candidates = [
            config.state_dir / "checkpoints" / "action-transformer",
            Path(__file__).resolve().parents[2].parent / "checkpoints" / "action-transformer",
        ]
        ckpt_dir: Optional[Path] = None
        for cand in candidates:
            if (cand / "tokenizer.json").exists():
                ckpt_dir = cand
                break
        if ckpt_dir is None:
            return None
        tok_path = ckpt_dir / "tokenizer.json"
        try:
            tokenizer = ActionTokenizer.load(tok_path)
        except Exception:
            return None
        model = None
        try:
            import torch

            from .model import ActionTransformer, ModelConfig

            cfg_path = ckpt_dir / "config.json"
            model_path = ckpt_dir / "model.pt"
            if cfg_path.exists() and model_path.exists():
                cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
                cfg = ModelConfig(**cfg_data)
                model = ActionTransformer(cfg)
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.eval()
        except Exception:
            model = None
        return cls(checkpoint_dir=ckpt_dir, tokenizer=tokenizer, model=model)

    def propose_plan(self, intent: Intent) -> list[Action]:
        if self.model is None:
            return []
        try:
            import torch

            prefix = self.tokenizer.encode_command(intent.to_dict())
            ids = torch.tensor([prefix], dtype=torch.long)
            eos = self.tokenizer.tid("<eos>")
            out = self.model.generate(ids, max_new=96, eos_id=eos)
            tokens = self.tokenizer.decode(out[0].tolist())
            return _validate_actions(_decode_actions(tokens))
        except Exception:
            return []


def _decode_actions(tokens: list[str]) -> list[Action]:
    actions: list[Action] = []
    current_name: str | None = None
    current_kwargs: list[tuple[str, str]] = []
    current_key: str | None = None
    current_value_words: list[str] = []

    def _flush_kv() -> None:
        nonlocal current_key, current_value_words
        if current_key is not None:
            current_kwargs.append((current_key, " ".join(current_value_words)))
        current_key = None
        current_value_words = []

    def _flush_action() -> None:
        nonlocal current_name
        if current_name is not None:
            _flush_kv()
            actions.append(Action(name=current_name, kwargs=tuple(current_kwargs)))
        current_name = None
        current_kwargs.clear()

    state = "init"
    for tok in tokens:
        if tok.startswith("<prim:"):
            _flush_action()
            current_name = tok[len("<prim:") : -1]
            state = "in_action"
        elif tok == "<arg_k>":
            _flush_kv()
            state = "key"
        elif tok == "<arg_v>":
            state = "value"
            current_value_words = []
        elif tok == "<eos>":
            _flush_action()
            break
        elif tok.startswith("w:"):
            word = tok[2:]
            if state == "key":
                current_key = (current_key or "") + word
            elif state == "value":
                current_value_words.append(word)
    _flush_action()
    return actions
