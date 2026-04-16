"""Tiny Action Transformer in PyTorch (importable even when torch is absent).

Architecture is a stock decoder-only transformer. Defaults are tuned for a
quick CPU smoke run (~1M params). The `scale="h200"` preset bumps the
width/depth to ~120M so it fits comfortably on a single H200.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    max_len: int = 512
    dropout: float = 0.0

    @classmethod
    def preset(cls, name: str, vocab_size: int) -> "ModelConfig":
        if name == "tiny":
            return cls(vocab_size=vocab_size, d_model=128, n_layers=4, n_heads=4, d_ff=512)
        if name == "small":
            return cls(vocab_size=vocab_size, d_model=512, n_layers=8, n_heads=8, d_ff=2048)
        if name == "h200":
            return cls(vocab_size=vocab_size, d_model=1024, n_layers=16, n_heads=16, d_ff=4096)
        raise ValueError(f"unknown preset: {name}")

    def approx_params(self) -> int:
        emb = self.vocab_size * self.d_model
        per_layer = 4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff
        return emb + self.n_layers * per_layer


def build_model(cfg: ModelConfig) -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Install el[train].")
    return ActionTransformer(cfg)


if TORCH_AVAILABLE:

    class ActionTransformer(nn.Module):
        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.stack = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
            self.ln = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
            b, t = ids.shape
            pos_ids = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
            x = self.tok(ids) + self.pos(pos_ids)
            mask = torch.triu(torch.ones(t, t, device=ids.device), diagonal=1).bool()
            h = self.stack(x, mask=mask)
            h = self.ln(h)
            return self.head(h)

        @torch.no_grad()
        def generate(self, prefix_ids: "torch.Tensor", *, max_new: int = 64, eos_id: int | None = None) -> "torch.Tensor":
            ids = prefix_ids.clone()
            for _ in range(max_new):
                logits = self.forward(ids[:, -self.cfg.max_len:])
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
                if eos_id is not None and (next_id == eos_id).all().item():
                    break
            return ids
