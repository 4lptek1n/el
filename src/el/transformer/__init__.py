"""Action Transformer — the statistical half of dual-memory.

A small Decision-Transformer-style model (50-200M params by default) trained
on tokenized (command, state, action, outcome) sequences produced by both
the seed bootstrap corpus (NL2Bash, tldr-pages, man page synthetics) and
accumulated real user interaction logs.

Training targets Modal H200. Inference is pure-Python-inspectable from
checkpoint metadata even when PyTorch is not installed — the adapter
gracefully degrades to None.
"""
from .tokenizer import ActionTokenizer, SPECIAL_TOKENS
from .dataset import SeedExample, build_training_examples, dump_jsonl, load_jsonl

__all__ = [
    "ActionTokenizer",
    "SPECIAL_TOKENS",
    "SeedExample",
    "build_training_examples",
    "dump_jsonl",
    "load_jsonl",
]
