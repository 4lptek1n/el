"""el — LLM-free, action-grounded, locally-learning PC agent.

The thesis: LLMs invested billions of parameters in compressed text patterns
but have almost no direct world access. A PC has unlimited world access
(files, network, processes, shell, internet) but no decision layer. Instead
of bolting a few tool calls onto an LLM, el starts from the PC and adds a
minimal two-memory decision layer on top.

Two memories, inspired by McClelland's Complementary Learning Systems (1995):

- Skill registry: exact episodic memory, grows by local plasticity on
  verified action outcomes. Hippocampus-like.
- Action Transformer: statistical generalization over (command, state,
  action, outcome) tokenized sequences. Neocortex-like.

The honest claim is NOT "smarter than GPT-4." It is that the specific
combination - no LLM + real OS substrate + dual symbolic-neural memory +
action tokenization - is unoccupied territory and may produce something
structurally different from frontier LLM agents.
"""
from .intent import Intent
from .parser import Parser, parse
from .registry import SkillRegistry, Skill
from .executor import Executor, ExecutionResult
from .rewards import score_outcome
from .selfplay import SelfPlay

__all__ = [
    "Intent",
    "Parser",
    "parse",
    "SkillRegistry",
    "Skill",
    "Executor",
    "ExecutionResult",
    "SelfPlay",
    "score_outcome",
]
__version__ = "0.1.0"
