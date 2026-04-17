# el: An LLM-free PC agent with dual-memory architecture

**Alptekin Keser** · keseralptekin@gmail.com · 2026

## Abstract

Frontier LLM agents invert the traditional embodied-AI stack: they place
a massive language model at the center and attach narrow tools to it. That
arrangement has produced striking fluency but unreliable action. It is
also structurally at odds with two phenomena that the cognitive-science
literature treats as load-bearing: *persistent, locally updated memory*
(McClelland, McNaughton, and O'Reilly, 1995) and *grounded, outcome-driven
learning* (Friston, 2010; Bi and Poo, 1998). We describe `el`, a small
experimental PC agent that starts from the opposite end. It has no
language model. Instead it runs on a PC - which already has unlimited world
access through shell, filesystem, network, and processes - and adds a
minimal two-memory decision layer on top. Memory one is a SQLite-backed
symbolic skill registry, grown by local plasticity on verified action
outcomes. Memory two is a small Decision-Transformer-style Action
Transformer (50-200M params) trained on tokenized
`(command, state, action, outcome)` sequences. The two memories feed each
other. We frame this as a literal implementation of Complementary Learning
Systems (CLS) for PC agents. We report v0 results on a 15-command demo
suite executed end-to-end without any LLM in the loop.

## 1. Thesis

LLMs invested billions of parameters in compressed text patterns but have
almost no direct world access. A PC has unlimited world access - shell,
processes, filesystem, internet, subprocess isolation, package managers -
but no decision layer. The common reflex is to bolt a few tool calls onto
an LLM. We do the opposite: start from the PC and add a minimal decision
layer. The claim is *not* that `el` is smarter than GPT-4 on language
tasks. The claim is that the architecture - *no LLM + real OS substrate +
dual symbolic-neural memory + action tokenization* - occupies unexplored
territory, and that several properties that are structurally hard for
LLM-centric agents (persistent memory, offline operation, deterministic
replay, direct OS action, compounding personal skill) fall out of it for
free.

## 2. Architecture

```
user command
    │
    ▼
parser  ───▶  Intent(verb, object, scope, args)
    │
    ▼
executor
    ├── registry.lookup(Intent)  ── hit  ──▶  run actions ──▶ reinforce
    │                                │
    │                                └── log ──▶ transformer corpus
    ▼
  miss
    ├── transformer.propose_plan(Intent) ──▶ candidate ──▶ run ──▶ graduate
    └── selfplay.generate(N) ──────────────▶ candidate ──▶ run ──▶ graduate
```

### 2.1 Parser

A deterministic bilingual grammar (Turkish + English) covering ~25 verbs,
object slots, and scope slots. No ML. Ambiguity yields multiple candidate
`Intent`s ranked by a simple surface score. Commands outside the grammar
return a low-confidence `unknown` intent rather than a hallucinated one.

### 2.2 Primitive action layer

A set of narrow, timeout-bounded OS primitives (`sh`, `http_get`,
`http_download`, `file_read`, `file_write`, `file_move`, `grep`,
`disk_usage`, `git_status`, `process_list`, ...). Each primitive returns a
`PrimResult` with `(ok, stdout, stderr, data, duration_ms)`, which is the
uniform record the reward model and the training corpus both consume.
Destructive primitives respect a confirmation gate.

### 2.3 Skill registry

SQLite, schema in `src/el/registry.py`. Rows store
`(trigger_key, intent, action_sequence, success_count, failure_count,
weight, origin, last_used_at)`. The plasticity rule is:

```
w ← w + lr · (1 − w)      on success
w ← w − lr · w            on failure
```

with a slow exponential decay of all weights per daemon tick and deletion
of skills whose weight falls below a floor and whose success count is
zero. This is *local plasticity*: no gradient, no backprop, no global
loss.

### 2.4 Self-play planner

When the registry misses and the transformer has not yet been trained (or
declines to propose), the planner generates N candidate action sequences
from verb-specific templates, scores them by a static feasibility heuristic
(sequence length, presence of file-producing or network primitives,
consistency with the Intent's verb), and returns the top candidate. If
execution yields a reward above the graduation threshold, the candidate
becomes a new skill.

### 2.5 Action Transformer

A decoder-only transformer over a deterministic action-grounded vocabulary
(`<verb:V>`, `<obj:O>`, `<scope:S>`, `<prim:P>`, `<arg_k>`, `<arg_v>`,
`<reward:bucket>`). Trained on three seed streams - an NL2Bash subset, a
tldr-pages subset, and synthesized man-page examples - plus the skill
registry's own exported rows as they accumulate. Defaults: 50-200M
parameters. Training target: Modal H200. Loss is next-token prediction on
the tokenized action stream.

The transformer is *decoder of plans, not of language*. It never produces
free-form prose. Its output vocabulary is finite and every token decodes to
either a verb/primitive slot or a short keyword argument.

## 3. Why two memories

We claim no novelty for dual-memory per se. The specific arrangement -
exact-episodic hippocampus-like memory grown by local plasticity on
verified action outcomes, plus a statistical cortex-like generalizer
trained on the same triples in a tokenized action vocabulary - is a
literal reading of McClelland et al.'s CLS hypothesis, adapted for an OS
substrate. Prior art closest to us: Voyager (Wang 2023) showed that
accumulating skills beats monolithic policy, but relied on GPT-4 as its
world model. DreamCoder (Ellis 2021) showed that programs as skills can be
compositionally learned, but in a narrow lab environment. Decision
Transformer (Chen 2021) showed that RL can be recast as sequence modeling
conditioned on returns. Gato (DeepMind 2022) showed that a single
transformer can speak many modalities; but it required pretraining on many
demonstrations, and the action space was not the OS. `el` is the specific
combination: no LLM, real OS, dual memories, action tokenization.

## 4. v0 demo suite

We ship a committed list of 15 natural-language commands in
`src/el/demos.py`. CI runs the full suite end-to-end with `--offline`
(no network, no LLM, no GPU). The suite covers:

- listing / searching / inspecting files
- disk and process reports
- summary and report generation
- folder organization
- git status and log
- one networked demo (`research https://example.com`) behind a flag

All offline demos are expected to pass from a fresh checkout after
`el seed --from-bundled`. The networked demo is skipped when `--offline`.

## 5. Reproducibility

`./reproduce.sh` at repo root does the full replay from a clean checkout:

1. `pip install -e .`
2. `el seed --from-bundled` to build the starter registry from the
   committed seed corpus.
3. `el demo --all --offline --json` to run the suite.
4. Emit a pass/fail report and exit nonzero on any regression.

## 6. Limitations

1. *v0 does not beat frontier LLM agents on open-ended tasks.* It is not
   supposed to. The claim is architectural, not benchmark.
2. *The Action Transformer is trained on a seed corpus plus accumulated
   user logs; open-domain generalization is future work.* We target
   compositional generalization in v1.
3. *Safety is perimeter-style, not sandboxed.* Destructive primitives
   require explicit confirmation; no jailed subprocess. This is honest
   about threat model: the adversary is buggy candidate plans, not a
   malicious prompt.
4. *Only shell, files, HTTP in v0.* Screen/mouse/keyboard control is
   deferred.

## 7. References

- McClelland, J. L., McNaughton, B. L., and O'Reilly, R. C. (1995). *Why
  there are complementary learning systems in the hippocampus and
  neocortex.* Psychological Review 102(3):419-457.
- Wang, G. et al. (2023). *Voyager: An open-ended embodied agent with
  large language models.*
- Ellis, K. et al. (2021). *DreamCoder: Bootstrapping inductive program
  synthesis with wake-sleep library learning.*
- Chen, L. et al. (2021). *Decision Transformer: Reinforcement learning via
  sequence modeling.*
- Reed, S. et al. (2022). *A generalist agent (Gato).*
- Friston, K. (2010). *The free-energy principle: A unified brain theory?*
- Bi, G. and Poo, M. (1998). *Synaptic modifications in cultured
  hippocampal neurons: dependence on spike timing, synaptic strength, and
  postsynaptic cell type.*
- Mitchell, T. M. et al. (2010). *Never-ending learning (NELL).*
