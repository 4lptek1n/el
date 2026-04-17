# el

**An LLM-free, action-grounded, locally-learning PC agent with a dual-memory
decision layer.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Abstract

Frontier LLM agents place a massive language model at the center and attach
narrow tools to it. `el` starts from the opposite end: the PC. A PC already
has unlimited world access - shell, filesystem, network, processes - and
*no* decision layer. `el` adds a minimal, two-memory decision layer on
top. No GPT-4. No Claude. No model-in-the-loop at inference time.

The two memories are a literal implementation of McClelland, McNaughton,
and O'Reilly's Complementary Learning Systems hypothesis (1995):

1. **Skill registry** - SQLite-backed symbolic memory of
   `(command, action_sequence, outcome)` triples. Grows by *local
   plasticity* on verified outcomes: weights go up on success, down on
   failure, slowly decay when unused. Hippocampus-like: episodic, exact,
   one-shot.

2. **Action Transformer** - a small (50-200M) Decision-Transformer-style
   model trained on the same triples tokenized into an action-grounded
   vocabulary (`<verb:list>`, `<prim:file_write>`, `<arg_v>`,
   `<reward:3>`, ...). Neocortex-like: statistical, compositional,
   slow-building.

The two memories feed each other. When the registry matches a command, it
executes instantly and logs the outcome as new training data for the
transformer. When it misses, the transformer (if trained) or a self-play
planner proposes a plan; the winning plan graduates into a new skill.

The honest claim is **not** "smarter than GPT-4." It is:

> The specific combination - **no LLM + real OS substrate + dual
> symbolic-neural memory + action tokenization** - is unoccupied territory.
> Several properties that are structurally hard for LLM-centric agents
> (persistent memory across months, offline operation, deterministic
> replay, direct OS action, compounding personal skill) fall out of this
> architecture for free.

## Architecture

```
user command
    │
    ▼
parser  ───▶  Intent(verb, object, scope, args)
    │
    ▼
executor
    │
    ├── registry.lookup(Intent) ── hit  ──▶  run actions ──▶ reinforce
    │                               │
    │                               └── log feeds transformer corpus
    ▼
  miss
    ├── transformer.propose_plan(Intent) ──▶ plan ──▶ run ──▶ graduate
    └── selfplay.generate(N=5)            ──▶ plan ──▶ run ──▶ graduate
```

- **parser** (`src/el/parser.py`) - deterministic bilingual grammar
  (Turkish + English), ~25 verbs, no ML.
- **primitives** (`src/el/primitives.py`) - timeout-bounded OS effectors:
  `sh`, `http_get`, `http_download`, `file_read`, `file_write`, `grep`,
  `disk_usage`, `git_status`, `process_list`, ...
- **registry** (`src/el/registry.py`) - SQLite skill store with a local
  plasticity update rule and exponential decay.
- **selfplay** (`src/el/selfplay.py`) - verb-templated candidate plans,
  scored by a static feasibility heuristic.
- **transformer** (`src/el/transformer/`) - tokenizer, dataset loader,
  model, and training loop. PyTorch is optional - the rest of `el` works
  without it.
- **daemon** (`src/el/daemon.py`) - autonomous maintenance loop
  (`el daemon`).
- **rewards** (`src/el/rewards.py`) - outcome scorer; combines automatic
  signals (exit code, output presence, duration) with verb-specific
  heuristics and explicit `el rate` feedback.

## Installation

```bash
git clone https://github.com/4lptek1n/el.git
cd el
pip install -e .
```

For training:

```bash
pip install -e ".[train]"     # PyTorch
```

For dev:

```bash
pip install -e ".[dev]"       # pytest, ruff
```

Verify:

```bash
el --help
el verbs
el seed --from-bundled
el demo --all --offline
```

## Usage

```bash
el "list this folder"
el "bu klasörü listele"
el "summarize the PDFs in ./papers"
el "research https://example.com"
el "git status"
el "report disk usage here"
el daemon --iterations 3 --tick 1
```

Each run prints a JSON record of the resolved `Intent`, the primitives
executed, and the reward. Every execution is logged to
`~/.el_state/events.jsonl` and the skill registry at
`~/.el_state/registry.sqlite3`.

Inspect what the parser thinks you said:

```bash
el parse "bu klasördeki pdf'leri özetle"
```

Reinforce the last run explicitly:

```bash
el rate up    # coming in v0.2; v0 uses automatic reward only
```

## Demo suite

The repo ships with **15 end-to-end demo commands** that double as the CI
benchmark. Run them:

```bash
el demo --all --offline
```

All 14 offline demos are expected to pass from a fresh checkout after
`el seed --from-bundled`. The one online demo (`research https://example.com`)
is skipped when `--offline` is set. The catalog lives at
[`src/el/demos.py`](src/el/demos.py).

## Training the Action Transformer

A trained tiny checkpoint is **shipped with the repo** at
[`checkpoints/action-transformer/`](checkpoints/action-transformer/).
The adapter loads it automatically — you don't have to train anything to
get a working transformer-route.

### Reproducing the bundled checkpoint (CPU, ~3 min)

```bash
pip install -e ".[train]"
el train --preset tiny --steps 3000 --batch 32 --synth-per-verb 200
el transformer-eval --max-eval 618
```

The corpus is built deterministically from the parser's verb grammar +
bundled tldr/man examples + a synthetic schema bank, so the same
`--seed` always produces the same training set. Defaults give 6,185
examples; the splitter is **group-aware** — every (intent + action
sequence) signature lives in exactly one of train/val/test, so
near-duplicates from synthetic combinatorics cannot leak across splits.
The `tiny` preset's bundled run yields 5,023 train / 679 val / 483 test
over a 443-token vocab.

**Bundled checkpoint stats** (`tiny` preset, CPU, seed=42, group-aware
split, reward-conditioned prefix):

| metric | value |
|---|---|
| parameters | 840,832 |
| training steps | 3,000 |
| wall-clock (CPU) | ~165 s |
| best validation loss | 0.364 |
| held-out first-action accuracy | **79.3 %** |
| held-out exact-sequence-of-names accuracy | **79.3 %** |
| held-out kwarg-key-set match | 0 % (see caveat below) |
| majority-class first-action baseline | 27.7 % |
| name-Jaccard | 0.79 |

That's ~3× the majority-class baseline on a leakage-free split — the
model is learning a genuine signal from the (intent → action sequence)
mapping, not memorising near-duplicates.

**Caveats — what the numbers do *not* claim**:
- The 79.3 % is **first-action / sequence-of-primitive-names** accuracy.
  The decoder still hallucinates duplicate / extra kwarg keys (e.g.
  predicting `('cmd', 'ext', 'timeout')` instead of `('cmd', 'timeout')`),
  so an exact key-set match scores 0 % at this size — the kwarg
  vocabulary entered the embedding only at this run and the `tiny`
  model needs more capacity / steps to lock the kwarg structure down.
- At inference, the executor **fills argument values from the parsed
  intent's argument bag**, so the model effectively only needs to
  predict the right primitive name for the LLM-free pipeline to work
  end-to-end. The transformer is the routing/ordering layer; the
  intent parser is the value layer.
- This is a `tiny` (840 K params) bundled checkpoint trained on CPU in
  under three minutes. The `small` and `h200` presets are intended for
  the real numbers — see *Bigger models* below.

### Bigger models

```bash
# Single GPU (~25M params, vocab≈400):
el train --preset small --steps 5000 --batch 64 --device cuda

# H200 / 4090 sweet spot (~120M params):
el train --preset h200 --steps 200000 --batch 128 --device cuda
```

After training to a custom location, place the artifacts under
`~/.el_state/checkpoints/action-transformer/` (or pass `--out` there
directly) — the adapter prefers that over the bundled checkpoint.

## Training data

Three seed streams, all committed or reproducible from public sources:

| Stream | Shipped bundle | External source |
|---|---|---|
| Core skills | `seed_data/core_skills.jsonl` | hand-curated |
| tldr-pages subset | `seed_data/tldr_subset.jsonl` | [tldr-pages/tldr](https://github.com/tldr-pages/tldr) (CC-BY-4.0) |
| Man-page synthetics | `seed_data/man_synth.jsonl` | generated |
| NL2Bash (optional) | not bundled | [IBM NL2Bash](https://github.com/IBM/clai/tree/master/clai/server/plugins/nlc2cmd) |

Plus the skill registry's own accumulated rows, exportable via:

```bash
el export-training --out training.jsonl
```

## Reproducibility

`./reproduce.sh` at the repo root does the full replay from a clean
checkout, with no network, no LLM, no GPU:

```bash
./reproduce.sh
```

The script installs `el` in editable mode, seeds the registry from the
bundled corpus, runs the offline demo suite, and exits nonzero on any
regression. A one-line verdict is printed per demo and the full JSON
report is at `.el_state_repro/demo_report.json`.

## Continuous integration

The CI workflow is shipped under
[`docs/github-workflow-template/ci.yml`](docs/github-workflow-template/ci.yml)
(see the README in that directory for the one-line install). It runs:

- `ruff check .`
- `pytest -q`
- The offline demo suite.

It is shipped as a template instead of `.github/workflows/ci.yml` because
the OAuth token used to push the initial release intentionally lacks the
`workflow` scope.

## Project layout

```
el/
├── src/el/
│   ├── cli.py                  # entry point: `el ...`
│   ├── parser.py               # deterministic bilingual grammar
│   ├── intent.py               # Intent dataclass
│   ├── primitives.py           # ~25 OS primitives
│   ├── registry.py             # SQLite skill registry + plasticity
│   ├── executor.py             # parse → lookup → selfplay → reinforce
│   ├── selfplay.py             # templated candidate plans
│   ├── rewards.py              # automatic + heuristic reward model
│   ├── daemon.py               # autonomous maintenance loop
│   ├── demos.py                # the 15 demo commands
│   ├── seed/bootstrap.py       # starter skills from bundled corpora
│   └── transformer/
│       ├── tokenizer.py        # action-grounded vocabulary
│       ├── dataset.py          # seed-corpus loader
│       ├── model.py            # PyTorch decoder-only transformer
│       ├── train.py            # training loop (tiny/small/h200 presets)
│       └── adapter.py          # inference bridge for executor
├── seed_data/                  # committed seed corpora (JSONL)
├── tests/                      # pytest suite
├── scripts/
│   ├── seed_registry.py
│   ├── run_demos.sh
│   └── modal_train.py          # Modal H200 entrypoint
├── docs/
│   ├── paper-draft.md          # short technical note
│   └── github-workflow-template/ci.yml
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
└── reproduce.sh
```

## Limitations

1. **v0 does not beat frontier LLM agents on open-ended language tasks.**
   It is not supposed to. The claim here is architectural, not benchmark.
2. **The Action Transformer is trained on a seed corpus plus accumulated
   user logs.** Open-domain generalization to verbs the grammar does not
   cover is future work.
3. **Safety is perimeter-style, not sandboxed.** Destructive primitives
   require explicit `--yes`; offline mode blocks network. There is no
   jailed subprocess. The threat model is buggy plans, not adversarial
   prompts.
4. **Shell, files, and HTTP only in v0.** Screen / mouse / keyboard
   control and package installation are deferred to v1.
5. **Reward is outcome-shaped but still coarse.** We use automatic exit
   codes + verb-specific heuristics. Explicit `el rate` is stubbed; the
   full user-feedback loop lands in v0.2.

## Citation

```bibtex
@software{keser_el_2026,
  author  = {Keser, Alptekin},
  title   = {{el}: an LLM-free, action-grounded PC agent with dual-memory architecture},
  year    = {2026},
  url     = {https://github.com/4lptek1n/el},
  version = {0.1.0}
}
```

See also [`CITATION.cff`](CITATION.cff) and
[`docs/paper-draft.md`](docs/paper-draft.md).

## References

- McClelland, J. L., McNaughton, B. L., and O'Reilly, R. C. (1995).
  *Why there are complementary learning systems in the hippocampus and
  neocortex: insights from the successes and failures of connectionist
  models of learning and memory.* Psychological Review 102(3):419-457.
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

## License

MIT — see [LICENSE](LICENSE).

## Contact

Alptekin Keser — keseralptekin@gmail.com
