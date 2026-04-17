# Contributing to el

Bug reports, methodological critiques, new primitives, new seed verbs, and
reproducibility runs are all welcome.

## Code of conduct

Be civil. Argue with the work, not the person. Reproducible counter-examples
beat opinions.

## Setting up a dev environment

```bash
git clone https://github.com/4lptek1n/el.git
cd el
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest -q
ruff check .
```

Run the offline demo suite (no network, no LLM, no GPU):

```bash
el seed
el demo --all --offline
```

## Architecture at a glance

Two memories, with symmetric responsibilities:

| Memory | Role | Implementation |
|---|---|---|
| Skill registry | exact episodic memory, hippocampus-like | `src/el/registry.py` (SQLite) |
| Action Transformer | statistical cortex-like generalization | `src/el/transformer/` (PyTorch) |

Control flow:

```
user command
    │
    ▼
parser.py  ────▶  Intent(verb, object, scope, args)
    │
    ▼
executor.py
    │
    ├── registry.lookup(Intent) ── HIT ──▶  run primitives, log outcome ──▶ done
    │                               │
    │                               └── log feeds transformer training corpus
    ▼
  MISS
    │
    ├── transformer.propose_plan(Intent) ──▶ candidate plan
    │                                         │
    │                                         ▼
    └── selfplay.generate(N=5) ──▶ candidate plans ──▶ sandboxed execution
                                                          │
                                                          ▼
                                             best plan ──▶ new skill in registry
```

## Adding a new verb to the grammar

Verbs live in `src/el/parser.py`. Each verb is a `VerbSpec`:

```python
VerbSpec(
    canonical="summarize",
    aliases=["özetle", "summarise", "özet"],
    objects=["pdf", "file", "url", "folder"],
    scopes=["this_folder", "path", "url"],
)
```

To add a verb:

1. Write the `VerbSpec` in `parser.py`.
2. Write a seed skill in `src/el/seed/bootstrap.py` that handles the
   canonical case.
3. Add at least one test in `tests/test_parser.py`.
4. Add at least one demo to `src/el/demos.py` if the verb is user-facing.

## Adding a primitive

Primitives live in `src/el/primitives.py`. Each primitive has a narrow
signature and a timeout. Destructive primitives (rm, mv into a parent,
package installs) must call `safety.require_confirmation()`.

## Pull request checklist

- `pytest -q` passes locally.
- `ruff check .` is clean.
- If you add a verb, you also add a demo and a test.
- If you change reward shape, you update `docs/paper-draft.md`.

## Reporting a reproducibility failure

Open an issue with:

- The exact command line.
- The contents of `~/.el_state/registry.sqlite3` (or equivalent path).
- `git rev-parse HEAD`.
- The last 200 lines of `~/.el_state/events.jsonl`.
