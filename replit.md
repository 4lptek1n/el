# Workspace

## Overview

pnpm workspace monorepo using TypeScript (shared backend infra) + Python project `el` cloned from GitHub.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)
- **Python version**: 3.11

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## el — Python PC Agent Project

Source: https://github.com/4lptek1n/el
Location: `/home/runner/workspace/el/`
Installed as editable package (dev mode) with Python 3.11.

### el Key Commands (run from `/home/runner/workspace/el/`)

```bash
# CLI is at:
/home/runner/workspace/.pythonlibs/bin/el

# Seed the skill registry (run once at the start)
el seed

# Run a natural language command
el run "list files"
el run "git durumu"

# Parse a command (see how parser interprets it)
el parse "dosyaları listele"

# Show skill registry stats
el stats

# Show recent events
el events

# List all grammar verbs
el verbs

# Run tests
cd el && python3 -m pytest

# Lint with ruff
cd el && python3 -m ruff check src/
```

### Optional extras
- `python3 -m pip install -e "el[train]"` — install PyTorch for the Action Transformer
- `python3 -m pip install -e "el[http]"` — install httpx for HTTP primitives

### HDM-EL: Hyperdimensional + Active Inference World Model

A third memory layer added on top of the registry (epizodik) and Action
Transformer (statistical). The world model is **predictive**: it learns
`(state, action) → outcome` mappings as bound hypervectors, and the
Active Inference planner uses Expected Free Energy to rank plans.

**Architecture**:
- `el.worldmodel.hdc` — bipolar 10K-dim vectors with bind / bundle / permute
- `el.worldmodel.world` — associative memory of (state⊗action)→outcome triples
- `el.worldmodel.planner` — EFE = -(pragmatic + 0.4 · epistemic), lower is better
- `el.worldmodel.store` — persists to `<state_dir>/worldmodel.npz`

Pure NumPy, CPU-only. ~1 MB per ~1000 experiences. The executor
auto-observes every command, so the world model grows passively.

**Theoretical basis**: Kanerva HDC (1988-2009), Friston Free Energy
Principle (2010), McClelland Complementary Learning Systems (1995).

**New CLI commands**:
```bash
# Plan via Active Inference (does NOT execute)
el think "list this folder" --top 5

# Show world model size & stats
el world-stats

# Erase the world model
el world-reset
```

**Tests**: `python3 -m pytest tests/worldmodel/ -v` (17 tests covering
HDC properties, world model prediction & persistence, EFE ranking).

## Thermofield experiment (radical, exploratory)

A 4th memory direction explored at the user's prompting: replace
[structure + weights + time] with a **single thermodynamic field** —
distance-weighted diffusion across a 2D grid of cells with a scalar
temperature, learned only via local plasticity (Hebbian + 1-cell
supervised nudge). No backprop. No global signal. Pure NumPy.

**Architecture** (`el.thermofield`):
- `field.py` — 2D grid with state-dependent conductivity (`c_eff = c·(1+α·T̄)`),
  Dirichlet boundary clamping for inputs, energy decay
- `plasticity.py` — Hebbian co-activation rule + supervised nudge at
  output cells only (1-cell-deep error, not backprop)
- `runner.py` — XOR/OR training loops

**Honest finding**: the state-dependent diffusion gives the field a
**naturally XOR-shaped response without any training** — `(1,1)` is
suppressed below `(0,1)` and `(1,0)` because two hot inputs short-circuit
each other before reaching the output. Current local plasticity rules
execute correctly and keep the field stable, but **do not yet reliably
sharpen this response**: training with a fixed 0.5 threshold collapses
to ~50% accuracy because the four XOR examples fight over the same
shared conductivity paths.

This is a real research finding, not a finished win. The substrate is
working; smarter plasticity (e.g., trainable readout on a fixed reservoir,
or anti-Hebbian competition) is the next direction.

**Demo**:
```bash
el thermo-demo --epochs 200
```
Outputs untrained natural response, post-training response, and field stats.

**Tests**: `python3 -m pytest tests/thermofield/ -v` (14 tests covering
field substrate, plasticity rules, natural XOR shape, training stability).

### Update — 4/4 XOR with two-population architecture

Following the user's "lateral inhibition / interneuron" insight, added
`el.thermofield.interneurons` — a separate inhibitory population with
its own receptive field (`w_in`), output gain (`w_out`), and threshold.
The output reading becomes `net = max(0, raw − sum(w_out · I_k))`.

Task split: the excitatory thermal field is trained only on the three
positive XOR cases via local plasticity. A coincidence-detector
interneuron (theta=1.0, listens to both inputs) suppresses the (1,1)
case at readout. **Result: 4/4 XOR accuracy on 5/5 random seeds**
(see `test_xor_with_interneurons_solves_4_of_4_across_seeds`).

This is the first reproducible win. With the local rules and geometry
tried so far, the single-population approach plateaued at 3/4 because
all four examples share the same conductivity paths; adding a second
population (one whose job is to SUBTRACT from the output) breaks that
shared-path conflict. (We do not claim XOR is formally unreachable
with one population — only that no local rule we tried got past 3/4.)

**Open research problem**: learn the interneuron weights from purely
local rules. Pure Hebbian saturates (heat distributions of single- and
double-input cases overlap heavily after diffusion). Contrastive
Hebbian collapses w_in toward zero. The current implementation hand-
configures the interneuron as a coincidence detector and only learns
the excitatory field; learning the interneuron remains future work.

**Demo**:
```bash
el thermo-xor --seeds 5
```
Reports per-seed accuracy, raw output, inhibition strength, and net
output for each of the four XOR cases.
