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

### Update 2 — interneuron is LEARNED, not hand-configured

The previous version hand-set the interneuron's receptive field at the
input cells. That has now been replaced with a fully local learning
rule (`learn_receptive_field_from_pattern` + threshold calibration):

- **Phase 2 — receptive field discovery**: the interneuron observes the
  *sharp* (pre-relax) input pattern and applies Hebbian + L1 normalization
  with continuous decay. Mass concentrates competitively on cells that
  are consistently active during (1,1) examples — i.e. the input cells.
  The diffused (post-relax) field was useless for this because heat
  blurs across the grid and Hebbian cannot localize. Using the sharp
  injected pattern is the key insight.
- **Phase 3 — threshold calibration**: each interneuron measures its
  own drive on each XOR case (a local quantity, just `sum(w_in · T)`)
  and places its threshold between `max(positive_drives)` and
  `min(coincidence_drives)`. No backprop, no global error.

The receptive field reliably converges to ~100% mass at the two input
cells (test: `test_interneuron_receptive_field_is_learned_at_input_cells`,
threshold ≥95% mass at inputs). Combined with the trained excitatory
field, this gives **4/4 XOR on 20/20 seeds with no hand-set parameters**.

The full pipeline is now: state-dependent thermal field +
gated Hebbian + supervised nudge (excitatory) + Hebbian/L1
normalization + drive-statistics calibration (inhibitory). All local,
all reproducible.

**Demo**:
```bash
el thermo-xor --seeds 5
```
Reports per-seed accuracy, raw output, inhibition strength, and net
output for each of the four XOR cases.

### Update 3 — temporal dimension (sequence learning)

The static field has no memory of *order*. Two additions enable
sequence learning with the same local-rules philosophy:

1. **Eligibility trace `E[r,c]`** — a low-pass of recent temperature:
   `E ← decay·E + (1−decay)·T`. Captures "this cell was hot recently"
   even after T has decayed.
2. **STDP-style plasticity** — replaces `pre_now × post_now` with
   `pre_trace × post_now`. Edges where the pre-cell led the post-cell
   in time get strengthened. Causal, temporally asymmetric, fully local.

**Two-channel substrate (validated)**:
Each edge in the Field now carries TWO weights:
  - `C_right`, `C_down` — symmetric conductance (heat flows equally
    in both directions). Updated by ordinary Hebbian rules
    (`plasticity.hebbian_update`).
  - `B_right`, `B_down` — directional **bias** (positive favors
    forward flow, negative favors backward). Updated by anti-symmetric
    STDP (`sequence.stdp_hebbian_update`).
Effective forward conductance = max(C + B, 0); backward = max(C − B, 0).
With B = 0 the field is direction-neutral (drop-in replacement for the
old single-channel field). XOR training (which only updates C) is
unchanged and the 7 field + 4 plasticity tests still pass.

**Validated — event-gated directional pairwise binding**:
(deliberately modest scope — this is two-event temporal direction
learning with a gated plasticity window, NOT yet multi-step sequence
learning with variable lags.)
Training rule: present A (clamped 5 steps), gap 2 steps, then at the
**event boundary** (the moment B is first clamped) apply STDP once on
B. After that, run B's hold and apply symmetric Hebbian on C. This
captures the clean "A leads B" snapshot before E_B grows large.

After 60 epochs of A→B pairing (averaged across 8 seeds, lr=0.07,
bias_clip=0.6):
  - Cue A → response at B = **0.049**
  - Cue B → response at A = **0.024**
  - Directional ratio = **~2.07×** (A→B significantly stronger than B→A)
  - Trained vs naive baseline = **+0.028** (substrate also learned the
    coactivity, on top of direction)

This is genuine directional pairwise binding with purely local rules
— no backprop, no supervised target, no global error signal. The
substrate represents temporal order *in its physical asymmetry of heat
conductance*. Three tests pin this down (the third is the symmetry
control that rules out probe-geometry artifact):
  - `test_directional_stdp_breaks_bidirectional_symmetry` (train A→B,
    assert A→B > 1.4× B→A across 5 seeds)
  - `test_directional_stdp_reverses_when_training_order_reverses`
    (train B→A on the same protocol, assert the asymmetry FLIPS:
    B→A > 1.4× A→B). This is the geometry-rules-out test the
    architect required.
  - `test_predict_next_creates_temporal_association_above_naive_baseline`
    (asserts trained > untrained by ≥0.003)

**Limits & next steps**:
The 2.07× ratio is solid but modest in absolute units (~0.025
temperature difference). Pushing it further requires either a larger
field, longer training, or graded inputs (not just clamps). The
*next* useful task is sequence **classification** (AB→1 vs BA→0):
the substrate has the directional capacity now, so a simple readout
(e.g. compare T at two probe sites after presenting cue A) should
already discriminate the two orderings. That would be the first
honest "the system learned a temporal pattern" demo.

Pipeline files: `el/src/el/thermofield/sequence.py` (EligibilityTrace,
stdp_hebbian_update, train_predict_next), `el/src/el/thermofield/field.py`
(directional B channels in step()).
Tests: `el/tests/thermofield/test_sequence.py` (5 tests, all green).
