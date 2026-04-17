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
