# Honest benchmark results — "Can we beat the rivals?"

Setup: substrate = thermofield (no backprop, online Hebb on shared
substrate). All comparisons are CPU only, no GPU, no special
hardware. Test images / probes are random but seed-locked.

## MNIST classification (10000 test)

| approach                       | params | acc (%) | notes |
|--------------------------------|--------|---------|-------|
| Substrate prototype-recall (1/class)  |    -   | 22.7  | bad — wrong API for the task |
| Substrate prototype-recall (5/class)  |    -   | 24.2  | overlap doesn't help |
| Substrate prototype-recall (10/class) |    -   | 17.1  | gets WORSE with more |
| Linear on raw pixels (sklearn)         | 7.9 k  | 89.2  | lower-bound baseline |
| **Substrate features (frozen) + linear**| ~24 k | **91.2** | **+2 pts vs raw linear** |
| MLP hidden=16                          | 12.7 k | 92.6  | beats us by 1.4 pts |
| MLP hidden=32                          | 25.4 k | 94.1  | beats us by 2.9 pts |
| MLP hidden=128                         |  102 k | 96.1  | beats us by 4.9 pts |

**Honest read**: we are not even in the same league as MLP-128, but
the substrate IS doing useful work as a frozen featurizer (+2 pts
over raw-pixel linear). Best discovered configuration: 4 snapshot
steps (1, 3, 6, 10) + 500-image Hebb pretraining.

## Class-incremental MNIST (catastrophic forgetting test)

Train on (0,1) → (2,3) → (4,5) → (6,7) → (8,9), evaluate on
all-classes-seen-so-far. No replay buffer, no EWC, no oracle.

| method                | task1 | task2 | task3 | task4 | task5 (10cls) |
|-----------------------|-------|-------|-------|-------|---------------|
| Substrate prototypes  | 0.783 | 0.417 | 0.277 | 0.150 | 0.153 |
| Substrate features    | 0.995 | 0.481 | 0.312 | 0.236 | 0.181 |
| MLP-naive (raw)       | 1.000 | 0.453 | 0.307 | 0.243 | 0.200 |
| **MLP + replay**      | 1.000 | 0.870 | 0.887 | 0.867 | **0.873** |

**Honest read**: substrate-features ≈ MLP-naive (both forget
catastrophically). MLP+replay crushes everything. The "no
catastrophic forgetting by construction" claim does NOT hold for
the readout — we'd need a per-class readout or sparse memory at
the readout to actually stop forgetting.

## Sequence chain learning (N transitions on shared field)

| n_links | epochs | overall discrim | links 2σ-positive |
|---------|--------|-----------------|-------------------|
| 1 (A→B) |   30   |  +0.0296        |   1/1 ✓           |
| 3       |   30   |  +0.0667        |   0/3             |
| 5       |  150   |  +0.0040        |   0/5             |
| 10      |  200   |  +0.0250        |   0/10            |

**Honest read**: substrate learns ONE A→B transition cleanly,
fails to chain. Some links spike high (+0.20) but with high
variance — no statistically reliable link learning beyond n=1.
This is a real architectural limit; sequence chaining needs
something the current STDP rule does not provide (likely
context-dependent gating or longer eligibility traces).

## Multi-task interference (same field, both tasks)

(Best decay tuning: write_lr=0.07, write_steps=15, write_decay=0.005)

| ordering             | pat keep | seq keep | both ≥90 % ? |
|----------------------|----------|----------|--------------|
| **multi_pat_then_seq** | **99 %** | **94 %** | ✅           |
| multi_seq_then_pat   | 112 %    | 72 %     | ❌           |
| interleaved          | 108 %    | 77 %     | ❌           |

**Honest read**: kızıl elma criterion met for the natural ordering
(load patterns first, learn sequences on top). Reverse ordering
hits a fundamental trade-off: pattern Hebb floods C → washes B-bias.
Tried 5 intervention families (pure-STDP, B-aware Hebb, B-protect,
write_decay sweep, write_lr sweep) — only the write_decay tuning
worked, and only for the pat→seq ordering.

## Compute / energy (THEORETICAL — not silicon-measured)

Energy reference: 32-bit fp MAC @ 28nm ≈ 1 pJ; int8 MAC ≈ 0.2 pJ.

| grid    | hz      | macs/step | est. mW (int8) |
|---------|---------|-----------|----------------|
| 14×14   |   100   |     1,860 | 0.0004 mW      |
| 28×28   |   100   |     7,560 | 0.0015 mW      |
| 56×56   |   100   |    30,360 | 0.0061 mW      |
| 168×168 |   100   |   338,016 | 0.0676 mW      |
| 168×168 |  1000   |   338,016 | 0.676 mW       |

**Honest read**: yes, the substrate is in the mW regime in theory,
but a tiny dense MLP at the same update rate is ALSO in the mW
regime. Without a silicon implementation and a side-by-side
energy/accuracy Pareto comparison, the "low-power frontier"
claim is not earned.

## Verdict — "can we beat the rivals?"

| metric                        | result |
|-------------------------------|--------|
| MNIST accuracy vs MLP         | LOSE (-5 pts vs MLP-128) |
| Continual MNIST vs MLP+replay | LOSE (-69 pts) |
| Sequence chaining             | LOSE (only n=1 works) |
| Multi-task pat→seq            | TIE / WIN (≥90 % both) |
| Multi-task seq→pat / interleaved | LOSE |
| Compute (mW)                  | TIE in theory |
| **Overall**                   | **NOT YET A RIVAL** |

The architecture has one clear specialty: shared substrate that
holds pattern memory and sequence STDP simultaneously in one
specific ordering. Everything else lags conventional baselines.
"Frontier non-LLM rival" requires either: (a) a substantial
architectural change (true context-gated plasticity for chains;
sparse-readout for continual; convolutional structure for image
classification), or (b) a niche where these losses don't matter
(very low-budget edge inference where 91 % MNIST is "good enough"
because retraining a 100k-MLP isn't an option).
