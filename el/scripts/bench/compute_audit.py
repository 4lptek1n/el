"""Theoretical compute / energy audit for the substrate.
Counts multiply-adds per Field.step() and per pattern store/recall;
projects mW assuming various silicon energy/op figures.
"""
import sys, numpy as np
sys.path.insert(0, "src")
from el.thermofield import FieldConfig, Field
from el.thermofield.pattern_memory import PatternMemory

# Energy figures (well-known approximations from H&P / EE references)
E_MAC_28NM_PJ = 1.0       # 32-bit fp MAC @ 28nm: ~1 pJ
E_MAC_INT8_PJ = 0.2       # int8 MAC: ~0.2 pJ
E_SRAM_8B_PJ = 0.5        # 8-byte SRAM read

def count_step_ops(field):
    """Field.step does diffusion + nonlinear + decay. All elementwise.
    Per cell: 1 diffusion (~5 MACs incl. 4 neighbors), 1 nonlinear,
    1 decay. Edges: B*C used, ~2 multiplies per edge.
    """
    R, C = field.cfg.rows, field.cfg.cols
    n_cells = R * C
    n_edges_h = R * (C - 1)
    n_edges_v = (R - 1) * C
    # Diffusion: each cell averages with its 4 neighbors → ~5 MACs
    diffusion_macs = n_cells * 5
    # Edge-weighted flow: each edge contributes 2 MACs (C+B, multiply T)
    edge_macs = (n_edges_h + n_edges_v) * 2
    # Nonlinear (1 + α·avg) + decay: ~3 MACs/cell
    other_macs = n_cells * 3
    return diffusion_macs + edge_macs + other_macs

def count_hebb_ops(field):
    """hebbian_update: per edge → 1 multiply (co), 1 add, 1 decay
    Per edge: ~3 MACs. Plus clip = trivial.
    """
    R, C = field.cfg.rows, field.cfg.cols
    n_edges = R*(C-1) + (R-1)*C
    return n_edges * 3

def project(grid, hz):
    cfg = FieldConfig(rows=grid, cols=grid)
    f = Field(cfg, seed=0)
    macs_step = count_step_ops(f)
    macs_hebb = count_hebb_ops(f)
    print(f"\nGrid {grid}x{grid}, update {hz} Hz:")
    print(f"  Field.step():     {macs_step:>10,d} MACs")
    print(f"  hebbian_update(): {macs_hebb:>10,d} MACs")
    # Pattern store: write_steps=15 → 15 step + 15 hebb
    pat_store = 15 * (macs_step + macs_hebb)
    pat_recall = 8 * macs_step + 10 * 50  # +scoring
    print(f"  pattern store:    {pat_store:>10,d} MACs")
    print(f"  pattern recall:   {pat_recall:>10,d} MACs")
    # Power = energy/op * ops/sec
    ops_per_sec = (macs_step + macs_hebb) * hz   # assume 1 hebb per step
    pj_per_sec_28nm_fp = ops_per_sec * E_MAC_28NM_PJ
    pj_per_sec_int8 = ops_per_sec * E_MAC_INT8_PJ
    mw_28nm_fp = pj_per_sec_28nm_fp / 1e9
    mw_int8 = pj_per_sec_int8 / 1e9
    print(f"  Power @ {hz}Hz update (theoretical):")
    print(f"    32-bit fp 28nm: {mw_28nm_fp:8.3f} mW")
    print(f"    int8           : {mw_int8:8.3f} mW")
    print(f"  Equivalent NN (1 layer 64x64 dense MLP @ {hz}Hz): "
          f"{64*64*hz*E_MAC_28NM_PJ/1e9:.3f} mW (32-bit fp ref)")

def main():
    print("=== Compute / energy audit (THEORETICAL — not silicon-measured) ===")
    print("Energy figures: 32-bit fp MAC @ 28nm ≈ 1 pJ; int8 MAC ≈ 0.2 pJ")
    for grid in [14, 28, 56, 168]:
        for hz in [10, 100, 1000]:
            project(grid, hz)

if __name__ == "__main__":
    main()
