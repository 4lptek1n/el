"""Eşik 3 — multi-task substrate cross-task interference probe.

Drives the same helpers used by `tests/thermofield/test_multitask.py`
across all four orderings and prints the honest scorecard:

  * pattern_only        → pattern recall baseline
  * seq_only            → A→B sequence discrim baseline
  * multi_pat_then_seq  → pattern first, then sequence (the kızıl-elma WIN)
  * multi_seq_then_pat  → sequence first, then pattern (KNOWN_OPEN)

The WIN is achieved with COTRAIN_PARAMS = dict(write_lr=0.07,
write_steps=15, write_decay=0.005). See replit.md "Eşik 3" section
and the regression tests for the pinned numbers.
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
from thermofield.test_multitask import _run, COTRAIN_PARAMS  # type: ignore


def main():
    print("Multi-task probe (kızıl-elma Eşik 3)")
    print("Tuning:", COTRAIN_PARAMS)
    print("-" * 64)

    pa_alone, _      = _run("pattern_only",       COTRAIN_PARAMS)
    _, sq_alone      = _run("seq_only",           COTRAIN_PARAMS)
    pa_pts, sq_pts   = _run("multi_pat_then_seq", COTRAIN_PARAMS)
    pa_stp, sq_stp   = _run("multi_seq_then_pat", COTRAIN_PARAMS)

    def fmt(arr): return f"{float(np.mean(arr)):+.4f} ± {float(np.std(arr)):.4f}"
    print(f"  pattern_only           pat_acc  = {fmt(pa_alone)}")
    print(f"  seq_only               seq_disc = {fmt(sq_alone)}")
    print(f"  multi_pat_then_seq     pat_acc  = {fmt(pa_pts)}")
    print(f"                         seq_disc = {fmt(sq_pts)}")
    print(f"  multi_seq_then_pat     pat_acc  = {fmt(pa_stp)}")
    print(f"                         seq_disc = {fmt(sq_stp)}")
    print("-" * 64)

    pa0 = float(np.mean(pa_alone)); sq0 = float(np.mean(sq_alone))
    pts_pa = float(np.mean(pa_pts)) / max(pa0, 1e-9)
    pts_sq = float(np.mean(sq_pts)) / max(sq0, 1e-9)
    stp_pa = float(np.mean(pa_stp)) / max(pa0, 1e-9)
    stp_sq = float(np.mean(sq_stp)) / max(sq0, 1e-9)
    print(f"  pat_then_seq retention:  pat {100*pts_pa:5.1f} %   "
          f"seq {100*pts_sq:5.1f} %   (kızıl-elma gate: ≥90 / ≥90)")
    print(f"  seq_then_pat retention:  pat {100*stp_pa:5.1f} %   "
          f"seq {100*stp_sq:5.1f} %   (KNOWN_OPEN: seq drops)")


if __name__ == "__main__":
    main()
