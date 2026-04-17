"""Tests for SparseCrossbar (non-local long-range connections)."""
from __future__ import annotations

import numpy as np

from el.thermofield import Field, FieldConfig
from el.thermofield.crossbar import SparseCrossbar, crossbar_for_field


def test_crossbar_construction() -> None:
    cb = SparseCrossbar(n_cells=64, k=8, seed=0)
    assert cb.n_edges() == 64 * 8
    # No self-loops
    assert not np.any(cb.src == cb.dst)
    # Conductances initialised in expected band
    assert cb.C.min() >= 0.30 - 1e-6
    assert cb.C.max() <= 0.60 + 1e-6
    # Bias starts at zero
    assert np.all(cb.B == 0.0)


def test_crossbar_conserves_energy_when_unclipped() -> None:
    """A symmetric exchange (B=0) should conserve total energy when no
    cell hits the [0,1] clamp."""
    rng = np.random.default_rng(0)
    T = rng.uniform(0.20, 0.40, size=64).astype(np.float32)  # well inside (0,1)
    cb = SparseCrossbar(n_cells=64, k=4, seed=1, flux_rate=0.05)

    e0 = float(T.sum())
    cb.step(T)
    e1 = float(T.sum())
    assert abs(e1 - e0) < 1e-3, f"Energy drift: {e0} -> {e1}"
    # And it should still be inside the legal range
    assert T.min() >= 0.0 and T.max() <= 1.0


def test_crossbar_breaks_distance_barrier() -> None:
    """A hand-placed crossbar edge between two diagonally-opposite cells
    on a 16x16 grid should let heat reach the far corner FAR sooner
    than pure local diffusion can."""
    cfg = FieldConfig(rows=16, cols=16)

    src_rc = (0, 0)
    dst_rc = (15, 15)
    src_flat = src_rc[0] * cfg.cols + src_rc[1]
    dst_flat = dst_rc[0] * cfg.cols + dst_rc[1]

    def run(use_crossbar: bool, steps: int = 8) -> float:
        field = Field(cfg, seed=0)
        field.reset_temp()
        field.inject([src_rc], [1.0])
        cb = SparseCrossbar(n_cells=cfg.rows * cfg.cols, k=1, seed=42,
                            flux_rate=0.20)
        # Force a single explicit edge from src to dst with very high C
        cb.src = np.array([src_flat], dtype=np.int64)
        cb.dst = np.array([dst_flat], dtype=np.int64)
        cb.C = np.array([1.0], dtype=np.float32)
        cb.B = np.array([0.0], dtype=np.float32)
        for _ in range(steps):
            field.step()
            if use_crossbar:
                T_flat = field.T.reshape(-1)
                cb.step(T_flat)
                field.T = T_flat.reshape(cfg.rows, cfg.cols)
        return float(field.T[dst_rc])

    no_cb = run(use_crossbar=False)
    with_cb = run(use_crossbar=True)

    # Without crossbar, far corner is essentially cold after 8 steps
    # (diffusion reach is ~sqrt(8) ~ 2.8 cells, distance is 21 cells)
    assert no_cb < 0.05, f"Pure diffusion already reaches far corner: {no_cb}"
    # With one crossbar edge, heat arrives in real amounts
    assert with_cb > no_cb * 5.0, (
        f"Crossbar did not amplify long-range transport: "
        f"no_cb={no_cb:.4f} with_cb={with_cb:.4f}"
    )


def test_crossbar_directional_bias() -> None:
    """B > 0 makes src->dst (forward) flow easier than reverse: when src
    is the hot side, dst should warm faster than the symmetric case."""
    rng_init = lambda b_val: SparseCrossbar(n_cells=4, k=1, seed=0, flux_rate=0.20)

    def run(b_val: float) -> float:
        T = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cb = rng_init(b_val)
        cb.src = np.array([0], dtype=np.int64)
        cb.dst = np.array([3], dtype=np.int64)
        cb.C = np.array([0.5], dtype=np.float32)
        cb.B = np.array([b_val], dtype=np.float32)
        for _ in range(3):
            cb.step(T)
        return float(T[3])

    sym = run(0.0)
    fwd = run(0.4)
    rev = run(-0.4)
    assert fwd > sym > rev, (
        f"Bias asymmetry not observed: fwd={fwd:.4f} sym={sym:.4f} rev={rev:.4f}"
    )


def test_helper_builds_for_field() -> None:
    field = Field(FieldConfig(rows=8, cols=8), seed=0)
    cb = crossbar_for_field(field, k=4, seed=0)
    assert cb.n_cells == 64
    assert cb.n_edges() == 64 * 4


# ---------------------------------------------------------------------------
# Dynamic-array add_edge scaling
# ---------------------------------------------------------------------------
def test_add_edge_is_amortised_constant_time():
    """add_edge must be amortised O(1). 5000 sequential adds must take
    well under 1 second on commodity hardware. The previous np.append-
    based implementation was O(n) per call → ~25M ops → would blow this."""
    import time
    from el.thermofield.crossbar import SparseCrossbar
    cb = SparseCrossbar(n_cells=10, k=2, seed=0)
    n_before = cb.n_edges()
    t0 = time.time()
    for i in range(5000):
        cb.add_edge(src=i % 10, dst=(i + 1) % 10, c=0.4, b=0.0)
    dt = time.time() - t0
    assert cb.n_edges() == n_before + 5000
    assert dt < 1.0, f"5000 add_edge calls took {dt:.3f}s — likely O(n) regression"


def test_add_edges_bulk_matches_sequential():
    """Bulk `add_edges` must produce the same edge set as repeated
    `add_edge` calls (modulo order, but order is preserved here)."""
    import numpy as np
    from el.thermofield.crossbar import SparseCrossbar
    cb1 = SparseCrossbar(n_cells=8, k=1, seed=0)
    cb2 = SparseCrossbar(n_cells=8, k=1, seed=0)
    src = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    dst = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)
    c   = np.full(8, 0.42, dtype=np.float32)
    b   = np.full(8, 0.13, dtype=np.float32)
    cb2.add_edges(src, dst, c, b)
    for s, d, cv, bv in zip(src, dst, c, b):
        cb1.add_edge(int(s), int(d), float(cv), float(bv))
    assert np.array_equal(cb1.src, cb2.src)
    assert np.array_equal(cb1.dst, cb2.dst)
    assert np.allclose(cb1.C, cb2.C)
    assert np.allclose(cb1.B, cb2.B)


def test_step_unchanged_after_grow():
    """Edges added via the new dynamic-array path must participate in
    `step()` exactly like initial edges."""
    import numpy as np
    from el.thermofield.crossbar import SparseCrossbar
    cb = SparseCrossbar(n_cells=4, k=1, seed=0, flux_rate=0.5)
    # Force a grow by appending more edges than initial capacity
    for i in range(50):
        cb.add_edge(src=0, dst=3, c=0.8, b=0.0)
    T = np.zeros(4, dtype=np.float32)
    T[0] = 1.0
    cb.step(T)
    # With many strong edges 0->3, heat should flow from 0 to 3
    assert T[3] > 0.05, f"expected heat at dst=3 after grow+step, got {T[3]}"


def test_set_C_with_wrong_length_raises():
    """C must match current edge count; arbitrary-length set is a bug."""
    import numpy as np
    import pytest
    from el.thermofield.crossbar import SparseCrossbar
    cb = SparseCrossbar(n_cells=4, k=2, seed=0)
    with pytest.raises(ValueError):
        cb.C = np.array([0.1, 0.2], dtype=np.float32)


def test_set_B_with_wrong_length_raises():
    import numpy as np
    import pytest
    from el.thermofield.crossbar import SparseCrossbar
    cb = SparseCrossbar(n_cells=4, k=2, seed=0)
    with pytest.raises(ValueError):
        cb.B = np.array([0.1, 0.2], dtype=np.float32)


def test_setting_src_then_dst_with_different_lengths_stays_consistent():
    """Architect-flagged hidden bug: previously, setting `src` then `dst`
    with different lengths left buffers of mismatched length and crashed
    `step()`. The setters must keep all four edge arrays the same length
    after every assignment.
    """
    import numpy as np
    from el.thermofield.crossbar import SparseCrossbar
    cb = SparseCrossbar(n_cells=4, k=2, seed=0, flux_rate=0.5)
    cb.src = np.array([0], dtype=np.int64)
    assert cb.src.shape[0] == cb.dst.shape[0] == cb.C.shape[0] == cb.B.shape[0]
    cb.dst = np.array([1, 2], dtype=np.int64)
    assert cb.src.shape[0] == cb.dst.shape[0] == cb.C.shape[0] == cb.B.shape[0]
    # And step() must not crash regardless of the order of assignments.
    T = np.zeros(4, dtype=np.float32)
    T[0] = 1.0
    cb.step(T)   # must not raise


def test_replace_edges_atomic_path():
    """Preferred length-changing path: replace_edges() with strict
    length checking on all arrays."""
    import numpy as np
    import pytest
    from el.thermofield.crossbar import SparseCrossbar
    cb = SparseCrossbar(n_cells=4, k=2, seed=0)
    cb.replace_edges(
        src=np.array([0, 1, 2], dtype=np.int64),
        dst=np.array([3, 0, 1], dtype=np.int64),
        C=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        B=np.array([0.0, 0.1, -0.1], dtype=np.float32),
    )
    assert cb.n_edges() == 3
    assert np.array_equal(cb.src, [0, 1, 2])
    assert np.allclose(cb.C, [0.4, 0.5, 0.6])
    # Mismatched lengths must raise
    with pytest.raises(ValueError):
        cb.replace_edges(
            src=np.array([0, 1], dtype=np.int64),
            dst=np.array([2], dtype=np.int64),
        )
