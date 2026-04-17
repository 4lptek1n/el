"""FrozenSubstrate must be deterministic and read-only."""
from __future__ import annotations
import os, tempfile, numpy as np, pytest

from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory, random_pattern, corrupt
from el.thermofield.frozen import FrozenSubstrate


GRID = 24


def _seeded_pm(seed=0, n=8):
    pm = PatternMemory(cfg=FieldConfig(rows=GRID, cols=GRID), seed=seed)
    rng = np.random.default_rng(seed)
    pats = [random_pattern(GRID, GRID, 16, rng) for _ in range(n)]
    for p in pats:
        pm.store(p)
    return pm, pats


def test_freeze_does_not_modify_substrate():
    pm, pats = _seeded_pm()
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=64, seed=1)
    fp1 = fr.fingerprint()
    rng = np.random.default_rng(99)
    for _ in range(50):
        cue = corrupt(pats[rng.integers(len(pats))], 0.5, rng)
        fr.encode(cue)
    fp2 = fr.fingerprint()
    assert fp1 == fp2, "frozen substrate weights changed during encode"


def test_encode_is_deterministic():
    pm, pats = _seeded_pm()
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=64, seed=1)
    cue = pats[0]
    a = fr.encode(cue)
    b = fr.encode(cue)
    np.testing.assert_array_equal(a, b)


def test_encode_shape_and_dtype():
    pm, _ = _seeded_pm()
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=37, seed=0)
    out = fr.encode([(0, 0), (1, 1)])
    assert out.shape == (37,)
    assert out.dtype == np.float32


def test_encode_batch_matches_encode():
    pm, pats = _seeded_pm()
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=32, seed=2)
    one = np.stack([fr.encode(p) for p in pats[:5]])
    batch = fr.encode_batch(pats[:5])
    np.testing.assert_array_equal(one, batch)


def test_save_load_roundtrip():
    pm, pats = _seeded_pm()
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=48, seed=3)
    cue = pats[0]
    a = fr.encode(cue)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fr.npz")
        fr.save(path)
        fr2 = FrozenSubstrate.load(path)
    assert fr.fingerprint() == fr2.fingerprint()
    np.testing.assert_array_equal(a, fr2.encode(cue))


def test_readonly_arrays():
    pm, _ = _seeded_pm()
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=16, seed=0)
    with pytest.raises(ValueError):
        fr.C_right[0, 0] = 99.0
    with pytest.raises(ValueError):
        fr.read_positions[0, 0] = 99


def test_different_inputs_give_different_features():
    pm, pats = _seeded_pm()
    fr = FrozenSubstrate.from_pattern_memory(pm, n_readout=128, seed=4)
    f1 = fr.encode(pats[0])
    f2 = fr.encode(pats[1])
    assert not np.allclose(f1, f2), "frozen encoder collapsed two patterns"
