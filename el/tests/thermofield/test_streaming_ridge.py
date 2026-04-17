"""StreamingRidge: closed-form ridge classifier with streaming sufficient stats."""
from __future__ import annotations
import os, tempfile, numpy as np

from el.thermofield.readout import StreamingRidge


def _toy(n=400, d=12, k=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2, size=(k, d))
    y = rng.integers(0, k, size=n)
    X = centers[y] + rng.normal(0, 0.5, size=(n, d))
    return X.astype(np.float32), y


def test_fits_linearly_separable():
    X, y = _toy(seed=0)
    r = StreamingRidge(n_features=12, n_classes=3, ridge_lambda=0.1)
    r.partial_fit(X, y)
    pred = r.predict(X)
    acc = float((pred == y).mean())
    assert acc > 0.9, f"toy acc {acc} too low"


def test_streaming_equals_batch():
    X, y = _toy(n=600, seed=1)
    rb = StreamingRidge(n_features=12, n_classes=3, ridge_lambda=0.5)
    rb.partial_fit(X, y); rb.solve()

    rs = StreamingRidge(n_features=12, n_classes=3, ridge_lambda=0.5)
    for s in range(0, 600, 50):
        rs.partial_fit(X[s:s + 50], y[s:s + 50])
    rs.solve()

    np.testing.assert_allclose(rb._W, rs._W, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(rb.predict(X), rs.predict(X))


def test_save_load_roundtrip():
    X, y = _toy(seed=2)
    r = StreamingRidge(n_features=12, n_classes=3, ridge_lambda=0.1)
    r.partial_fit(X, y); r.solve()
    pred = r.predict(X)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.npz")
        r.save(p)
        r2 = StreamingRidge.load(p)
    np.testing.assert_array_equal(r2.predict(X), pred)
    assert r2.n_seen == r.n_seen


def test_memory_independent_of_n():
    """Sufficient statistics size must NOT grow with N — that is the
    whole point of streaming. Verify by attribute shapes."""
    r = StreamingRidge(n_features=64, n_classes=10, ridge_lambda=1.0)
    rng = np.random.default_rng(7)
    for _ in range(20):
        X = rng.normal(size=(100, 64)).astype(np.float32)
        y = rng.integers(0, 10, size=100)
        r.partial_fit(X, y)
    assert r._A.shape == (64, 64)
    assert r._B.shape == (64, 10)
    assert r.n_seen == 2000


def test_handles_zero_features_gracefully():
    r = StreamingRidge(n_features=4, n_classes=2, ridge_lambda=1.0)
    r.partial_fit(np.zeros((10, 4), dtype=np.float32),
                  np.array([0] * 5 + [1] * 5))
    pred = r.predict(np.zeros((4, 4), dtype=np.float32))
    assert pred.shape == (4,)
