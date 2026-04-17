"""Substrate-as-feature-extractor regression test (synthetic proxy).

10 class × distinct anchor patterns. Hard noise: each sample drops
40 % of class anchors and adds 12-18 random distractor pixels.
Substrate = frozen Field, snapshot T at 3 relax steps, concat as
feature vector. Linear readout (Adam + L2). Compared to a linear
readout on raw pixels. Gates:

  * substrate features beat chance by a wide margin (>40 pts)
  * substrate features at least match raw-pixel linear (within 3 pts)
  * deterministic for fixed seed
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn as nn
from el.thermofield import FieldConfig, Field

GRID = 14
SNAP_STEPS = (2, 5, 8)
RNG = np.random.default_rng(42)


def _make_class_templates(n_classes=10, k=8):
    templates = []; used = set()
    for _ in range(n_classes):
        cells = []
        while len(cells) < k:
            r = int(RNG.integers(0, GRID)); cl = int(RNG.integers(0, GRID))
            if (r, cl) not in used:
                used.add((r, cl)); cells.append((r, cl))
        templates.append(cells)
    return templates


def _make_dataset(templates, n_per_class=10, drop=0.4, n_distractors=15):
    X = np.zeros((len(templates) * n_per_class, GRID*GRID), dtype=np.float32)
    y = np.zeros(len(templates) * n_per_class, dtype=np.int64)
    for ci, t in enumerate(templates):
        for j in range(n_per_class):
            img = np.zeros(GRID*GRID, dtype=np.float32)
            kept = [c for c in t if RNG.random() > drop]
            for (r, cl) in kept:
                img[r*GRID + cl] = 1.0
            for _ in range(int(RNG.integers(n_distractors-3, n_distractors+3))):
                idx = int(RNG.integers(0, GRID*GRID))
                img[idx] = 1.0
            X[ci*n_per_class + j] = img
            y[ci*n_per_class + j] = ci
    return X, y


def _featurize(field, img):
    field.reset_temp()
    flat = img.reshape(GRID, GRID) > 0.5
    rs, cs = np.where(flat)
    pos = list(zip([int(r) for r in rs], [int(c) for c in cs]))
    if not pos:
        return np.zeros(GRID*GRID*len(SNAP_STEPS), dtype=np.float32)
    field.inject(pos, [1.0]*len(pos), clamp=False)
    out = []; mx = max(SNAP_STEPS)
    for t in range(mx + 1):
        if t in SNAP_STEPS: out.append(field.T.flatten().copy())
        if t < mx: field.step()
    return np.concatenate(out)


def _train_linear(X, y, dim, n_classes=10, epochs=120):
    torch.manual_seed(0)
    net = nn.Linear(dim, n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(Xt), yt)
        loss.backward(); opt.step()
    return net


def test_substrate_features_useful_for_classification():
    templates = _make_class_templates()
    X_tr, y_tr = _make_dataset(templates, n_per_class=15, drop=0.4)
    X_te, y_te = _make_dataset(templates, n_per_class=10, drop=0.4)

    raw_net = _train_linear(X_tr, y_tr, dim=GRID*GRID)
    with torch.no_grad():
        raw_acc = (raw_net(torch.from_numpy(X_te).float()).argmax(1).numpy() == y_te).mean()

    cfg = FieldConfig(rows=GRID, cols=GRID)
    field = Field(cfg, seed=0)
    Ftr = np.stack([_featurize(field, X_tr[i]) for i in range(len(X_tr))])
    Fte = np.stack([_featurize(field, X_te[i]) for i in range(len(X_te))])
    sub_net = _train_linear(Ftr, y_tr, dim=Ftr.shape[1])
    with torch.no_grad():
        sub_acc = (sub_net(torch.from_numpy(Fte).float()).argmax(1).numpy() == y_te).mean()

    chance = 0.10
    assert sub_acc > chance + 0.40, (
        f"substrate features barely above chance: sub={sub_acc:.3f}")
    # Honest gate: substrate must be at least competitive with raw
    # (no worse than 3 pts behind). It needn't dominate — featurization
    # is a frozen recurrent map, not a learned representation.
    assert sub_acc >= raw_acc - 0.03, (
        f"substrate features hurt vs raw-pixel linear: "
        f"sub={sub_acc:.3f} raw={raw_acc:.3f}")


def test_field_relaxation_is_deterministic_for_fixed_seed():
    cfg = FieldConfig(rows=GRID, cols=GRID)
    img = np.zeros(GRID*GRID, dtype=np.float32)
    img[[10, 50, 100]] = 1.0
    f1 = Field(cfg, seed=0); f2 = Field(cfg, seed=0)
    a = _featurize(f1, img); b = _featurize(f2, img)
    assert np.allclose(a, b)
