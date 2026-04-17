"""Continual / class-incremental learning regression tests.

Synthetic 5-class proxy: each class has a distinct anchor pattern;
samples are noisy versions. Class-incremental presentation
(class 0, then class 1, ...). Two readouts compared:

  * Per-class head (the new lib API) — must keep accuracy on early
    classes after later classes are introduced. Final 5-class
    accuracy gate: ≥0.70.
  * Shared softmax — known to forget catastrophically; honest baseline
    that should fall well below the per-class head.

Also a basic unit test on `feature_snapshot` shape + determinism.
"""
from __future__ import annotations
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from el.thermofield import FieldConfig, Field
from el.thermofield.continual import PerClassReadout, feature_snapshot


GRID = 12
SNAP_STEPS = (2, 5, 8)
N_CLASSES = 5
N_PER_CLASS = 25
N_TEST_PER_CLASS = 12
RNG = np.random.default_rng(123)


def _class_templates(n_classes: int = N_CLASSES, k: int = 8):
    used = set(); templates = []
    for _ in range(n_classes):
        cells = []
        while len(cells) < k:
            r = int(RNG.integers(0, GRID)); c = int(RNG.integers(0, GRID))
            if (r, c) not in used:
                used.add((r, c)); cells.append((r, c))
        templates.append(cells)
    return templates


def _noisy_sample(template, drop=0.3, n_distract=8):
    cells = [c for c in template if RNG.random() > drop]
    used = set(cells)
    while len(used) < len(cells) + n_distract:
        r = int(RNG.integers(0, GRID)); c = int(RNG.integers(0, GRID))
        used.add((r, c))
    return list(used)


def _make_split():
    templates = _class_templates()
    train = {c: [_noisy_sample(templates[c]) for _ in range(N_PER_CLASS)]
             for c in range(N_CLASSES)}
    test  = {c: [_noisy_sample(templates[c]) for _ in range(N_TEST_PER_CLASS)]
             for c in range(N_CLASSES)}
    return train, test


def _featurize_set(field, samples):
    return np.stack([feature_snapshot(field, s, SNAP_STEPS) for s in samples])


def test_feature_snapshot_shape_and_determinism():
    cfg = FieldConfig(rows=GRID, cols=GRID)
    f1 = Field(cfg, seed=0); f2 = Field(cfg, seed=0)
    pat = [(1, 2), (3, 5), (7, 8)]
    a = feature_snapshot(f1, pat, SNAP_STEPS)
    b = feature_snapshot(f2, pat, SNAP_STEPS)
    assert a.shape == (GRID * GRID * len(SNAP_STEPS),)
    assert np.allclose(a, b)
    # Empty pattern should give zeros (no injection)
    z = feature_snapshot(f1, [], SNAP_STEPS)
    assert np.allclose(z, 0.0)


def test_per_class_head_beats_shared_softmax_on_class_incremental():
    cfg = FieldConfig(rows=GRID, cols=GRID)
    field = Field(cfg, seed=0)

    train, test = _make_split()
    # Pre-featurize the WHOLE test set once (frozen substrate)
    Fte_per = {c: _featurize_set(field, test[c]) for c in range(N_CLASSES)}
    feat_dim = Fte_per[0].shape[1]

    # Per-class readout: train each class as it arrives; never touch
    # earlier classes again
    head = PerClassReadout(dim=feat_dim, n_classes=N_CLASSES)

    # Shared softmax: also class-incremental, sees only current class
    # vs negatives drawn from the same set the per-class head sees.
    torch.manual_seed(0)
    shared = nn.Linear(feat_dim, N_CLASSES)
    opt = torch.optim.Adam(shared.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for cls in range(N_CLASSES):
        F_pos = _featurize_set(field, train[cls])
        # Negatives: union of all OTHER classes' training data
        neg_samples = [s for c in range(N_CLASSES) if c != cls for s in train[c]]
        # Subsample for fair compute
        neg_idx = RNG.choice(len(neg_samples),
                             min(len(neg_samples), len(F_pos)*2),
                             replace=False)
        F_neg = _featurize_set(field, [neg_samples[i] for i in neg_idx])
        head.train_class(cls, F_pos, F_neg, epochs=80)

        # Shared softmax: trained ONLY on this class as positive (naive)
        Xb = torch.from_numpy(F_pos).float()
        yb = torch.full((len(F_pos),), cls, dtype=torch.long)
        for _ in range(8):
            opt.zero_grad()
            loss = loss_fn(shared(Xb), yb)
            loss.backward(); opt.step()

    # Eval on full test set
    Fte = np.concatenate([Fte_per[c] for c in range(N_CLASSES)], axis=0)
    yte = np.concatenate([np.full(N_TEST_PER_CLASS, c) for c in range(N_CLASSES)])

    head_pred = head.predict(Fte)
    head_acc = (head_pred == yte).mean()

    with torch.no_grad():
        shared_pred = shared(torch.from_numpy(Fte).float()).argmax(1).numpy()
    shared_acc = (shared_pred == yte).mean()

    chance = 1.0 / N_CLASSES
    assert head_acc >= 0.70, (
        f"per-class head should reach ≥0.70 on 5-way class-incremental "
        f"proxy, got {head_acc:.3f}")
    assert head_acc > shared_acc + 0.20, (
        f"per-class head must clearly beat shared softmax under "
        f"catastrophic-forgetting conditions: "
        f"head={head_acc:.3f} shared={shared_acc:.3f} chance={chance:.2f}")


def test_per_class_head_predict_only_uses_trained_classes():
    cfg = FieldConfig(rows=GRID, cols=GRID)
    field = Field(cfg, seed=0)
    train, test = _make_split()
    F_pos = _featurize_set(field, train[0])
    F_neg = _featurize_set(field, train[1])
    feat_dim = F_pos.shape[1]
    head = PerClassReadout(dim=feat_dim, n_classes=N_CLASSES)
    head.train_class(0, F_pos, F_neg, epochs=40)
    # Only class 0 trained → all predictions must be class 0
    F_te = _featurize_set(field, test[0] + test[1] + test[2])
    pred = head.predict(F_te)
    assert (pred == 0).all(), (
        f"predict must only output trained classes, got {set(pred.tolist())}")
