"""Properties of the HDC primitives.

These properties are non-negotiable: bind self-inverse, bundle
preserves similarity to ingredients, random vectors are quasi-orthogonal.
"""
from __future__ import annotations

import numpy as np

from el.worldmodel.hdc import (
    DEFAULT_DIM,
    HDC,
    bind,
    bundle,
    cosine_sim,
    permute,
    random_hv,
)


def test_random_hv_is_bipolar_and_correct_dim():
    v = random_hv(name="alpha")
    assert v.shape == (DEFAULT_DIM,)
    assert v.dtype == np.int8
    uniq = set(np.unique(v).tolist())
    assert uniq.issubset({-1, 1})


def test_random_hv_seeded_is_deterministic():
    a = random_hv(name="same")
    b = random_hv(name="same")
    assert np.array_equal(a, b)


def test_random_hvs_are_quasi_orthogonal():
    a = random_hv(name="alpha")
    b = random_hv(name="beta")
    # For D=10000 random bipolar, |cos| << 1
    assert abs(cosine_sim(a, b)) < 0.05


def test_bind_is_self_inverse():
    a = random_hv(name="key")
    b = random_hv(name="value")
    bound = bind(a, b)
    recovered = bind(bound, b)
    assert np.array_equal(recovered, a)


def test_bundle_preserves_similarity_to_ingredients():
    a = random_hv(name="x")
    b = random_hv(name="y")
    c = random_hv(name="z")
    s = bundle([a, b, c])
    # Each ingredient should remain notably similar to the bundle.
    assert cosine_sim(s, a) > 0.3
    assert cosine_sim(s, b) > 0.3
    assert cosine_sim(s, c) > 0.3


def test_permute_changes_vector():
    a = random_hv(name="seq")
    p1 = permute(a, k=1)
    assert not np.array_equal(p1, a)
    # Inverse rotation recovers
    back = permute(p1, k=-1)
    assert np.array_equal(back, a)


def test_hdc_atom_interns():
    h = HDC()
    a1 = h.atom("foo")
    a2 = h.atom("foo")
    assert a1 is a2  # cached


def test_encode_intent_atoms_compositional():
    h = HDC()
    list_dir = h.encode_intent_atoms("list", "file", "this_folder")
    list_other = h.encode_intent_atoms("list", "file", "downloads")
    delete_dir = h.encode_intent_atoms("delete", "file", "this_folder")
    different = h.encode_intent_atoms("git_status", "", "")
    # Same verb+obj, different scope: still moderately similar
    assert cosine_sim(list_dir, list_other) > 0.3
    # Same scope+obj, different verb: still moderately similar
    assert cosine_sim(list_dir, delete_dir) > 0.3
    # Totally different: near orthogonal
    assert abs(cosine_sim(list_dir, different)) < 0.3
