"""Thermodynamic field with local plasticity.

A radical experiment in non-LLM intelligence: replace [structure + weights + time]
with a single [continuous temperature field + distance-weighted diffusion +
local Hebbian plasticity]. No backprop. No labels (except a local nudge at
output cells). All learning rules are local — a cell only sees its neighbors.

Inspired by:
- Reservoir computing (fixed dynamics, output readout)
- Hopfield networks (energy minimization)
- Karl Friston's free energy principle (settling to low-surprise state)
- Hebbian learning ("neurons that fire together wire together")

This is NOT claimed to beat frontier LLMs. It is a working prototype of an
unexplored architectural combination, useful for measurement and intuition.
"""
from .field import Field, FieldConfig
from .interneurons import Interneurons, InterneuronConfig
from .plasticity import (
    gated_hebbian_update,
    hebbian_update,
    lateral_inhibition_update,
    supervised_nudge,
)
from .runner import (
    XORWithInterneuronResult,
    evaluate,
    make_or_dataset,
    make_xor_dataset,
    train_or,
    train_xor,
    train_xor_with_interneurons,
)

__all__ = [
    "Field",
    "FieldConfig",
    "Interneurons",
    "InterneuronConfig",
    "hebbian_update",
    "gated_hebbian_update",
    "lateral_inhibition_update",
    "supervised_nudge",
    "train_xor",
    "train_or",
    "evaluate",
    "make_xor_dataset",
    "make_or_dataset",
    "train_xor_with_interneurons",
    "XORWithInterneuronResult",
]
