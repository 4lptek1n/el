"""Train and evaluate the thermodynamic field on simple tasks.

The training loop is deliberately minimal: present input, relax field,
read output, apply local plasticity rules. Repeat. There is no global
loss, no backprop, no optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .field import Field, FieldConfig
from .plasticity import hebbian_update, supervised_nudge


def make_or_dataset() -> list[tuple[list[float], float]]:
    return [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 1.0),
    ]


def make_xor_dataset() -> list[tuple[list[float], float]]:
    return [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]


@dataclass
class TrainResult:
    field: Field
    history: list[float]
    final_error: float
    accuracy: float


def _default_io(cfg: FieldConfig) -> tuple[list, list]:
    in_positions = [(0, cfg.cols // 4), (0, 3 * cfg.cols // 4)]
    out_positions = [(cfg.rows - 1, cfg.cols // 2)]
    return in_positions, out_positions


def train(
    dataset_fn: Callable[[], list[tuple[list[float], float]]],
    *,
    epochs: int = 300,
    seed: int = 0,
    cfg: FieldConfig | None = None,
    hebb_lr: float = 0.004,
    nudge_lr: float = 0.06,
    verbose: bool = False,
) -> TrainResult:
    cfg = cfg or FieldConfig()
    field = Field(cfg, seed=seed)
    in_positions, out_positions = _default_io(cfg)

    dataset = dataset_fn()
    rng = np.random.default_rng(seed + 1)
    history: list[float] = []

    for epoch in range(epochs):
        order = list(dataset)
        rng.shuffle(order)
        epoch_err = 0.0
        for inputs, target in order:
            field.reset_temp()
            field.inject(in_positions, inputs)
            field.relax()
            err = supervised_nudge(field, out_positions, [target], lr=nudge_lr)
            hebbian_update(field, lr=hebb_lr)
            epoch_err += err
        epoch_err /= len(order)
        history.append(epoch_err)
        if verbose and epoch % 25 == 0:
            print(f"epoch {epoch:4d}  err={epoch_err:.4f}")

    results = evaluate(field, dataset_fn, in_positions, out_positions)
    correct = sum(1 for r in results if (r["output"] >= 0.5) == (r["target"] >= 0.5))
    accuracy = correct / len(results)
    final_err = sum(abs(r["target"] - r["output"]) for r in results) / len(results)
    return TrainResult(field=field, history=history, final_error=final_err, accuracy=accuracy)


def train_or(**kwargs) -> TrainResult:
    return train(make_or_dataset, **kwargs)


def train_xor(**kwargs) -> TrainResult:
    return train(make_xor_dataset, **kwargs)


def evaluate(
    field: Field,
    dataset_fn: Callable[[], list[tuple[list[float], float]]],
    in_positions=None,
    out_positions=None,
) -> list[dict]:
    if in_positions is None or out_positions is None:
        in_positions, out_positions = _default_io(field.cfg)
    out = []
    for inputs, target in dataset_fn():
        field.reset_temp()
        field.inject(in_positions, inputs)
        field.relax()
        reading = float(field.read(out_positions)[0])
        out.append({"input": list(inputs), "target": float(target), "output": reading})
    return out
