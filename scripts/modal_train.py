"""Modal H200 training entrypoint for the el Action Transformer.

Run (after `modal setup`):

    modal run scripts/modal_train.py::train --preset h200 --steps 200000

This script is intentionally self-contained. It does NOT import from the
`el` package at decorator-evaluation time so that a machine without the
package installed can still register the Modal function. The training body
runs inside the Modal container where `el` is pip-installed from the
committed working tree.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _maybe_import_modal():
    try:
        import modal

        return modal
    except Exception:
        return None


modal = _maybe_import_modal()


def train(preset: str = "h200", steps: int = 200_000, batch: int = 32, lr: float = 1e-4) -> None:
    subprocess.check_call(["pip", "install", "-e", ".[train]"], cwd=str(Path(__file__).resolve().parent.parent))
    from el.transformer.train import main as train_main  # noqa: E402

    import sys

    sys.argv = [
        "train",
        "--preset", preset,
        "--steps", str(steps),
        "--batch", str(batch),
        "--lr", str(lr),
        "--out", "/root/el/checkpoints/el-action-transformer",
    ]
    raise SystemExit(train_main())


if modal is not None:
    image = modal.Image.debian_slim().pip_install("torch>=2.0", "click>=8.0")
    app = modal.App("el-action-transformer")

    @app.function(image=image, gpu="H200", timeout=60 * 60 * 6)
    def _modal_train(preset: str = "h200", steps: int = 200_000, batch: int = 32, lr: float = 1e-4) -> None:
        train(preset=preset, steps=steps, batch=batch, lr=lr)
