"""Seed the skill registry from bundled corpora.

Usage:
    python scripts/seed_registry.py [--state-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from el.config import load_config  # noqa: E402
from el.registry import SkillRegistry  # noqa: E402
from el.seed.bootstrap import seed_registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", type=str, default=None)
    args = ap.parse_args()
    config = load_config(state_dir=Path(args.state_dir)) if args.state_dir else load_config()
    config.ensure_dirs()
    registry = SkillRegistry(config.registry_path)
    stats = seed_registry(registry, use_bundled=True)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
