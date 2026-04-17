"""Subprocess entrypoint for running one self-play candidate in isolation.

Invoked as `python -m el._candidate_runner` with the candidate plan on stdin
as JSON:

    {
      "actions":  [{"name": ..., "kwargs": [[k,v], ...]}, ...],
      "intent":   {...},
      "workdir":  "/tmp/el-sandbox-xxxx",
      "offline":  true,
      "timeout":  20.0
    }

The runner sets the working directory, optionally forces offline mode, and
writes a JSON result on stdout:

    {
      "reward":   float,
      "ok":       bool,
      "results":  [...],
    }

Per-action execution errors are captured. The parent enforces a wall-clock
timeout via `subprocess.run(..., timeout=...)`, so a hung candidate is
killed without affecting others.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    payload = json.loads(sys.stdin.read())
    workdir = payload.get("workdir")
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)
    from .intent import Intent
    from .primitives import Action, is_destructive, is_networked
    from .rewards import score_outcome

    intent = Intent.from_dict(payload["intent"])
    offline = bool(payload.get("offline", True))
    actions = [Action.from_dict(a) for a in payload.get("actions", [])]

    results = []
    all_ok = True
    for action in actions:
        if is_destructive(action):
            from .primitives import PrimResult

            r = PrimResult(name=action.name, ok=False, duration_ms=0, error="destructive action blocked in candidate sandbox")
            results.append(r)
            all_ok = False
            break
        if is_networked(action) and offline:
            from .primitives import PrimResult

            r = PrimResult(name=action.name, ok=False, duration_ms=0, error="offline sandbox blocks network")
            results.append(r)
            all_ok = False
            break
        r = action.call()
        results.append(r)
        if not r.ok and action.name != "noop":
            all_ok = False
            break

    reward = score_outcome(tuple(results), intent)
    out = {
        "reward": reward,
        "ok": all_ok,
        "results": [r.to_dict() for r in results],
    }
    sys.stdout.write(json.dumps(out, default=str))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
