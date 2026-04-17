"""Tests for the review-remediation features:

- root-level `el "..."` and `--daemon` flag
- `el rate up|down`
- daemon-integrated decay
- sandboxed N-candidate self-play
- `sh` destructive payload gating
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

from el.cli import main
from el.config import load_config
from el.daemon import Daemon
from el.executor import Executor
from el.intent import Intent
from el.primitives import Action, is_destructive, is_networked, sh
from el.registry import SkillRegistry
from el.selfplay import SelfPlayRunner


def _state(tmp_path: Path) -> list[str]:
    return ["--state-dir", str(tmp_path)]


def test_root_command_without_run(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(main, [*_state(tmp_path), "list this folder"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["origin"] in {"registry", "selfplay", "transformer"}


def test_root_daemon_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(main, [*_state(tmp_path), "--daemon", "--iterations", "1", "--tick", "0.01"])
    assert res.exit_code == 0, res.output
    assert "exited after 1" in res.output


def test_rate_updates_last_skill(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, [*_state(tmp_path), "list this folder"])
    before_rate = runner.invoke(main, [*_state(tmp_path), "rate", "up"])
    assert before_rate.exit_code == 0, before_rate.output
    after_rate = runner.invoke(main, [*_state(tmp_path), "rate", "down"])
    assert after_rate.exit_code == 0, after_rate.output
    up_payload = json.loads(before_rate.output)
    down_payload = json.loads(after_rate.output)
    assert up_payload["direction"] == "up"
    assert down_payload["direction"] == "down"
    assert up_payload["skill_id"] == down_payload["skill_id"]
    assert down_payload["weight"] < up_payload["weight"]


def test_rate_without_prior_run_fails(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(main, [*_state(tmp_path), "rate", "up"])
    assert res.exit_code == 1
    assert "no previous run" in res.output


def test_daemon_calls_decay(tmp_path: Path) -> None:
    cfg = load_config(state_dir=tmp_path)
    cfg.ensure_dirs()
    registry = SkillRegistry(cfg.registry_path)
    executor = Executor(config=cfg, registry=registry, selfplay=SelfPlayRunner())
    decayed = []
    daemon = Daemon(executor=executor, tick_seconds=0.001, decay_every=1)
    daemon._on_event = lambda k, p: decayed.append(k) if k == "daemon_decay" else None
    daemon.run(max_iterations=2)
    assert "daemon_decay" in decayed


def test_sh_refuses_destructive_without_confirm() -> None:
    res = sh("rm -rf /tmp/some-nonexistent-path-xyz")
    assert res.ok is False
    assert "refusing destructive" in (res.error or "")


def test_is_destructive_detects_sh_payload() -> None:
    a = Action.make("sh", cmd="pip install fakepkg")
    assert is_destructive(a) is True
    assert is_networked(a) is True
    b = Action.make("sh", cmd="echo hi")
    assert is_destructive(b) is False


def test_selfplay_generates_n_candidates() -> None:
    runner = SelfPlayRunner(num_candidates=4)
    intent = Intent(verb="list", obj="directory", args=(("path", "."),))
    cands = runner.candidates(intent)
    assert len(cands) >= 1
    assert len(cands) <= 4


def test_candidate_runner_subprocess(tmp_path: Path) -> None:
    payload = {
        "actions": [Action.make("file_list", path=str(tmp_path)).to_dict()],
        "intent": Intent(verb="list", obj="directory", args=(("path", str(tmp_path)),)).to_dict(),
        "workdir": str(tmp_path),
        "offline": True,
        "timeout": 10.0,
    }
    proc = subprocess.run(
        [sys.executable, "-m", "el._candidate_runner"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["ok"] is True
    assert out["reward"] > 0


def test_seeding_preserves_same_verb_variants(tmp_path: Path) -> None:
    """Seeding must use exact-key dedup so many distinct intents sharing a
    verb are all inserted (the previous lookup() fallback matched verb|% and
    dropped variants)."""
    from el.seed.bootstrap import bundled_skills, seed_registry

    cfg = load_config(state_dir=tmp_path)
    cfg.ensure_dirs()
    registry = SkillRegistry(cfg.registry_path)
    stats = seed_registry(registry, use_bundled=True)

    bundled = list(bundled_skills())
    unique_keys = {intent.canonical_key() for intent, _, _ in bundled}
    # distinct same-verb variants must survive dedup
    verbs = [intent.verb for intent, _, _ in bundled]
    repeated_verbs = {v for v in verbs if verbs.count(v) > 1}
    assert repeated_verbs, "test requires the bundled corpus to contain same-verb variants"

    # every unique trigger key should be present
    for intent, _, _ in bundled:
        assert registry.has_exact_intent(intent), f"missing seeded intent: {intent.canonical_key()}"

    assert stats["added"] >= len(unique_keys)


def test_candidate_runner_blocks_destructive(tmp_path: Path) -> None:
    payload = {
        "actions": [Action.make("file_delete", path=str(tmp_path / "x"), confirmed=True).to_dict()],
        "intent": Intent(verb="delete", obj="file").to_dict(),
        "workdir": str(tmp_path),
        "offline": True,
        "timeout": 5.0,
    }
    proc = subprocess.run(
        [sys.executable, "-m", "el._candidate_runner"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["ok"] is False
    assert "destructive" in (out["results"][0].get("error") or "")
