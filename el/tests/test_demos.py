import os
from pathlib import Path

from el.config import load_config
from el.demos import DEMOS, ephemeral_workspace
from el.executor import Executor
from el.parser import Parser
from el.registry import SkillRegistry
from el.seed.bootstrap import seed_registry
from el.selfplay import SelfPlayRunner


def test_all_demos_have_unique_names():
    names = [d.name for d in DEMOS]
    assert len(set(names)) == len(names)


def test_at_least_15_demos():
    assert len(DEMOS) >= 15


def test_offline_demo_suite_all_pass(tmp_path: Path):
    state = tmp_path / "state"
    config = load_config(state_dir=state, offline=True)
    registry = SkillRegistry(config.registry_path)
    seed_registry(registry, use_bundled=True)
    workspace = ephemeral_workspace()
    cwd = os.getcwd()
    os.chdir(workspace)
    try:
        passed = 0
        offline_total = 0
        for demo in DEMOS:
            if demo.online:
                continue
            offline_total += 1
            executor = Executor(
                config=config,
                parser=Parser(),
                registry=SkillRegistry(config.registry_path),
                selfplay=SelfPlayRunner(),
            )
            result = executor.handle(demo.command)
            if result.origin in {"registry", "selfplay"} and result.reward >= 0.3:
                passed += 1
        assert offline_total >= 10
        assert passed == offline_total, f"{passed}/{offline_total} demos passed"
    finally:
        os.chdir(cwd)
