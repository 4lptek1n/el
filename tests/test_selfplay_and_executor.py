import os
from pathlib import Path

from el.config import Config, load_config
from el.executor import Executor
from el.intent import Intent
from el.parser import Parser
from el.primitives import Action
from el.registry import SkillRegistry
from el.selfplay import SelfPlayRunner


def _mkconfig(tmp_path: Path, **overrides) -> Config:
    return load_config(state_dir=tmp_path, offline=True, **overrides)


def test_selfplay_returns_plan_for_known_verb():
    planner = SelfPlayRunner(num_candidates=3)
    plan = planner.plan(Intent(verb="list", scope="this_folder"))
    assert plan
    assert plan[0].name == "file_list"


def test_selfplay_empty_for_download_without_url():
    planner = SelfPlayRunner(num_candidates=3)
    plan = planner.plan(Intent(verb="download"))
    assert plan == []


def test_executor_skill_hit(tmp_path: Path):
    config = _mkconfig(tmp_path)
    os.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hi")
    registry = SkillRegistry(config.registry_path)
    intent = Intent(verb="list", scope="this_folder")
    registry.add_skill(intent, [Action.make("file_list", path=".")], weight=0.9)
    executor = Executor(config=config, registry=registry, parser=Parser(), selfplay=None)
    result = executor.handle("list this folder")
    assert result.origin == "registry"
    assert result.skill is not None
    assert result.reward > 0.5


def test_executor_selfplay_miss_graduates_skill(tmp_path: Path):
    config = _mkconfig(tmp_path)
    os.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("a")
    executor = Executor(
        config=config,
        registry=SkillRegistry(config.registry_path),
        parser=Parser(),
        selfplay=SelfPlayRunner(num_candidates=3),
    )
    res = executor.handle("list this folder")
    assert res.origin in {"selfplay", "registry"}
    assert res.reward > 0.0


def test_executor_unknown_returns_no_plan(tmp_path: Path):
    config = _mkconfig(tmp_path)
    executor = Executor(
        config=config,
        registry=SkillRegistry(config.registry_path),
        parser=Parser(),
        selfplay=SelfPlayRunner(num_candidates=3),
    )
    res = executor.handle("zxqzxq blargh")
    assert res.origin in {"unknown", "no_plan"}
