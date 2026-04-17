from pathlib import Path

from el.intent import Intent
from el.primitives import Action
from el.registry import SkillRegistry


def _intent(verb="list", scope="this_folder") -> Intent:
    return Intent(verb=verb, scope=scope)


def test_add_and_lookup(tmp_path: Path):
    reg = SkillRegistry(tmp_path / "r.sqlite3")
    skill = reg.add_skill(_intent(), [Action.make("file_list", path=".")])
    hits = reg.lookup(_intent())
    assert len(hits) == 1
    assert hits[0].id == skill.id


def test_reinforce_success_raises_weight(tmp_path: Path):
    reg = SkillRegistry(tmp_path / "r.sqlite3")
    skill = reg.add_skill(_intent(), [Action.make("noop")])
    before = skill.weight
    after = reg.reinforce(skill.id, success=True)
    assert after.weight > before
    assert after.success_count == 1


def test_reinforce_failure_drops_weight(tmp_path: Path):
    reg = SkillRegistry(tmp_path / "r.sqlite3")
    skill = reg.add_skill(_intent(), [Action.make("noop")], weight=0.7)
    after = reg.reinforce(skill.id, success=False)
    assert after.weight < 0.7
    assert after.failure_count == 1


def test_decay_removes_unused_low_weight(tmp_path: Path):
    reg = SkillRegistry(tmp_path / "r.sqlite3")
    reg.add_skill(_intent(), [Action.make("noop")], weight=0.01)
    removed = reg.decay(factor=0.5)
    assert removed >= 1


def test_lookup_fallback_by_verb(tmp_path: Path):
    reg = SkillRegistry(tmp_path / "r.sqlite3")
    reg.add_skill(_intent(scope="downloads"), [Action.make("noop")])
    hits = reg.lookup(_intent(scope="nonexistent"))
    assert hits
    assert hits[0].intent.verb == "list"


def test_export_training_rows(tmp_path: Path):
    reg = SkillRegistry(tmp_path / "r.sqlite3")
    skill = reg.add_skill(_intent(), [Action.make("noop")])
    reg.reinforce(skill.id, success=True)
    rows = reg.export_training_rows()
    assert rows and rows[0]["intent"]["verb"] == "list"


def test_event_log(tmp_path: Path):
    reg = SkillRegistry(tmp_path / "r.sqlite3")
    reg.log_event("test", {"k": "v"})
    evs = reg.events(kind="test")
    assert evs and evs[0]["payload"]["k"] == "v"
