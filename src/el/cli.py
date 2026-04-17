"""el CLI entrypoint.

Usage:

    el "<free-text command>"      execute a command directly
    el --daemon                   run autonomous maintenance loop
    el --daemon --iterations 10   bounded daemon run (for CI/tests)

Subcommands:

    el run "<text>"               explicit form of the default command
    el seed [--from-bundled]      bootstrap registry from seed data
    el demo [--all|--name N]      run curated demos
    el stats                      show registry stats
    el events [--kind K]          show recent events
    el daemon                     explicit form of --daemon
    el parse "<text>"             show how the parser interprets a command
    el rate up|down               reinforce (or weaken) the last executed skill
    el verbs                      list grammar verbs
    el export-training            dump training rows for the transformer

The first positional argument that is not a known subcommand is treated as
a command string, so `el "list this folder"` works without `run`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import click

from .config import load_config
from .daemon import Daemon
from .demos import DEMOS, demo_by_name, ephemeral_workspace
from .executor import Executor
from .parser import Parser, list_verbs
from .registry import SkillRegistry
from .selfplay import SelfPlayRunner
from .seed.bootstrap import seed_registry


_SUBCOMMAND_NAMES: set[str] = set()


class ElGroup(click.Group):
    """Group that treats an unknown first positional as a free-text command."""

    def resolve_command(self, ctx: click.Context, args: list[str]):  # type: ignore[override]
        if args and args[0] not in _SUBCOMMAND_NAMES and not args[0].startswith("-"):
            args = ["run", *args]
        return super().resolve_command(ctx, args)


@click.group(cls=ElGroup, context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.option("--state-dir", type=click.Path(), default=None, help="State directory (defaults to $EL_STATE_DIR or ~/.el_state).")
@click.option("--offline", is_flag=True, help="Disable network primitives.")
@click.option("--yes", is_flag=True, help="Auto-confirm destructive actions.")
@click.option("--daemon", "as_daemon", is_flag=True, help="Run the autonomous maintenance loop.")
@click.option("--iterations", type=int, default=None, help="Bounded daemon iterations (with --daemon).")
@click.option("--tick", type=float, default=5.0, help="Daemon tick seconds (with --daemon).")
@click.pass_context
def main(
    ctx: click.Context,
    state_dir: str | None,
    offline: bool,
    yes: bool,
    as_daemon: bool,
    iterations: int | None,
    tick: float,
) -> None:
    """el — an LLM-free PC agent with dual-memory architecture."""
    overrides: dict[str, Any] = {"offline": offline}
    if state_dir:
        overrides["state_dir"] = Path(state_dir)
    config = load_config(**overrides)
    config.ensure_dirs()
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["auto_confirm"] = yes

    if as_daemon:
        executor = _build_executor(ctx)
        daemon = Daemon(executor=executor, tick_seconds=tick)
        daemon._on_event = lambda kind, payload: click.echo(f"[daemon] {kind} {json.dumps(payload, default=str)}")
        ran = daemon.run(max_iterations=iterations)
        click.echo(f"[daemon] exited after {ran} tick(s)")
        ctx.exit(0)

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command("run")
@click.argument("command", nargs=-1, required=True)
@click.pass_context
def cli_run(ctx: click.Context, command: tuple[str, ...]) -> None:
    """Execute a free-text command."""
    raw = " ".join(command)
    _run_command(ctx, raw)


@main.command("parse")
@click.argument("command", nargs=-1, required=True)
@click.pass_context
def cli_parse(ctx: click.Context, command: tuple[str, ...]) -> None:
    """Show how the parser interprets a command."""
    raw = " ".join(command)
    intents = Parser().parse(raw)
    click.echo(json.dumps([i.to_dict() for i in intents], indent=2, ensure_ascii=False))


@main.command("seed")
@click.option("--from-bundled/--no-bundled", default=True, help="Use bundled seed corpus.")
@click.option("--extra", type=click.Path(exists=True), default=None, help="Extra JSONL file of seed triples.")
@click.pass_context
def cli_seed(ctx: click.Context, from_bundled: bool, extra: str | None) -> None:
    """Bootstrap the skill registry from seed data."""
    config = ctx.obj["config"]
    registry = SkillRegistry(config.registry_path)
    stats = seed_registry(registry, use_bundled=from_bundled, extra_path=Path(extra) if extra else None)
    click.echo(json.dumps(stats, indent=2, ensure_ascii=False))


@main.command("stats")
@click.pass_context
def cli_stats(ctx: click.Context) -> None:
    """Show skill registry statistics."""
    config = ctx.obj["config"]
    registry = SkillRegistry(config.registry_path)
    click.echo(json.dumps(registry.stats(), indent=2, ensure_ascii=False))


@main.command("events")
@click.option("--kind", default=None, help="Filter by event kind.")
@click.option("--limit", type=int, default=50)
@click.pass_context
def cli_events(ctx: click.Context, kind: str | None, limit: int) -> None:
    """Show recent registry events."""
    config = ctx.obj["config"]
    registry = SkillRegistry(config.registry_path)
    events = registry.events(kind=kind, limit=limit)
    click.echo(json.dumps(events, indent=2, ensure_ascii=False, default=str))


@main.command("verbs")
def cli_verbs() -> None:
    """List all grammar verbs."""
    for v in list_verbs():
        click.echo(v)


@main.command("export-training")
@click.option("--out", type=click.Path(), default="training.jsonl")
@click.pass_context
def cli_export_training(ctx: click.Context, out: str) -> None:
    """Dump training rows for the action transformer."""
    config = ctx.obj["config"]
    registry = SkillRegistry(config.registry_path)
    rows = registry.export_training_rows()
    with open(out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    click.echo(f"wrote {len(rows)} rows to {out}")


@main.command("daemon")
@click.option("--iterations", type=int, default=None, help="Exit after N ticks (for testing).")
@click.option("--tick", type=float, default=5.0, help="Seconds between ticks.")
@click.pass_context
def cli_daemon(ctx: click.Context, iterations: int | None, tick: float) -> None:
    """Run the autonomous maintenance loop."""
    executor = _build_executor(ctx)
    daemon = Daemon(executor=executor, tick_seconds=tick)
    daemon._on_event = lambda kind, payload: click.echo(f"[daemon] {kind} {json.dumps(payload, default=str)}")
    ran = daemon.run(max_iterations=iterations)
    click.echo(f"[daemon] exited after {ran} tick(s)")


@main.command("rate")
@click.argument("direction", type=click.Choice(["up", "down"]))
@click.pass_context
def cli_rate(ctx: click.Context, direction: str) -> None:
    """Reinforce or weaken the most recently executed skill."""
    config = ctx.obj["config"]
    registry = SkillRegistry(config.registry_path)
    last_path = Path(config.state_dir) / "last_run.json"
    if not last_path.exists():
        click.echo(json.dumps({"ok": False, "error": "no previous run to rate"}))
        ctx.exit(1)
    info = json.loads(last_path.read_text(encoding="utf-8"))
    skill_id = info.get("skill_id")
    if not skill_id:
        click.echo(json.dumps({"ok": False, "error": "last run had no skill match"}))
        ctx.exit(1)
    updated = registry.reinforce(skill_id, success=(direction == "up"))
    registry.log_event("user_rating", {"skill_id": skill_id, "direction": direction, "new_weight": updated.weight})
    click.echo(json.dumps({"ok": True, "skill_id": skill_id, "direction": direction, "weight": updated.weight}))


@main.command("train")
@click.option("--preset", type=click.Choice(["tiny", "small", "h200"]), default="tiny")
@click.option("--steps", type=int, default=2000)
@click.option("--batch", type=int, default=16)
@click.option("--lr", type=float, default=3e-4)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto")
@click.option("--out", type=click.Path(), default=None, help="Checkpoint dir (default: <state>/checkpoints/action-transformer).")
@click.option("--no-synth", is_flag=True, help="Use only bundled corpus, skip synthetic.")
@click.option("--synth-per-verb", type=int, default=120)
@click.option("--extra", type=click.Path(exists=True), default=None)
@click.pass_context
def cli_train(
    ctx: click.Context, preset: str, steps: int, batch: int, lr: float,
    device: str, out: str | None, no_synth: bool, synth_per_verb: int, extra: str | None,
) -> None:
    """Train the Action Transformer on bundled+synthetic corpus."""
    config = ctx.obj["config"]
    out_dir = Path(out) if out else config.state_dir / "checkpoints" / "action-transformer"
    argv = [
        "el.transformer.train",
        "--preset", preset, "--steps", str(steps), "--batch", str(batch),
        "--lr", str(lr), "--device", device, "--out", str(out_dir),
        "--synth-per-verb", str(synth_per_verb),
    ]
    if no_synth:
        argv.append("--no-synth")
    if extra:
        argv += ["--extra", extra]
    sys.argv = argv
    from .transformer.train import main as train_main
    rc = train_main()
    if rc != 0:
        ctx.exit(rc)


@main.command("transformer-eval")
@click.option("--checkpoint", type=click.Path(exists=True), default=None,
              help="Checkpoint dir (default: <state>/checkpoints/action-transformer).")
@click.option("--max-eval", type=int, default=300)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto")
@click.option("--report", type=click.Path(), default=None)
@click.pass_context
def cli_transformer_eval(
    ctx: click.Context, checkpoint: str | None, max_eval: int, device: str, report: str | None,
) -> None:
    """Evaluate a trained Action Transformer on the held-out test split."""
    config = ctx.obj["config"]
    ckpt = Path(checkpoint) if checkpoint else config.state_dir / "checkpoints" / "action-transformer"
    argv = [
        "el.transformer.eval",
        "--checkpoint", str(ckpt),
        "--max-eval", str(max_eval), "--device", device,
    ]
    if report:
        argv += ["--report", report]
    sys.argv = argv
    from .transformer.eval import main as eval_main
    rc = eval_main()
    if rc != 0:
        ctx.exit(rc)


@main.command("demo")
@click.option("--all", "run_all", is_flag=True, help="Run the full demo suite.")
@click.option("--name", "single", default=None, help="Run a single demo by name.")
@click.option("--offline", is_flag=True, help="Skip demos marked online.")
@click.option("--json", "json_out", is_flag=True, help="Emit JSON report only.")
@click.option("--workspace", is_flag=True, help="Run inside an ephemeral workspace.")
@click.pass_context
def cli_demo(
    ctx: click.Context,
    run_all: bool,
    single: str | None,
    offline: bool,
    json_out: bool,
    workspace: bool,
) -> None:
    """Run curated demo commands."""
    if offline:
        ctx.obj["config"] = load_config(
            state_dir=ctx.obj["config"].state_dir,
            offline=True,
        )
    prev_cwd = os.getcwd()
    tmp = None
    if workspace:
        tmp = ephemeral_workspace()
        os.chdir(tmp)
    try:
        if single:
            demo = demo_by_name(single)
            if not demo:
                raise click.BadParameter(f"unknown demo: {single}")
            demos = (demo,)
        elif run_all:
            demos = DEMOS
        else:
            click.echo("available demos:")
            for d in DEMOS:
                mark = "online" if d.online else "offline"
                click.echo(f"  [{mark}] {d.name} — {d.description}")
            return
        report = {"demos": []}
        for d in demos:
            if offline and d.online:
                report["demos"].append({"name": d.name, "status": "skipped", "duration_ms": 0, "reward": 0.0, "origin": "offline"})
                if not json_out:
                    click.echo(f"[skip-online] {d.name}")
                continue
            executor = _build_executor(ctx)
            t0 = time.monotonic()
            result = executor.handle(d.command)
            dt = int((time.monotonic() - t0) * 1000)
            status = "ok" if result.origin in {"registry", "selfplay", "transformer"} else "fail"
            report["demos"].append(
                {
                    "name": d.name,
                    "status": status,
                    "duration_ms": dt,
                    "reward": result.reward,
                    "origin": result.origin,
                }
            )
            if not json_out:
                click.echo(f"[{status}] {d.name}  reward={result.reward:.2f}  origin={result.origin}  {dt}ms")
        if json_out:
            click.echo(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        if tmp is not None:
            os.chdir(prev_cwd)


def _build_executor(ctx: click.Context) -> Executor:
    config = ctx.obj["config"]
    registry = SkillRegistry(config.registry_path)
    selfplay = SelfPlayRunner(num_candidates=config.selfplay_candidates)
    try:
        from .transformer.adapter import TransformerAdapter

        transformer = TransformerAdapter.try_load(config)
    except Exception:
        transformer = None
    return Executor(
        config=config,
        registry=registry,
        selfplay=selfplay,
        transformer=transformer,
        auto_confirm=ctx.obj.get("auto_confirm", False),
    )


def _run_command(ctx: click.Context, raw: str) -> None:
    config = ctx.obj["config"]
    executor = _build_executor(ctx)
    result = executor.handle(raw)
    try:
        last_path = Path(config.state_dir) / "last_run.json"
        last_path.write_text(
            json.dumps(
                {
                    "raw": raw,
                    "ts": time.time(),
                    "skill_id": result.skill.id if result.skill else None,
                    "reward": result.reward,
                    "origin": result.origin,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
    click.echo(
        json.dumps(
            {
                "intent": result.intent.to_dict(),
                "origin": result.origin,
                "reward": result.reward,
                "skill_id": result.skill.id if result.skill else None,
                "actions": [a.to_dict() for a in result.actions],
                "duration_ms": result.duration_ms,
                "results": [r.to_dict() for r in result.results],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    if result.origin == "no_plan":
        sys.exit(2)


for _name in ("run", "parse", "seed", "stats", "events", "verbs", "export-training", "daemon", "rate", "demo"):
    _SUBCOMMAND_NAMES.add(_name)


if __name__ == "__main__":
    main()
