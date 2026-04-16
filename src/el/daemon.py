"""Daemon mode — autonomous maintenance loop.

`el --daemon` runs a periodic loop that triggers registered canonical
skills without user prompting. Autonomy is bounded: the daemon only ever
runs skills marked safe (non-destructive, non-networked by default).
"""
from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from typing import Callable

from .executor import Executor


@dataclass
class ScheduledTask:
    name: str
    command: str
    every_seconds: float
    last_run: float = 0.0

    def due(self, now: float) -> bool:
        return now - self.last_run >= self.every_seconds


DEFAULT_SCHEDULE: tuple[ScheduledTask, ...] = (
    ScheduledTask(name="heartbeat", command="help", every_seconds=60.0),
    ScheduledTask(name="git_snapshot", command="git status", every_seconds=300.0),
    ScheduledTask(name="disk_snapshot", command="report disk usage here", every_seconds=900.0),
)


@dataclass
class Daemon:
    executor: Executor
    tasks: tuple[ScheduledTask, ...] = field(default_factory=lambda: DEFAULT_SCHEDULE)
    tick_seconds: float = 5.0
    decay_every: int = 12  # ticks between registry decay sweeps
    retire_threshold: float = 0.1
    _running: bool = True
    _on_event: Callable[[str, dict], None] | None = None

    def run(self, max_iterations: int | None = None) -> int:
        self._install_handlers()
        iters = 0
        while self._running:
            now = time.time()
            for task in self.tasks:
                if task.due(now):
                    result = self.executor.handle(task.command)
                    task.last_run = now
                    if self._on_event:
                        self._on_event(
                            "daemon_tick",
                            {
                                "task": task.name,
                                "reward": result.reward,
                                "origin": result.origin,
                            },
                        )
            iters += 1
            if iters % max(1, self.decay_every) == 0:
                try:
                    summary = self.executor.registry.decay_and_retire(
                        retire_threshold=self.retire_threshold
                    )
                    if self._on_event:
                        self._on_event("daemon_decay", summary)
                except Exception as exc:  # keep daemon alive
                    if self._on_event:
                        self._on_event("daemon_decay_error", {"error": str(exc)})
            if max_iterations is not None and iters >= max_iterations:
                break
            time.sleep(self.tick_seconds)
        return iters

    def stop(self) -> None:
        self._running = False

    def _install_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                pass

    def _handle_signal(self, signum, frame) -> None:  # noqa: ANN001
        self._running = False
