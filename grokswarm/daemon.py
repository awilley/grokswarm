"""Autonomous daemon mode — watches files and runs agents on changes.

Usage:
    /daemon start          — start watching for file changes
    /daemon stop           — stop the daemon
    /daemon status         — show daemon state
    /daemon add <pattern>  — add a watch pattern (e.g. *.py, tests/*.py)
    /daemon log            — show recent daemon actions
"""

import os
import time
import asyncio
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import grokswarm.shared as shared


@dataclass
class DaemonAction:
    timestamp: str
    trigger: str       # "file_changed:foo.py" or "test_failed"
    action: str        # what the daemon did
    result: str        # brief outcome


@dataclass
class DaemonState:
    running: bool = False
    watch_patterns: list[str] = field(default_factory=lambda: ["*.py"])
    file_hashes: dict[str, str] = field(default_factory=dict)
    actions: list[DaemonAction] = field(default_factory=list)
    _task: asyncio.Task | None = field(default=None, repr=False)
    poll_interval: float = 2.0    # seconds between scans
    auto_test: bool = True        # run tests on change
    auto_lint: bool = False       # run linter on change
    max_actions: int = 50         # keep last N actions

    def log_action(self, trigger: str, action: str, result: str):
        self.actions.append(DaemonAction(
            timestamp=datetime.now().isoformat()[:19],
            trigger=trigger,
            action=action,
            result=result,
        ))
        if len(self.actions) > self.max_actions:
            self.actions = self.actions[-self.max_actions:]


_daemon = DaemonState()


def get_daemon() -> DaemonState:
    return _daemon


def _hash_file(path: Path) -> str:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _scan_files(project_dir: Path, patterns: list[str]) -> dict[str, str]:
    """Scan project for files matching patterns and return {path: hash}."""
    result = {}
    for pattern in patterns:
        for f in project_dir.glob(f"**/{pattern}"):
            # Skip hidden dirs, __pycache__, .git, node_modules
            parts = f.relative_to(project_dir).parts
            if any(p.startswith(".") or p in ("__pycache__", "node_modules", ".git", "venv", ".venv") for p in parts):
                continue
            rel = str(f.relative_to(project_dir))
            result[rel] = _hash_file(f)
    return result


async def _daemon_loop():
    """Main daemon loop — polls for file changes and triggers actions."""
    from grokswarm.tools_test import run_tests

    daemon = _daemon
    project_dir = shared.PROJECT_DIR

    # Initial scan
    daemon.file_hashes = _scan_files(project_dir, daemon.watch_patterns)
    daemon.log_action("daemon", "started", f"watching {len(daemon.file_hashes)} files")

    try:
        while daemon.running:
            await asyncio.sleep(daemon.poll_interval)

            if not daemon.running:
                break

            # Scan for changes
            new_hashes = _scan_files(project_dir, daemon.watch_patterns)
            changed = []
            added = []
            deleted = []

            for path, h in new_hashes.items():
                if path not in daemon.file_hashes:
                    added.append(path)
                elif daemon.file_hashes[path] != h:
                    changed.append(path)

            for path in daemon.file_hashes:
                if path not in new_hashes:
                    deleted.append(path)

            daemon.file_hashes = new_hashes

            if not (changed or added or deleted):
                continue

            # Log the change
            change_desc = []
            if changed:
                change_desc.append(f"modified: {', '.join(changed[:5])}")
            if added:
                change_desc.append(f"added: {', '.join(added[:5])}")
            if deleted:
                change_desc.append(f"deleted: {', '.join(deleted[:5])}")
            trigger = "; ".join(change_desc)

            # Auto-test if enabled and a .py file changed
            py_changed = any(f.endswith(".py") for f in changed + added)
            if daemon.auto_test and py_changed:
                try:
                    shared.console.print(f"[dim][daemon] Change detected: {trigger}[/dim]")
                    test_result = await asyncio.to_thread(run_tests, None)
                    # Parse test result
                    passed = "passed" in test_result.lower()
                    short_result = test_result.strip().split("\n")[-1][:120] if test_result else "no output"
                    status = "PASS" if passed else "FAIL"
                    daemon.log_action(trigger[:100], f"auto-test ({status})", short_result)
                    if not passed:
                        shared.console.print(f"[bold yellow][daemon] Tests failed after change:[/bold yellow]")
                        shared.console.print(f"[dim]{short_result}[/dim]")
                    else:
                        shared.console.print(f"[dim][daemon] Tests passed after change.[/dim]")
                except Exception as e:
                    daemon.log_action(trigger[:100], "auto-test (error)", str(e)[:120])

    except asyncio.CancelledError:
        pass
    finally:
        daemon.running = False
        daemon.log_action("daemon", "stopped", "")


async def start_daemon():
    """Start the daemon file watcher."""
    daemon = _daemon
    if daemon.running:
        return "Daemon is already running."
    daemon.running = True
    daemon._task = asyncio.create_task(_daemon_loop())
    return f"Daemon started. Watching {', '.join(daemon.watch_patterns)} with {daemon.poll_interval}s interval."


async def stop_daemon():
    """Stop the daemon."""
    daemon = _daemon
    if not daemon.running:
        return "Daemon is not running."
    daemon.running = False
    if daemon._task:
        daemon._task.cancel()
        try:
            await daemon._task
        except (asyncio.CancelledError, Exception):
            pass
        daemon._task = None
    return "Daemon stopped."


def daemon_status() -> str:
    """Get daemon status."""
    daemon = _daemon
    lines = [
        f"Running: {'yes' if daemon.running else 'no'}",
        f"Patterns: {', '.join(daemon.watch_patterns)}",
        f"Files tracked: {len(daemon.file_hashes)}",
        f"Auto-test: {'on' if daemon.auto_test else 'off'}",
        f"Poll interval: {daemon.poll_interval}s",
        f"Actions logged: {len(daemon.actions)}",
    ]
    return "\n".join(lines)


def daemon_log(n: int = 10) -> str:
    """Get recent daemon actions."""
    daemon = _daemon
    if not daemon.actions:
        return "No daemon actions logged."
    lines = []
    for a in daemon.actions[-n:]:
        lines.append(f"[{a.timestamp}] {a.trigger} -> {a.action}: {a.result}")
    return "\n".join(lines)


def add_watch_pattern(pattern: str) -> str:
    """Add a file pattern to watch."""
    daemon = _daemon
    if pattern not in daemon.watch_patterns:
        daemon.watch_patterns.append(pattern)
    return f"Watching: {', '.join(daemon.watch_patterns)}"
