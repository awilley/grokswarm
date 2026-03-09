"""Bug tracker for GrokSwarm — two scopes:

1. Self-bugs:    ~/.grokswarm/bugs.json  — issues with GrokSwarm itself
2. Project-bugs: .grokswarm/bugs.json    — issues in the current project

Both use the same JSON format and BugTracker API.
"""

import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import grokswarm.shared as shared

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

BugSeverity = Literal["low", "medium", "high", "critical"]
BugStatus = Literal["open", "in_progress", "fixed", "wont_fix", "duplicate"]

_SELF_BUGS_FILE = Path.home() / ".grokswarm" / "bugs.json"


@dataclass
class Bug:
    id: int
    title: str
    description: str
    severity: BugSeverity = "medium"
    status: BugStatus = "open"
    created: str = ""
    updated: str = ""
    source: str = ""          # "auto" | "agent" | "user" | "eval"
    context: dict = field(default_factory=dict)  # file, error, traceback, agent, etc.

    def __post_init__(self):
        now = datetime.now().isoformat(timespec="seconds")
        if not self.created:
            self.created = now
        if not self.updated:
            self.updated = now


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

class BugTracker:
    """JSON-file bug tracker. One instance per file path."""

    def __init__(self, path: Path):
        self.path = path
        self._ensure_file()

    def _ensure_file(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]")

    def _load(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, bugs: list[dict]):
        try:
            self.path.write_text(json.dumps(bugs, indent=2))
        except OSError:
            pass

    def _next_id(self, bugs: list[dict]) -> int:
        if not bugs:
            return 1
        return max(b.get("id", 0) for b in bugs) + 1

    def log(self, title: str, description: str, severity: BugSeverity = "medium",
            source: str = "user", context: dict | None = None) -> Bug:
        """Log a new bug. Returns the created Bug."""
        bugs = self._load()
        bug = Bug(
            id=self._next_id(bugs),
            title=title,
            description=description,
            severity=severity,
            source=source,
            context=context or {},
        )
        bugs.append(asdict(bug))
        self._save(bugs)
        return bug

    def list(self, status: str | None = None, severity: str | None = None) -> list[Bug]:
        """List bugs, optionally filtered by status or severity."""
        bugs = self._load()
        result = []
        for b in bugs:
            if status and b.get("status") != status:
                continue
            if severity and b.get("severity") != severity:
                continue
            result.append(Bug(**b))
        return result

    def get(self, bug_id: int) -> Bug | None:
        """Get a specific bug by ID."""
        for b in self._load():
            if b.get("id") == bug_id:
                return Bug(**b)
        return None

    def update(self, bug_id: int, **kwargs) -> Bug | None:
        """Update a bug's fields (status, severity, description, etc.)."""
        bugs = self._load()
        for b in bugs:
            if b.get("id") == bug_id:
                for key, value in kwargs.items():
                    if key in b:
                        b[key] = value
                b["updated"] = datetime.now().isoformat(timespec="seconds")
                self._save(bugs)
                return Bug(**b)
        return None

    def count(self, status: str | None = "open") -> int:
        """Count bugs, optionally filtered by status."""
        return len(self.list(status=status))


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

_self_tracker: BugTracker | None = None
_project_tracker: BugTracker | None = None
_project_tracker_dir: Path | None = None


def get_self_tracker() -> BugTracker:
    """Get the global GrokSwarm self-bug tracker (~/.grokswarm/bugs.json)."""
    global _self_tracker
    if _self_tracker is None:
        _self_tracker = BugTracker(_SELF_BUGS_FILE)
    return _self_tracker


def get_project_tracker() -> BugTracker:
    """Get the project-specific bug tracker (.grokswarm/bugs.json)."""
    global _project_tracker, _project_tracker_dir
    project_dir = shared.PROJECT_DIR
    if _project_tracker is None or _project_tracker_dir != project_dir:
        _project_tracker = BugTracker(project_dir / ".grokswarm" / "bugs.json")
        _project_tracker_dir = project_dir
    return _project_tracker


# ---------------------------------------------------------------------------
# Convenience functions (used by auto-logging and agent tool)
# ---------------------------------------------------------------------------

def log_self_bug(title: str, description: str, severity: BugSeverity = "medium",
                 source: str = "auto", context: dict | None = None) -> Bug:
    """Log a bug against GrokSwarm itself."""
    return get_self_tracker().log(title, description, severity, source, context)


def log_project_bug(title: str, description: str, severity: BugSeverity = "medium",
                    source: str = "agent", context: dict | None = None) -> Bug:
    """Log a bug against the current project."""
    return get_project_tracker().log(title, description, severity, source, context)


# ---------------------------------------------------------------------------
# Auto-logging hooks (called from guardrails, engine, agents)
# ---------------------------------------------------------------------------

def log_exception(exc: Exception, context_label: str = ""):
    """Auto-log an unhandled exception as a GrokSwarm self-bug."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_str = "".join(tb[-3:])  # last 3 frames
    log_self_bug(
        title=f"Unhandled {type(exc).__name__}: {str(exc)[:80]}",
        description=f"{context_label}\n\n{tb_str}",
        severity="high",
        source="auto",
        context={"exception_type": type(exc).__name__, "label": context_label},
    )


def log_loop_detection(agent_name: str, escalation_count: int, error_sig: str = ""):
    """Auto-log when loop detector fires (potential GrokSwarm or task issue)."""
    if escalation_count >= 2:
        # Repeated loops suggest a systemic issue
        log_self_bug(
            title=f"Repeated loop in agent '{agent_name}' ({escalation_count} escalations)",
            description=f"Agent got stuck in a loop pattern. Error signature: {error_sig[:200]}",
            severity="medium",
            source="auto",
            context={"agent": agent_name, "escalations": escalation_count, "error_sig": error_sig},
        )


def log_tool_error(tool_name: str, error: str, agent_name: str = ""):
    """Auto-log a tool execution error."""
    log_self_bug(
        title=f"Tool error: {tool_name}: {error[:60]}",
        description=f"Tool '{tool_name}' failed during execution.\nAgent: {agent_name or 'repl'}\nError: {error[:500]}",
        severity="low",
        source="auto",
        context={"tool": tool_name, "agent": agent_name},
    )


def log_guardrail_failure(guardrail: str, detail: str, agent_name: str = ""):
    """Auto-log when a guardrail detects a problem."""
    log_self_bug(
        title=f"Guardrail '{guardrail}' triggered: {detail[:60]}",
        description=f"Guardrail: {guardrail}\nAgent: {agent_name or 'repl'}\nDetail: {detail[:500]}",
        severity="low",
        source="auto",
        context={"guardrail": guardrail, "agent": agent_name},
    )


# ---------------------------------------------------------------------------
# Agent tool implementation
# ---------------------------------------------------------------------------

def report_bug_impl(title: str, description: str, severity: str = "medium",
                    scope: str = "project") -> str:
    """Agent-callable: report a bug. scope='project' or 'self'."""
    sev = severity if severity in ("low", "medium", "high", "critical") else "medium"
    if scope == "self":
        bug = log_self_bug(title, description, sev, source="agent")
        return f"GrokSwarm self-bug #{bug.id} logged: {title}"
    else:
        bug = log_project_bug(title, description, sev, source="agent")
        return f"Project bug #{bug.id} logged: {title}"


def list_bugs_impl(scope: str = "project", status: str = "open") -> str:
    """Agent-callable: list bugs."""
    tracker = get_self_tracker() if scope == "self" else get_project_tracker()
    st = status if status in ("open", "in_progress", "fixed", "wont_fix", "duplicate") else None
    bugs = tracker.list(status=st)
    if not bugs:
        return f"No {status or 'any'} bugs in {scope} tracker."
    lines = [f"{scope.upper()} BUGS ({len(bugs)} {status or 'total'}):"]
    for b in bugs:
        lines.append(f"  #{b.id} [{b.severity}] [{b.status}] {b.title}")
        if b.description:
            desc_preview = b.description.split("\n")[0][:80]
            lines.append(f"      {desc_preview}")
    return "\n".join(lines)


def update_bug_impl(bug_id: int, status: str = "", severity: str = "",
                    scope: str = "project") -> str:
    """Agent-callable: update a bug's status or severity."""
    tracker = get_self_tracker() if scope == "self" else get_project_tracker()
    kwargs = {}
    if status:
        kwargs["status"] = status
    if severity:
        kwargs["severity"] = severity
    if not kwargs:
        return "Nothing to update. Provide status or severity."
    bug = tracker.update(bug_id, **kwargs)
    if bug:
        return f"Bug #{bug.id} updated: {', '.join(f'{k}={v}' for k, v in kwargs.items())}"
    return f"Bug #{bug_id} not found in {scope} tracker."
