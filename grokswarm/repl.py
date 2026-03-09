"""_chat_async, slash commands, SwarmCompleter, sessions, main() callback, show_welcome."""

import os
import sys
import json
import shutil
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime

import typer
import yaml
from typer import Context
from rich.panel import Panel
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.patch_stdout import patch_stdout
import grokswarm.shared as shared
from grokswarm.models import AgentState
from grokswarm.context import (
    scan_project_context, scan_project_context_cached, _save_context_cache,
    build_system_prompt,
)
from grokswarm.engine import _stream_with_tools, _trim_conversation
from grokswarm.agents import get_bus, _load_project_costs
from grokswarm.guardrails import Orchestrator, drain_notifications
from grokswarm.cmd_dispatch import CmdEntry, CmdContext, register as _cmd, get_command, all_commands, busy_allowed_set
from grokswarm.cmd_handlers import (
    handle_quit, handle_help, handle_clear, handle_context, handle_session,
    handle_list, handle_read, handle_write, handle_run, handle_search, handle_grep,
    handle_swarm, handle_experts, handle_skills, handle_git, handle_web, handle_x,
    handle_browse, handle_test, handle_undo, handle_trust, handle_readonly,
    handle_verbose, handle_project, handle_doctor, handle_dashboard, handle_metrics,
    handle_watch, handle_tell, handle_abort, handle_clear_swarm, handle_agents,
    handle_peek, handle_pause, handle_resume, handle_approve, handle_reject,
    handle_tasks, handle_budget, handle_model, handle_bugs, handle_memory,
    handle_eval, handle_self_improve, handle_daemon, handle_self_eval,
    handle_vim, handle_history, handle_self_scores,
    handle_diff, handle_copy,
)

# ---------------------------------------------------------------------------
# Command Registration — handlers wired to cmd_handlers.py
# ---------------------------------------------------------------------------
# fmt: off
_cmd("quit",         "Exit",                              aliases=["exit", "q"])(handle_quit)
_cmd("help",         "Show this help")(handle_help)
_cmd("clear",        "Clear conversation & screen")(handle_clear)
_cmd("context",      "Show project context (refresh to rescan)")(handle_context)
_cmd("session",      "Manage sessions (list/save/load/delete)")(handle_session)
_cmd("list",         "List project directory")(handle_list)
_cmd("read",         "Read file contents")(handle_read)
_cmd("write",        "Write/create file (with approval)")(handle_write)
_cmd("run",          "Run shell command (with approval)")(handle_run)
_cmd("search",       "Search files by name in project")(handle_search)
_cmd("grep",         "Search inside files for text")(handle_grep)
_cmd("swarm",        "Run multi-agent supervisor",         allow_while_busy=False)(handle_swarm)
_cmd("experts",      "List available experts")(handle_experts)
_cmd("skills",       "List available skills")(handle_skills)
_cmd("git",          "Git status (log, diff, branch)")(handle_git)
_cmd("web",          "Search the web (xAI live)")(handle_web)
_cmd("x",            "Search X/Twitter posts (xAI live)")(handle_x)
_cmd("browse",       "Fetch URL content (Playwright)")(handle_browse)
_cmd("test",         "Run project tests (auto-detect)")(handle_test)
_cmd("undo",         "Undo last file edit (multi-level)")(handle_undo)
_cmd("trust",        "Toggle trust mode (auto-approve)")(handle_trust)
_cmd("readonly",     "Toggle read-only mode (block writes)")(handle_readonly)
_cmd("verbose",      "Toggle output detail (compact/full)")(handle_verbose)
_cmd("project",      "Switch project directory")(handle_project)
_cmd("doctor",       "Check environment health")(handle_doctor)
_cmd("dashboard",    "Open live TUI dashboard")(handle_dashboard)
_cmd("metrics",      "Show token usage and cost metrics")(handle_metrics)
_cmd("watch",        "Live monitor for running agents")(handle_watch)
_cmd("tell",         "Send guidance to a running agent")(handle_tell)
_cmd("abort",        "Abort currently running swarm")(handle_abort)
_cmd("clear-swarm",  "Clear stale swarm data")(handle_clear_swarm)
_cmd("agents",       "List active agents and states")(handle_agents)
_cmd("peek",         "View agent plan/progress")(handle_peek)
_cmd("pause",        "Pause a running agent")(handle_pause)
_cmd("resume",       "Resume a paused agent")(handle_resume)
_cmd("approve",      "Approve agent plan")(handle_approve)
_cmd("reject",       "Reject plan and send feedback")(handle_reject)
_cmd("tasks",        "Show orchestrator task DAG")(handle_tasks)
_cmd("budget",       "Set/view session cost limit")(handle_budget)
_cmd("model",        "View/set model tiers")(handle_model)
_cmd("bugs",         "View/manage bugs")(handle_bugs)
_cmd("self-improve", "Improve own source (shadow + test)",  allow_while_busy=False)(handle_self_improve)
_cmd("memory",       "View/prune agent memory files")(handle_memory)
_cmd("eval",         "Run evaluation harness",              allow_while_busy=False)(handle_eval)
_cmd("daemon",       "File watcher daemon (auto-test)")(handle_daemon)
_cmd("self-eval",    "Eval -> fix -> re-eval loop",         allow_while_busy=False)(handle_self_eval)
_cmd("vim",          "Toggle vi editing mode")(handle_vim)
_cmd("history",      "Search command history",              aliases=["hist"])(handle_history)
_cmd("self-scores",  "Show latest eval scores",             aliases=["scores"])(handle_self_scores)
_cmd("diff",         "Show session file changes as diff")(handle_diff)
_cmd("copy",         "Copy last response to clipboard")(handle_copy)
# fmt: on

# -- Context-Aware Tab Completion --
class SwarmCompleter(Completer):
    """Smart completer: slash commands -> subcommands -> file paths / session names."""

    @staticmethod
    def _build_slash_commands() -> dict[str, str]:
        """Derive completer entries from the command registry."""
        cmds = {}
        for entry in all_commands().values():
            cmds[f"/{entry.name}"] = entry.description
            for alias in entry.aliases:
                cmds[f"/{alias}"] = entry.description
        return cmds

    SESSION_SUBCMDS = ["list", "save", "load", "delete", "search"]
    CONTEXT_SUBCMDS = ["refresh"]
    GIT_SUBCMDS = ["log", "diff", "branch"]
    PATH_COMMANDS = {"read", "edit", "list"}
    PROJECT_SUBCMDS = ["list", "switch"]
    MODEL_SUBCMDS = ["list", "reset", "fast", "reasoning", "hardcore", "multi_agent"]
    BUGS_SUBCMDS = ["list", "add", "show", "fix", "self", "project"]
    MEMORY_SUBCMDS = ["list", "prune"]
    DAEMON_SUBCMDS = ["start", "stop", "status", "log", "add"]

    # Class-level attribute — populated after class definition
    SLASH_COMMANDS: dict[str, str] = {}

    def __init__(self):
        self._path_completer = PathCompleter(only_directories=False, expanduser=True,
                                             get_paths=lambda: [str(shared.PROJECT_DIR)])
        self._dir_completer = PathCompleter(only_directories=True, expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        parts = text.split(maxsplit=1)
        cmd_part = parts[0]

        if len(parts) == 1 and not text.endswith(" "):
            for cmd, desc in self.SLASH_COMMANDS.items():
                if cmd.startswith(cmd_part.lower()):
                    yield Completion(cmd, start_position=-len(cmd_part), display_meta=desc)
            return

        cmd = cmd_part[1:].lower()
        arg_text = parts[1] if len(parts) > 1 else ""

        if cmd == "session":
            yield from self._complete_session(arg_text)
        elif cmd == "context":
            if not arg_text.endswith(" "):
                for sc in self.CONTEXT_SUBCMDS:
                    if sc.startswith(arg_text.lower()):
                        yield Completion(sc, start_position=-len(arg_text))
        elif cmd == "git":
            if not arg_text.endswith(" "):
                for sc in self.GIT_SUBCMDS:
                    if sc.startswith(arg_text.lower()):
                        yield Completion(sc, start_position=-len(arg_text))
        elif cmd == "model":
            if not arg_text.endswith(" "):
                for sc in self.MODEL_SUBCMDS:
                    if sc.startswith(arg_text.lower()):
                        yield Completion(sc, start_position=-len(arg_text))
        elif cmd == "bugs":
            if not arg_text.endswith(" "):
                for sc in self.BUGS_SUBCMDS:
                    if sc.startswith(arg_text.lower()):
                        yield Completion(sc, start_position=-len(arg_text))
        elif cmd == "memory":
            if not arg_text.endswith(" "):
                for sc in self.MEMORY_SUBCMDS:
                    if sc.startswith(arg_text.lower()):
                        yield Completion(sc, start_position=-len(arg_text))
        elif cmd == "daemon":
            if not arg_text.endswith(" "):
                for sc in self.DAEMON_SUBCMDS:
                    if sc.startswith(arg_text.lower()):
                        yield Completion(sc, start_position=-len(arg_text))
        elif cmd == "project":
            if not arg_text or not arg_text.endswith(" "):
                for sc in self.PROJECT_SUBCMDS:
                    if sc.startswith(arg_text.lower()):
                        yield Completion(sc, start_position=-len(arg_text))
            for rp in _load_recent_projects():
                name = Path(rp).name
                if name.lower().startswith(arg_text.lower()) or rp.lower().startswith(arg_text.lower()):
                    yield Completion(rp, start_position=-len(arg_text), display_meta="recent")
            sub_doc = Document(arg_text, len(arg_text))
            yield from self._dir_completer.get_completions(sub_doc, complete_event)
        elif cmd in self.PATH_COMMANDS:
            sub_doc = Document(arg_text, len(arg_text))
            yield from self._path_completer.get_completions(sub_doc, complete_event)

    def _complete_session(self, arg_text: str):
        arg_parts = arg_text.split(maxsplit=1)
        subcmd = arg_parts[0] if arg_parts else ""

        if len(arg_parts) <= 1 and not arg_text.endswith(" "):
            for sc in self.SESSION_SUBCMDS:
                if sc.startswith(subcmd.lower()):
                    yield Completion(sc, start_position=-len(subcmd))
            return

        if subcmd.lower() in ("load", "delete", "save"):
            name_prefix = arg_parts[1] if len(arg_parts) > 1 else ""
            for s in list_sessions():
                sname = s["name"]
                if sname.lower().startswith(name_prefix.lower()):
                    yield Completion(sname, start_position=-len(name_prefix),
                                    display_meta=f"{s['messages']} msgs")


# Populate SLASH_COMMANDS from the command registry at module load time.
SwarmCompleter.SLASH_COMMANDS = SwarmCompleter._build_slash_commands()


# -- Recent Projects --
def _load_recent_projects() -> list[str]:
    try:
        if shared._RECENT_PROJECTS_FILE.exists():
            return json.loads(shared._RECENT_PROJECTS_FILE.read_text(encoding="utf-8"))[:5]
    except Exception:
        pass
    return []


def _update_recent_projects(project_dir: Path):
    recents = _load_recent_projects()
    path_str = str(project_dir.resolve())
    recents = [p for p in recents if p != path_str]
    recents.insert(0, path_str)
    recents = recents[:5]
    try:
        shared._RECENT_PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        shared._RECENT_PROJECTS_FILE.write_text(json.dumps(recents), encoding="utf-8")
    except Exception:
        pass


def _switch_project(new_dir: str):
    p = Path(new_dir).resolve()
    if not p.is_dir():
        shared.console.print(f"[swarm.error]Not a directory: {new_dir}[/swarm.error]")
        return False
    shared.PROJECT_DIR = p
    _update_recent_projects(shared.PROJECT_DIR)
    shared.state.reset_project_state()
    shared.console.print(f"[swarm.accent]Switching to project:[/swarm.accent] [bold]{shared.PROJECT_DIR}[/bold]")
    shared.PROJECT_CONTEXT = scan_project_context_cached(shared.PROJECT_DIR)
    shared.SYSTEM_PROMPT = build_system_prompt(shared.PROJECT_CONTEXT)
    _load_project_costs()
    file_count = len(shared.PROJECT_CONTEXT.get('key_files', {}))
    shared.console.print(f"[swarm.dim]  context: {file_count} key file{'s' if file_count != 1 else ''} loaded[/swarm.dim]")
    return True


async def _switch_project_async(new_dir: str):
    return await asyncio.to_thread(_switch_project, new_dir)


# -- Session Persistence --
def _session_path(name: str) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
    return shared.SESSIONS_DIR / f"{safe_name}.json"


def _build_session_summary(messages: list) -> str:
    """Build a concise text summary of conversation messages for session restore."""
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        if role == "user":
            parts.append(f"User: {content[:200]}")
        elif role == "assistant" and content:
            parts.append(f"Grok: {content[:200]}")
        elif role == "assistant" and m.get("tool_calls"):
            tools = [tc["function"]["name"] for tc in m["tool_calls"]]
            parts.append(f"Grok used tools: {', '.join(tools)}")
    return "\n".join(parts[-30:])  # last 30 exchanges max


def save_session(name: str, conversation: list):
    msgs = [m for m in conversation if m["role"] != "system"]
    data = {
        "name": name,
        "project": str(shared.PROJECT_DIR),
        "updated": datetime.now().isoformat(),
        "message_count": len(msgs),
        "summary": _build_session_summary(msgs),
        "messages": msgs,
    }
    _session_path(name).write_text(shared._redact_secrets(json.dumps(data, indent=2)))


def load_session(name: str) -> tuple[list, str] | None:
    """Load a session. Returns (messages, summary) or None."""
    path = _session_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        msgs = data.get("messages", [])
        summary = data.get("summary", "")
        if not summary and msgs:
            summary = _build_session_summary(msgs)
        return msgs, summary
    except (json.JSONDecodeError, KeyError):
        return None


def list_sessions() -> list[dict]:
    sessions = []
    for f in sorted(shared.SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "name": data.get("name", f.stem),
                "updated": data.get("updated", "unknown"),
                "messages": data.get("message_count", 0),
                "project": data.get("project", "unknown"),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return sessions


def delete_session(name: str) -> bool:
    path = _session_path(name)
    if path.exists():
        path.unlink()
        return True
    return False


def _show_help():
    shared.console.print()
    shared.console.print("[swarm.accent]Commands[/swarm.accent]")
    for entry in all_commands().values():
        name = f"/{entry.name}"
        aliases = ""
        if entry.aliases:
            aliases = f" (/{'/'.join(entry.aliases)})"
        shared.console.print(f"  [bold]{name + aliases:<24}[/bold] [dim]{entry.description}[/dim]")
    shared.console.print()


def _show_context(arg: str = ""):
    shared.console.print()
    shared.console.print("[swarm.accent]Project Context[/swarm.accent]")
    shared.console.print(f"  [bold]Project:[/bold]   {shared.PROJECT_CONTEXT['project_name']}")
    shared.console.print(f"  [bold]Directory:[/bold] {shared.PROJECT_CONTEXT['project_dir']}")
    shared.console.print()
    shared.console.print("[swarm.accent]File Tree[/swarm.accent]")
    shared.console.print(f"[dim]{shared.PROJECT_CONTEXT['tree']}[/dim]")
    shared.console.print()
    key_files = shared.PROJECT_CONTEXT.get("key_files", {})
    if key_files:
        shared.console.print(f"[swarm.accent]Key Files Loaded ({len(key_files)})[/swarm.accent]")
        for fname in key_files:
            size = len(key_files[fname])
            shared.console.print(f"  [bold]{fname:<25}[/bold] [dim]{size:,} chars[/dim]")
    else:
        shared.console.print("[swarm.dim]  No key files found.[/swarm.dim]")

    lang_stats = shared.PROJECT_CONTEXT.get("language_stats", {})
    if lang_stats:
        shared.console.print()
        shared.console.print("[swarm.accent]Languages[/swarm.accent]")
        stats_str = "  ".join(f"[bold]{ext}[/bold]:{count}" for ext, count in list(lang_stats.items())[:10])
        shared.console.print(f"  {stats_str}")

    code_struct = shared.PROJECT_CONTEXT.get("code_structure", {})
    if code_struct:
        shared.console.print()
        shared.console.print(f"[swarm.accent]Code Structure ({len(code_struct)} files)[/swarm.accent]")
        for filepath, defs in list(code_struct.items())[:10]:
            shared.console.print(f"  [bold]{filepath}[/bold]")
            for d in defs[:5]:
                shared.console.print(f"    [dim]{d}[/dim]")
            if len(defs) > 5:
                shared.console.print(f"    [dim]... ({len(defs) - 5} more)[/dim]")

    shared.console.print()
    if arg == "refresh":
        shared.console.print("[swarm.accent]Context refreshed from disk.[/swarm.accent]")
        shared.console.print()


def _handle_session_command(arg: str, conversation: list, current_session: str | None) -> str | None:
    parts = arg.split(maxsplit=1) if arg else []
    subcmd = parts[0].lower() if parts else ""
    subarg = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "list" or not subcmd:
        sessions = list_sessions()
        if not sessions:
            shared.console.print("[swarm.dim]No saved sessions.[/swarm.dim]")
            return None
        shared.console.print()
        shared.console.print("[swarm.accent]Saved Sessions[/swarm.accent]")
        for s in sessions:
            marker = " [bold green]< active[/bold green]" if current_session and s["name"] == current_session else ""
            ts = s["updated"][:16].replace("T", " ") if s["updated"] != "unknown" else "?"
            shared.console.print(f"  [bold]{s['name']:<20}[/bold] [dim]{s['messages']} msgs  *  {ts}[/dim]{marker}")
        shared.console.print()
        return None
    elif subcmd == "save":
        if not subarg:
            if current_session:
                subarg = current_session
            else:
                shared.console.print("[swarm.warning]Usage: /session save <name>[/swarm.warning]")
                return None
        save_session(subarg, conversation)
        shared.console.print(f"[swarm.accent]Session '[bold]{subarg}[/bold]' saved ({len([m for m in conversation if m['role'] != 'system'])} messages).[/swarm.accent]")
        return subarg
    elif subcmd == "load":
        if not subarg:
            sessions = list_sessions()
            if not sessions:
                shared.console.print("[swarm.dim]No saved sessions to load.[/swarm.dim]")
                return None
            names = [s["name"] for s in sessions]
            shared.console.print("[swarm.accent]Sessions:[/swarm.accent]")
            for i, n in enumerate(names, 1):
                shared.console.print(f"  [bold]{i}.[/bold] {n}")
            try:
                from prompt_toolkit import prompt as pt_prompt
                from prompt_toolkit.completion import FuzzyWordCompleter
                pick = pt_prompt("load session> ", completer=FuzzyWordCompleter(names)).strip()
            except (EOFError, KeyboardInterrupt):
                shared.console.print("[swarm.dim]Cancelled.[/swarm.dim]")
                return None
            if not pick:
                shared.console.print("[swarm.dim]Cancelled.[/swarm.dim]")
                return None
            # Allow picking by number
            if pick.isdigit() and 1 <= int(pick) <= len(names):
                subarg = names[int(pick) - 1]
            else:
                subarg = pick
        result = load_session(subarg)
        if result:
            msgs, summary = result
            conversation.clear()
            conversation.append({"role": "system", "content": shared.SYSTEM_PROMPT})
            conversation.extend(msgs)
            # Inject resume context so the model knows this is a restored session
            if summary:
                conversation.append({"role": "user", "content": f"[SESSION RESTORED] This is a continuation of a previous conversation named '{subarg}'. Here is a summary of what we discussed:\n{summary}\n\nPlease continue from where we left off."})
                conversation.append({"role": "assistant", "content": f"I remember our previous conversation. I have the full history loaded ({len(msgs)} messages). Let's continue where we left off."})
            shared.console.print(f"[swarm.accent]Loaded session '[bold]{subarg}[/bold]' ({len(msgs)} messages). Auto-saving enabled.[/swarm.accent]")
            return subarg
        else:
            shared.console.print(f"[swarm.warning]Session '{subarg}' not found.[/swarm.warning]")
            return None
    elif subcmd == "delete":
        if not subarg:
            shared.console.print("[swarm.warning]Usage: /session delete <name>[/swarm.warning]")
            return None
        if delete_session(subarg):
            shared.console.print(f"[swarm.accent]Session '{subarg}' deleted.[/swarm.accent]")
        else:
            shared.console.print(f"[swarm.warning]Session '{subarg}' not found.[/swarm.warning]")
        return None
    elif subcmd == "search":
        if not subarg:
            shared.console.print("[swarm.warning]Usage: /session search <query>[/swarm.warning]")
            return None
        query_lower = subarg.lower()
        hits = []
        for f in shared.SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                # Search in summary, name, and message content
                searchable = data.get("summary", "") + " " + data.get("name", "")
                for m in data.get("messages", []):
                    c = m.get("content")
                    if isinstance(c, str):
                        searchable += " " + c
                if query_lower in searchable.lower():
                    hits.append({
                        "name": data.get("name", f.stem),
                        "updated": data.get("updated", "?"),
                        "messages": data.get("message_count", 0),
                    })
            except Exception:
                continue
        if hits:
            shared.console.print(f"\n[swarm.accent]Sessions matching '{subarg}' ({len(hits)} found)[/swarm.accent]")
            for h in hits:
                ts = h["updated"][:16].replace("T", " ") if h["updated"] != "?" else "?"
                shared.console.print(f"  [bold]{h['name']:<20}[/bold] [dim]{h['messages']} msgs  *  {ts}[/dim]")
            shared.console.print()
        else:
            shared.console.print(f"[swarm.dim]No sessions matching '{subarg}'.[/swarm.dim]")
        return None
    else:
        shared.console.print("[swarm.dim]Usage: /session [list|save|load|delete|search] <name>[/swarm.dim]")
        return None


def _run_doctor():
    checks = []
    import sys as _sys
    checks.append(("Python", f"{_sys.version.split()[0]}", True))
    has_key = bool(os.environ.get("XAI_API_KEY"))
    checks.append(("XAI_API_KEY", "set" if has_key else "MISSING", has_key))
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True, timeout=5)
        checks.append(("git", "installed", True))
    except Exception:
        checks.append(("git", "not found", False))
    checks.append(("ripgrep (rg)", "installed" if shutil.which("rg") else "not found (optional)", shutil.which("rg") is not None))
    try:
        import playwright  # noqa: F811
        _pw_dir = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH",
                       os.environ.get("LOCALAPPDATA", "")))
        if _pw_dir.name != "ms-playwright":
            _pw_dir = _pw_dir / "ms-playwright"
        if not _pw_dir.exists():
            _pw_dir = Path.home() / ".cache" / "ms-playwright"
        if _pw_dir.exists() and any(_pw_dir.glob("chromium-*")):
            checks.append(("playwright", "installed + chromium ready", True))
        else:
            checks.append(("playwright", "installed (run: playwright install chromium)", False))
    except ImportError:
        checks.append(("playwright", "not installed (optional)", False))
    checks.append(("project dir", str(shared.PROJECT_DIR), shared.PROJECT_DIR.is_dir()))
    cfg_exists = (shared.PROJECT_DIR / ".grokswarm.yml").exists()
    checks.append((".grokswarm.yml", "found" if cfg_exists else "not found (optional)", cfg_exists))
    shared.console.print()
    shared.console.print("[swarm.accent]Doctor \u2014 Environment Check[/swarm.accent]")
    for label, value, ok in checks:
        icon = "[bold green]\u2713[/bold green]" if ok else "[bold yellow]\u25cb[/bold yellow]"
        shared.console.print(f"  {icon} [bold]{label:<20}[/bold] {value}")
    shared.console.print()


def show_welcome(session_name: str | None = None):
    shared.console.print()
    shared.console.print(Panel(
        f"[bold white]Grok Swarm[/bold white]  [dim]v{shared.VERSION}[/dim]\n"
        f"[dim]model: {shared.MODEL}[/dim]",
        border_style="bright_green",
        padding=(1, 2),
        width=42,
    ))
    display_dir = os.environ.get("GROKSWARM_HOST_DIR") or str(shared.PROJECT_DIR)
    shared.console.print(f"[swarm.dim]  project:    [bold]{display_dir}[/bold][/swarm.dim]")
    if shared.PROJECT_CONTEXT:
        file_count = len(shared.PROJECT_CONTEXT.get('key_files', {}))
        shared.console.print(f"[swarm.dim]  context:    {file_count} key file{'s' if file_count != 1 else ''} loaded[/swarm.dim]")
    if session_name:
        shared.console.print(f"[swarm.dim]  session:    [bold]{session_name}[/bold] (auto-saving)[/swarm.dim]")
    else:
        shared.console.print("[swarm.dim]  session:    (ephemeral -- use /session save <name> to persist)[/swarm.dim]")
    shared.console.print("[swarm.dim]  /help for commands * /context to view * tab to complete[/swarm.dim]")
    shared.console.print()


async def _swarm_async(description: str):
    from grokswarm.dashboard import _watch_agents
    from rich.rule import Rule

    bus = get_bus()
    bus.clear()
    shared.state.clear_swarm()

    # Use orchestrator for task decomposition and DAG-based execution
    orchestrator_task = asyncio.create_task(Orchestrator.run(description, bus))

    await _watch_agents(task_description=description)

    # Wait for orchestrator to finish if not done
    try:
        await orchestrator_task
    except Exception as e:
        shared.console.print(f"[swarm.error]Orchestrator error: {e}[/swarm.error]")

    shared.console.print()
    shared.console.print(Rule("[bold]Swarm Summary[/bold]", style="cyan"))
    summary_table = Table(show_header=True, header_style="bold", border_style="dim", width=110)
    summary_table.add_column("Agent", style="cyan", width=18)
    summary_table.add_column("Status", width=10)
    summary_table.add_column("Tokens", justify="right", width=10)
    summary_table.add_column("Cost", justify="right", width=10)
    summary_table.add_column("Output", ratio=1)
    for agent_name, agent in shared.state.agents.items():
        agent_state = agent.state.value
        tokens = agent.tokens_used
        status_color = "green" if agent_state == "done" else "red" if agent_state == "error" else "yellow"
        summary_table.add_row(
            agent_name,
            f"[{status_color}]{agent_state}[/{status_color}]",
            f"{tokens:,}",
            f"${agent.cost_usd:.4f}",
            agent.task[:60],
        )
    shared.console.print(summary_table)
    total_tokens = sum(a.tokens_used for a in shared.state.agents.values())
    shared.console.print(f"  [dim]Total: {len(shared.state.agents)} agents, {total_tokens:,} tokens, ${shared.state.global_cost_usd:.4f}[/dim]")

    # Show task DAG summary if available
    dag = getattr(shared, '_current_dag', None)
    if dag and dag.subtasks:
        done = sum(1 for t in dag.subtasks if t.status == "done")
        failed = sum(1 for t in dag.subtasks if t.status == "failed")
        shared.console.print(f"  [dim]DAG: {done}/{len(dag.subtasks)} tasks done, {failed} failed[/dim]")
    shared.console.print()


@shared.app.callback(invoke_without_command=True)
def main(
    ctx: Context,
    session: str = typer.Option(None, "--session", "-s", help="Resume or start a named session"),
    model: str = typer.Option(None, "--model", "-m", help="Override model name (e.g. grok-3-latest)"),
    base_url: str = typer.Option(None, "--base-url", help="Override API base URL (e.g. https://api.openai.com/v1)"),
    api_key: str = typer.Option(None, "--api-key", help="Override API key (instead of XAI_API_KEY env var)"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="Override max output tokens per response (default: 16384)"),
    project_dir: str = typer.Option(None, "--project-dir", "-d", help="Set project directory (default: current working directory)"),
):
    from grokswarm.context import IGNORE_DIRS, _IGNORE_PATTERNS, _IGNORE_LITERALS
    import grokswarm.context as context_mod

    if project_dir:
        p = Path(project_dir).resolve()
        if p.is_dir():
            shared.PROJECT_DIR = p
        else:
            shared.console.print(f"[swarm.warning]Warning: --project-dir '{project_dir}' not found, using cwd[/swarm.warning]")

    config_path = shared.PROJECT_DIR / ".grokswarm.yml"
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not model and cfg.get("model"):
                shared.MODEL = cfg["model"]
            if not max_tokens and cfg.get("max_tokens"):
                shared.MAX_TOKENS = int(cfg["max_tokens"])
            if not base_url and cfg.get("base_url"):
                shared.BASE_URL = cfg["base_url"]
                from grokswarm import llm
                llm.reset_client(api_key=os.getenv(cfg.get("api_key_env", "XAI_API_KEY")) or shared.XAI_API_KEY)
            if cfg.get("code_model"):
                shared.CODE_MODEL = cfg["code_model"]
            extra_ignore = cfg.get("ignore_dirs", [])
            if extra_ignore:
                context_mod.IGNORE_DIRS = context_mod.IGNORE_DIRS | set(extra_ignore)
                context_mod._IGNORE_PATTERNS = [p for p in context_mod.IGNORE_DIRS if "*" in p]
                context_mod._IGNORE_LITERALS = context_mod.IGNORE_DIRS - set(context_mod._IGNORE_PATTERNS)
        except Exception as e:
            shared.console.print(f"[swarm.warning]Warning: .grokswarm.yml: {e}[/swarm.warning]")

    if model:
        shared.MODEL = model
    if max_tokens:
        shared.MAX_TOKENS = max_tokens
    if base_url:
        shared.BASE_URL = base_url
    if api_key or base_url:
        from grokswarm import llm
        llm.reset_client(api_key=api_key or shared.XAI_API_KEY)

    if ctx.invoked_subcommand is None:
        show_welcome(session)
        _update_recent_projects(shared.PROJECT_DIR)
        with shared.console.status("[bold cyan]Scanning project...[/bold cyan]", spinner="dots"):
            shared.PROJECT_CONTEXT = scan_project_context_cached(shared.PROJECT_DIR)
        shared.SYSTEM_PROMPT = build_system_prompt(shared.PROJECT_CONTEXT)
        _load_project_costs()
        # Auto-prune old memory + bus messages on startup
        from grokswarm.registry_helpers import startup_cleanup
        cleanup = startup_cleanup()
        if cleanup["memory_pruned"] or cleanup["bus_pruned"]:
            shared.console.print(f"[swarm.dim]  cleanup: pruned {cleanup['memory_pruned']} old memories, {cleanup['bus_pruned']} old bus messages[/swarm.dim]")
        asyncio.run(_chat_async(session_name=session))


@shared.app.command()
def chat(session_name: str = typer.Argument(None, hidden=True)):
    """Interactive mode with tab completion + streaming."""
    asyncio.run(_chat_async(session_name))


async def _chat_async(session_name: str | None = None):
    """Async implementation of the interactive chat loop."""
    from grokswarm.dashboard import dashboard, _watch_agents

    history_file = Path("~/.grokswarm/history.txt").expanduser()
    history_file.parent.mkdir(exist_ok=True, parents=True)

    completer = SwarmCompleter()

    kb = KeyBindings()

    @kb.add('escape')
    def _handle_escape(event):
        if shared.state.vi_mode:
            return  # let prompt_toolkit's vi handler process escape
        buf = event.current_buffer
        if buf.complete_state:
            buf.cancel_completion()
        elif processing_busy:
            shared._cancel_event.set()
        elif buf.text:
            buf.document = Document('')

    @kb.add('backspace', eager=True)
    def _handle_backspace(event):
        buf = event.current_buffer
        if buf.text:
            buf.delete_before_cursor(1)
            if buf.text.startswith('/'):
                buf.start_completion()
            elif buf.complete_state:
                buf.cancel_completion()

    @kb.add('s-tab', eager=True)
    def _handle_shift_tab(event):
        if not shared.state.trust_mode and not shared.state.read_only:
            shared.state.trust_mode = True
            shared.state.read_only = False
        elif shared.state.trust_mode:
            shared.state.trust_mode = False
            shared.state.read_only = True
        else:
            shared.state.trust_mode = False
            shared.state.read_only = False
        event.app.invalidate()

    @kb.add('c-x', 'c-e', eager=True)
    def _handle_edit_in_editor(event):
        event.current_buffer.open_in_editor(event.app)

    @kb.add('escape', 'v', eager=True)
    def _handle_paste_image(event):
        import time as _time
        try:
            from PIL import ImageGrab
        except ImportError:
            shared._toolbar_status = "Pillow not installed (pip install Pillow)"
            shared._toolbar_status_expires = _time.monotonic() + 2.0
            event.app.invalidate()
            return
        try:
            import base64
            from io import BytesIO
            img = ImageGrab.grabclipboard()
            if img is None:
                shared._toolbar_status = "No image in clipboard"
                shared._toolbar_status_expires = _time.monotonic() + 2.0
                event.app.invalidate()
                return
            buf = BytesIO()
            img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            shared._pending_images.append(f"data:image/png;base64,{b64}")
            event.app.invalidate()
        except Exception:
            shared._toolbar_status = "Failed to grab clipboard image"
            shared._toolbar_status_expires = _time.monotonic() + 2.0
            event.app.invalidate()

    _multiline_supported = False
    try:
        @kb.add('s-enter', eager=True)
        def _handle_shift_enter(event):
            event.current_buffer.insert_text('\n')
        _multiline_supported = True
    except (ValueError, KeyError):
        pass

    if _multiline_supported:
        @kb.add('enter', eager=True)
        def _handle_enter(event):
            buf = event.current_buffer
            if buf.complete_state:
                buf.complete_state = None
            else:
                buf.validate_and_handle()

    session = PromptSession(history=shared.SafeFileHistory(str(history_file)), completer=completer,
                            complete_while_typing=True, key_bindings=kb, multiline=_multiline_supported,
                            erase_when_done=True, auto_suggest=AutoSuggestFromHistory(),
                            enable_history_search=True)
    session.app.ttimeoutlen = 0.01
    session.app.timeoutlen = 0.01

    shared._toolbar_app_ref = session.app
    _toolbar_spinner_task = None

    async def _spinner_tick():
        import time as _time
        while True:
            await asyncio.sleep(0.12)
            if shared._toolbar_status_expires and _time.monotonic() >= shared._toolbar_status_expires:
                shared._toolbar_status = ""
                shared._toolbar_status_expires = 0.0
            if shared._toolbar_status and not shared._toolbar_suspended and not shared._is_prompt_suspended:
                shared._toolbar_spinner_idx += 1
                if shared._toolbar_app_ref:
                    try:
                        shared._toolbar_app_ref.invalidate()
                    except Exception:
                        pass

    shared._open_session_log()
    conversation = [{"role": "system", "content": shared.SYSTEM_PROMPT}]

    if session_name:
        result = load_session(session_name)
        if result:
            saved_msgs, summary = result
            conversation.extend(saved_msgs)
            # Inject resume context so the model knows this is a restored session
            if summary:
                conversation.append({"role": "user", "content": f"[SESSION RESTORED] This is a continuation of a previous conversation named '{session_name}'. Here is a summary of what we discussed:\n{summary}\n\nPlease continue from where we left off."})
                conversation.append({"role": "assistant", "content": f"I remember our previous conversation. I have the full history loaded ({len(saved_msgs)} messages). Let's continue where we left off."})
            shared.console.print(f"[swarm.accent]Resumed session '[bold]{session_name}[/bold]' ({len(saved_msgs)} messages)[/swarm.accent]")
        else:
            shared.console.print(f"[swarm.accent]Started new session '[bold]{session_name}[/bold]'[/swarm.accent]")
        shared.console.print()

    processing_busy = False

    def _fmt_tokens(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return str(n)

    def get_bottom_toolbar():
        if shared.state.trust_mode:
            mode_str = "<ansigreen>TRUST</ansigreen>"
        elif shared.state.read_only:
            mode_str = "<ansiyellow>READ-ONLY</ansiyellow>"
        else:
            mode_str = "<ansidarkgray>NORMAL</ansidarkgray>"
        parts = []
        if shared._toolbar_status and not shared._is_prompt_suspended:
            icon = shared.THINKING_FRAMES[shared._toolbar_spinner_idx % len(shared.THINKING_FRAMES)]
            parts.append(f"  <ansicyan>{icon}</ansicyan> <ansiwhite>{shared._toolbar_status}</ansiwhite>")
        # Show active agents with current step
        active_agents = [(n, a) for n, a in shared.state.agents.items()
                         if a.state in (AgentState.THINKING, AgentState.WORKING)]
        if active_agents:
            agent_parts = []
            for aname, ag in active_agents[:2]:
                if ag.plan:
                    done = sum(1 for s in ag.plan if s["status"] == "done")
                    total = len(ag.plan)
                    bar_w = 10
                    filled = int(bar_w * done / total) if total else 0
                    bar = "\u2501" * filled + "\u2578" + "\u2500" * max(0, bar_w - filled - 1)
                    agent_parts.append(f"{aname} [{bar}] {done}/{total}")
                else:
                    agent_parts.append(f"{aname}: {ag.phase.title()}...")
            parts.append(f"  <ansidarkgray>{len(active_agents)} agent{'s' if len(active_agents) != 1 else ''} running | {' | '.join(agent_parts)}</ansidarkgray>")
        # Context window usage bar
        try:
            from grokswarm.engine import _estimate_tokens
            est = _estimate_tokens(conversation)
            ctx_limit = shared._get_context_window(shared.MODEL)
            pct = min(est / ctx_limit, 1.0) if ctx_limit else 0
            bar_w = 10
            filled = int(bar_w * pct)
            ctx_bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            pct_int = int(pct * 100)
            ctx_color = "ansired" if pct_int >= 80 else ("ansiyellow" if pct_int >= 60 else "ansidarkgray")
            ctx_tag = f" <ansidarkgray>|</ansidarkgray> <{ctx_color}>[{ctx_bar}] {pct_int}%</{ctx_color}>"
        except Exception:
            ctx_tag = ""
        # Status line: project | model | tokens | cost | ctx% | [VI] | >> MODE
        proj_name = shared.PROJECT_DIR.name
        model_short = shared.MODEL.split("/")[-1].split("-")[-1][:12] if shared.MODEL else "?"
        tok_str = _fmt_tokens(shared.state.global_tokens_used)
        cost_str = f"${shared.state.project_cost_usd:.3f}"
        vi_tag = " <ansicyan>[VI]</ansicyan>" if getattr(shared.state, 'vi_mode', False) else ""
        parts.append(
            f"  <ansidarkgray>{proj_name}</ansidarkgray>"
            f" <ansidarkgray>|</ansidarkgray> <ansidarkgray>{model_short}</ansidarkgray>"
            f" <ansidarkgray>|</ansidarkgray> <ansidarkgray>{tok_str} tok</ansidarkgray>"
            f" <ansidarkgray>|</ansidarkgray> <ansidarkgray>{cost_str}</ansidarkgray>"
            f"{ctx_tag}{vi_tag}"
            f" <ansidarkgray>|</ansidarkgray> <ansimagenta>\u25b6\u25b6</ansimagenta> {mode_str}"
        )
        return HTML("\n".join(parts))

    def get_prompt_continuation(width, line_number, is_soft_wrap):
        spaces = max(0, width - 2)
        return HTML(f"{' ' * spaces}<ansidarkgray>\u00b7 </ansidarkgray>")

    _toolbar_spinner_task = asyncio.create_task(_spinner_tick())

    async def _process_input_queue():
        nonlocal conversation, processing_busy
        while True:
            user_input = await shared._input_queue.get()
            if user_input is None:
                break
            processing_busy = True
            shared._cancel_event.clear()
            try:
                text = shared._sanitize_surrogates(user_input)
                if shared._pending_images:
                    content = [{"type": "text", "text": text}]
                    for uri in shared._pending_images:
                        content.append({"type": "image_url", "image_url": {"url": uri, "detail": "auto"}})
                    shared._pending_images.clear()
                    conversation.append({"role": "user", "content": content})
                else:
                    conversation.append({"role": "user", "content": text})
                conversation = await _trim_conversation(conversation)
                shared.state.request_auto_approve = False
                await _stream_with_tools(conversation)
                if session_name:
                    save_session(session_name, conversation)
            except KeyboardInterrupt:
                shared.console.print("\n  [swarm.warning]\u26a0 Interrupted.[/swarm.warning]")
            except Exception as e:
                shared.console.print(f"[swarm.error]Error: {e}[/swarm.error]")
            finally:
                processing_busy = False
                shared._cancel_event.clear()
                shared._clear_status()

    processor_task = asyncio.create_task(_process_input_queue())

    with patch_stdout(raw=True):
        try:
            while True:
                if shared._is_prompt_suspended:
                    shared._prompt_suspend_event.set()
                    await shared._prompt_resume_event.wait()
                    shared._prompt_resume_event.clear()
                    continue

                # -- Drain guardrails notifications --
                for _level, _msg in drain_notifications():
                    if _level == "warning":
                        shared.console.print(f"[bold yellow]{_msg}[/]")
                    elif _level == "error":
                        shared.console.print(f"[bold red]{_msg}[/]")
                    else:
                        shared.console.print(f"[dim]{_msg}[/]")

                def get_message():
                    _cols = shutil.get_terminal_size((80, 20)).columns
                    line_str = "\u2500" * _cols
                    parts = [""]  # blank line for breathing room
                    if shared._toolbar_status and not shared._is_prompt_suspended:
                        icon = shared.THINKING_FRAMES[shared._toolbar_spinner_idx % len(shared.THINKING_FRAMES)]
                        parts.append(f"  <ansicyan>{icon}</ansicyan> <ansidarkgray>{shared._toolbar_status}</ansidarkgray>")
                    if shared._pending_images:
                        count = len(shared._pending_images)
                        label = "image attached" if count == 1 else f"{count} images attached"
                        parts.append(f"  <ansiyellow>[{label}]</ansiyellow>")
                    parts.append(f"<style fg='#444444'>{line_str}</style>")
                    parts.append("<b><ansibrightcyan>> </ansibrightcyan></b>")
                    return HTML("\n".join(parts))

                default_text = shared._saved_prompt_text
                shared._saved_prompt_text = ""

                user_input = await session.prompt_async(
                    get_message,
                    bottom_toolbar=get_bottom_toolbar,
                    prompt_continuation=get_prompt_continuation,
                    default=default_text
                )

                if user_input == "__MAGIC_SUSPEND__":
                    continue

                user_input = user_input.strip()
                if not user_input:
                    continue

                lines = user_input.split('\n')
                img_tag = ""
                if shared._pending_images:
                    n = len(shared._pending_images)
                    img_tag = f" [yellow][{'image' if n == 1 else f'{n} images'} attached][/yellow]"
                shared.console.print(f"[bold cyan]> [/bold cyan][bright_white]{lines[0]}[/bright_white]{img_tag}")
                for l in lines[1:]:
                    shared.console.print(f"  [dim]\u00b7 [/dim][bright_white]{l}[/bright_white]")

                # -- Slash Commands (dispatched via cmd_dispatch registry) --
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0][1:].lower()
                    arg = parts[1] if len(parts) > 1 else ""

                    if processing_busy and cmd not in busy_allowed_set():
                        shared.console.print("[swarm.dim]Busy processing previous prompt. Wait, or use /abort.[/swarm.dim]")
                        continue

                    entry = get_command(cmd)
                    if entry:
                        ctx = CmdContext(
                            conversation=conversation,
                            session_name=session_name,
                            session=session,
                            save_session=save_session,
                        )
                        result = entry.handler(arg, ctx)
                        if asyncio.iscoroutine(result):
                            await result
                        if ctx.quit_flag:
                            break
                        if ctx.new_session_name is not None:
                            session_name = ctx.new_session_name
                    else:
                        shared.console.print(f"[swarm.dim]Unknown command: /{cmd} -- type /help[/swarm.dim]")
                    continue

                # -- Exit without slash --
                if user_input.lower() in ["exit", "quit", "q"]:
                    if session_name:
                        save_session(session_name, conversation)
                        shared.console.print(f"[swarm.dim]Session '{session_name}' saved.[/swarm.dim]")
                    shared.console.print("[swarm.dim]Goodbye.[/swarm.dim]")
                    break

                # -- Process conversational input via background queue --
                await shared._input_queue.put(user_input)
        except KeyboardInterrupt:
            shared.console.print("\n[swarm.dim]Interrupted. Type /quit to exit.[/swarm.dim]")
        except EOFError:
            if session_name:
                save_session(session_name, conversation)
                shared.console.print(f"[swarm.dim]Session '{session_name}' saved.[/swarm.dim]")
            shared.console.print("[swarm.dim]Goodbye.[/swarm.dim]")
        except Exception as e:
            shared.console.print(f"[swarm.error]Error: {e}[/swarm.error]")
        finally:
            # Auto-save on any exit (crash, Ctrl+C, exception) if session is named
            if session_name and conversation and len(conversation) > 1:
                try:
                    save_session(session_name, conversation)
                    shared.console.print(f"[swarm.dim]Session '{session_name}' auto-saved.[/swarm.dim]")
                except Exception:
                    pass
            shared._clear_status()
            if _toolbar_spinner_task:
                _toolbar_spinner_task.cancel()
            processor_task.cancel()


