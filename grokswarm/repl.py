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
from prompt_toolkit.patch_stdout import patch_stdout
from openai import AsyncOpenAI

import grokswarm.shared as shared
from grokswarm.models import AgentState
from grokswarm.context import (
    scan_project_context, scan_project_context_cached, _save_context_cache,
    build_system_prompt, _safe_path,
)
from grokswarm.tools_fs import list_dir, read_file, write_file, search_files, grep_files
from grokswarm.tools_shell import run_shell
from grokswarm.tools_test import run_tests
from grokswarm.tools_git import git_status, git_diff, git_log, git_branch
from grokswarm.tools_search import web_search, x_search
from grokswarm.tools_browser import fetch_page
from grokswarm.registry_helpers import list_experts, list_skills
from grokswarm.engine import _stream_with_tools, _trim_conversation
from grokswarm.agents import (
    get_bus, run_supervisor, run_expert, _load_project_costs,
    _list_agents_impl,
)
from grokswarm.guardrails import PlanGate, Orchestrator, drain_notifications


# -- Context-Aware Tab Completion --
class SwarmCompleter(Completer):
    """Smart completer: slash commands -> subcommands -> file paths / session names."""

    SLASH_COMMANDS = {
        "/help": "Show this help",
        "/list": "List project directory",
        "/read": "Read file contents",
        "/write": "Write/create file (with approval)",
        "/run": "Run shell command (with approval)",
        "/search": "Search files by name in project",
        "/grep": "Search inside files for text",
        "/git": "Git status (log, diff, branch)",
        "/web": "Search the web (xAI live)",
        "/x": "Search X/Twitter posts (xAI live)",
        "/browse": "Fetch URL content (Playwright)",
        "/test": "Run project tests (auto-detect)",
        "/undo": "Undo last file edit (multi-level)",
        "/trust": "Toggle trust mode (auto-approve)",
        "/readonly": "Toggle read-only mode (block writes)",
        "/project": "Switch project directory",
        "/doctor": "Check environment health",
        "/dashboard": "Open live TUI dashboard",
        "/metrics": "Show token usage and cost metrics",
        "/self-improve": "Improve own source (shadow + test)",
        "/swarm": "Run multi-agent supervisor",
        "/watch": "Live monitor for running background agents",
        "/abort": "Abort currently running swarm",
        "/tell": "Send guidance to a running agent: /tell <agent> <message>",
        "/clear-swarm": "Clear stale swarm data (agents, bus messages)",
        "/experts": "List available experts",
        "/skills": "List available skills",
        "/agents": "List active agents and their states",
        "/peek": "View agent plan/progress: /peek [name]",
        "/pause": "Pause a running agent: /pause <name>",
        "/resume": "Resume a paused agent: /resume <name>",
        "/context": "Show project context",
        "/session": "Manage sessions",
        "/clear": "Clear conversation & screen",
        "/quit": "Exit",
    }
    SESSION_SUBCMDS = ["list", "save", "load", "delete"]
    CONTEXT_SUBCMDS = ["refresh"]
    GIT_SUBCMDS = ["log", "diff", "branch"]
    PATH_COMMANDS = {"read", "edit", "list"}
    PROJECT_SUBCMDS = ["list", "switch"]

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


def save_session(name: str, conversation: list):
    data = {
        "name": name,
        "project": str(shared.PROJECT_DIR),
        "updated": datetime.now().isoformat(),
        "message_count": len([m for m in conversation if m["role"] != "system"]),
        "messages": [m for m in conversation if m["role"] != "system"],
    }
    _session_path(name).write_text(shared._redact_secrets(json.dumps(data, indent=2)))


def load_session(name: str) -> list | None:
    path = _session_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("messages", [])
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
    help_items = [
        ("/help", "Show this help"),
        ("/list [path]", "List project directory"),
        ("/read <file>", "Read file contents"),
        ("/write <file>", "Write/create file (with approval)"),
        ("/run <cmd>", "Run shell command (with approval)"),
        ("/search <query>", "Search files by name in project"),
        ("/grep <pattern> [path]", "Search inside files for text"),
        ("/git", "Git status (log, diff, branch)"),
        ("/web <query>", "Search the web (xAI live)"),
        ("/x <query>", "Search X/Twitter posts (xAI live)"),
        ("/browse <url>", "Fetch URL content (Playwright)"),
        ("/test [cmd]", "Run project tests (auto-detect framework)"),
        ("/undo", "Undo last file edit (multi-level)"),
        ("/trust", "Toggle trust mode (auto-approve)"),
        ("/readonly", "Toggle read-only mode (block writes)"),
        ("/verbose", "Toggle output detail (compact/full)"),
        ("/project <path>", "Switch project directory"),
        ("/doctor", "Check environment health"),
        ("/dashboard", "Open live TUI dashboard"),
        ("/metrics", "Show token usage and cost metrics"),
        ("/self-improve <desc>", "Improve own source code (shadow + auto-test)"),
        ("/swarm <task>", "Run multi-agent supervisor"),
        ("/abort", "Abort currently running swarm"),
        ("/tell <agent> <msg>", "Send guidance to a running agent mid-task"),
        ("/approve <agent>", "Approve an agent's plan (transition to execution)"),
        ("/reject <agent> <feedback>", "Reject plan and send feedback"),
        ("/tasks", "Show orchestrator task DAG with dependencies"),
        ("/experts", "List available experts"),
        ("/skills", "List available skills"),
        ("/context", "Show project context (refresh to rescan)"),
        ("/session", "Manage sessions (list/save/load/delete)"),
        ("/clear", "Clear conversation & screen"),
        ("/quit", "Exit"),
    ]
    for cmd, desc in help_items:
        shared.console.print(f"  [bold]{cmd:<24}[/bold] [dim]{desc}[/dim]")
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
            shared.console.print("[swarm.warning]Usage: /session load <name>[/swarm.warning]")
            return None
        msgs = load_session(subarg)
        if msgs:
            conversation.clear()
            conversation.append({"role": "system", "content": shared.SYSTEM_PROMPT})
            conversation.extend(msgs)
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
    else:
        shared.console.print("[swarm.dim]Usage: /session [list|save|load|delete] <name>[/swarm.dim]")
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
                shared.client = AsyncOpenAI(api_key=os.getenv(cfg.get("api_key_env", "XAI_API_KEY")) or shared.XAI_API_KEY, base_url=shared.BASE_URL)
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
        shared.client = AsyncOpenAI(api_key=api_key or shared.XAI_API_KEY, base_url=shared.BASE_URL)

    if ctx.invoked_subcommand is None:
        show_welcome(session)
        _update_recent_projects(shared.PROJECT_DIR)
        with shared.console.status("[bold cyan]Scanning project...[/bold cyan]", spinner="dots"):
            shared.PROJECT_CONTEXT = scan_project_context_cached(shared.PROJECT_DIR)
        shared.SYSTEM_PROMPT = build_system_prompt(shared.PROJECT_CONTEXT)
        _load_project_costs()
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

    @kb.add('escape', eager=True)
    def _handle_escape(event):
        buf = event.current_buffer
        if buf.complete_state:
            buf.cancel_completion()
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
                            erase_when_done=True)
    session.app.ttimeoutlen = 0.01
    session.app.timeoutlen = 0.01

    shared._toolbar_app_ref = session.app
    _toolbar_spinner_task = None

    async def _spinner_tick():
        while True:
            await asyncio.sleep(0.12)
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
        saved_msgs = load_session(session_name)
        if saved_msgs:
            conversation.extend(saved_msgs)
            msg_count = len(saved_msgs)
            shared.console.print(f"[swarm.accent]Resumed session '[bold]{session_name}[/bold]' ({msg_count} messages)[/swarm.accent]")
        else:
            shared.console.print(f"[swarm.accent]Started new session '[bold]{session_name}[/bold]'[/swarm.accent]")
        shared.console.print()

    processing_busy = False

    def get_bottom_toolbar():
        if shared.state.trust_mode:
            mode_str = "<ansigreen>TRUST</ansigreen> <ansidarkgray>(auto-approve)</ansidarkgray>"
        elif shared.state.read_only:
            mode_str = "<ansiyellow>READ-ONLY</ansiyellow> <ansidarkgray>(block writes)</ansidarkgray>"
        else:
            mode_str = "<ansidarkgray>NORMAL</ansidarkgray> <ansidarkgray>(require approval)</ansidarkgray>"
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
                    current = next((s for s in ag.plan if s["status"] == "in-progress"), None)
                    done = sum(1 for s in ag.plan if s["status"] == "done")
                    total = len(ag.plan)
                    if current:
                        agent_parts.append(f"{aname}: Step {done+1}/{total} \"{current['step'][:30]}\"")
                    else:
                        agent_parts.append(f"{aname}: {ag.phase.title()}...")
                else:
                    agent_parts.append(f"{aname}: {ag.phase.title()}...")
            parts.append(f"  <ansidarkgray>{len(active_agents)} agent{'s' if len(active_agents) != 1 else ''} running | {' | '.join(agent_parts)}</ansidarkgray>")
        parts.append(
            f"  <ansimagenta>\u25b6\u25b6</ansimagenta> <ansidarkgray>mode:</ansidarkgray> {mode_str}  <ansidarkgray>(shift+tab to cycle)</ansidarkgray>"
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
            try:
                conversation.append({"role": "user", "content": shared._sanitize_surrogates(user_input)})
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
                    parts = []
                    if shared._toolbar_status and not shared._is_prompt_suspended:
                        icon = shared.THINKING_FRAMES[shared._toolbar_spinner_idx % len(shared.THINKING_FRAMES)]
                        parts.append(f"  <ansicyan>{icon}</ansicyan> <ansidarkgray>{shared._toolbar_status}</ansidarkgray>")
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
                shared.console.print(f"[bold cyan]> [/bold cyan]{lines[0]}")
                for l in lines[1:]:
                    shared.console.print(f"  [dim]\u00b7 [/dim]{l}")

                # -- Slash Commands --
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0][1:].lower()
                    arg = parts[1] if len(parts) > 1 else ""

                    if processing_busy and cmd not in {"abort", "tell", "agents", "watch", "verbose", "help", "quit", "exit", "q",
                                                       "doctor", "dashboard", "metrics", "context", "experts", "skills",
                                                       "trust", "readonly", "git", "list", "read", "search", "grep",
                                                       "session", "project", "undo"}:
                        shared.console.print("[swarm.dim]Busy processing previous prompt. Wait, or use /abort.[/swarm.dim]")
                        continue

                    if cmd in ["quit", "exit", "q"]:
                        if session_name:
                            save_session(session_name, conversation)
                            shared.console.print(f"[swarm.dim]Session '{session_name}' saved.[/swarm.dim]")
                        shared.console.print("[swarm.dim]Goodbye.[/swarm.dim]")
                        break
                    elif cmd == "help":
                        _show_help()
                    elif cmd == "clear":
                        conversation = [{"role": "system", "content": shared.SYSTEM_PROMPT}]
                        os.system("cls" if os.name == "nt" else "clear")
                        show_welcome()
                    elif cmd == "context":
                        _show_context(arg)
                        if arg == "refresh":
                            shared.console.print("[swarm.dim]  refreshing context...[/swarm.dim]")
                            shared.PROJECT_CONTEXT = await asyncio.to_thread(scan_project_context, shared.PROJECT_DIR)
                            await asyncio.to_thread(_save_context_cache, shared.PROJECT_DIR, shared.PROJECT_CONTEXT)
                            shared.SYSTEM_PROMPT = build_system_prompt(shared.PROJECT_CONTEXT)
                            conversation[0] = {"role": "system", "content": shared.SYSTEM_PROMPT}
                    elif cmd == "session":
                        result = _handle_session_command(arg, conversation, session_name)
                        if result:
                            session_name = result
                    elif cmd == "list":
                        shared.console.print(list_dir(arg or "."))
                    elif cmd == "read":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /read <file>[/swarm.warning]")
                        else:
                            shared.console.print(read_file(arg))
                    elif cmd == "write":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /write <file>[/swarm.warning]")
                        else:
                            file_path = arg.split(maxsplit=1)[0]
                            shared.console.print("[swarm.dim]Enter content (type END on a new line to finish):[/swarm.dim]")
                            _lines = []
                            while True:
                                line = await session.prompt_async("  ")
                                if line.strip() == "END":
                                    break
                                _lines.append(line)
                            shared.console.print(write_file(file_path, "\n".join(_lines)))
                    elif cmd == "run":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /run <command>[/swarm.warning]")
                        else:
                            shared.console.print(run_shell(arg))
                    elif cmd == "search":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /search <query>[/swarm.warning]")
                        else:
                            shared.console.print(search_files(arg))
                    elif cmd == "grep":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /grep <pattern> [path]  (quote multi-word patterns)[/swarm.warning]")
                        else:
                            if arg.startswith('"') or arg.startswith("'"):
                                quote = arg[0]
                                end = arg.find(quote, 1)
                                if end > 0:
                                    grep_pattern = arg[1:end]
                                    rest = arg[end+1:].strip()
                                    grep_path = rest if rest else "."
                                else:
                                    grep_pattern = arg[1:]
                                    grep_path = "."
                            else:
                                grep_parts = arg.split(maxsplit=1)
                                grep_pattern = grep_parts[0]
                                grep_path = grep_parts[1] if len(grep_parts) > 1 else "."
                            shared.console.print(grep_files(grep_pattern, grep_path))
                    elif cmd == "swarm":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /swarm <task>[/swarm.warning]")
                        else:
                            await _swarm_async(arg)
                    elif cmd == "experts":
                        experts_list_cmd()
                    elif cmd == "skills":
                        skills_list_cmd()
                    elif cmd == "git":
                        if not arg:
                            shared.console.print(git_status())
                        elif arg.startswith("log"):
                            count_str = arg.split(maxsplit=1)[1] if len(arg.split()) > 1 else "10"
                            shared.console.print(git_log(int(count_str) if count_str.isdigit() else 10))
                        elif arg.startswith("diff"):
                            diff_path = arg.split(maxsplit=1)[1] if len(arg.split()) > 1 else None
                            shared.console.print(git_diff(diff_path))
                        elif arg.startswith("branch"):
                            branch_name = arg.split(maxsplit=1)[1] if len(arg.split()) > 1 else None
                            shared.console.print(git_branch(branch_name))
                        else:
                            shared.console.print("[swarm.dim]Usage: /git [log|diff|branch] [args][/swarm.dim]")
                    elif cmd == "web":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /web <query>[/swarm.warning]")
                        else:
                            shared.console.print(f"[swarm.dim]Searching web: {arg}...[/swarm.dim]")
                            shared.console.print(web_search(arg))
                    elif cmd == "x":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /x <query>[/swarm.warning]")
                        else:
                            shared.console.print(f"[swarm.dim]Searching X: {arg}...[/swarm.dim]")
                            shared.console.print(x_search(arg))
                    elif cmd == "browse":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /browse <url>[/swarm.warning]")
                        else:
                            shared.console.print(f"[swarm.dim]Fetching {arg}...[/swarm.dim]")
                            shared.console.print(fetch_page(arg))
                    elif cmd == "test":
                        shared.console.print(run_tests(arg if arg else None))
                    elif cmd == "undo":
                        if not shared.state.edit_history:
                            shared.console.print("[swarm.warning]Nothing to undo. No edit history.[/swarm.warning]")
                        else:
                            undo_path, undo_content = shared.state.edit_history[-1]
                            if undo_content is None:
                                shared.console.print(f"[bold yellow]Undo file creation:[/bold yellow] {undo_path}")
                                shared.console.print(f"[dim]Edit history depth: {len(shared.state.edit_history)}[/dim]")
                                if shared._terminal_confirm("Delete this newly created file?", default=False):
                                    fp = _safe_path(undo_path)
                                    if fp and fp.is_file():
                                        fp.unlink()
                                        shared.state.edit_history.pop()
                                        shared.console.print(f"[swarm.accent]Deleted {undo_path} (history: {len(shared.state.edit_history)} remaining)[/swarm.accent]")
                                    elif fp:
                                        shared.console.print("[swarm.error]File no longer exists.[/swarm.error]")
                                        shared.state.edit_history.pop()
                                    else:
                                        shared.console.print("[swarm.error]Cannot undo: path outside project.[/swarm.error]")
                                else:
                                    shared.console.print("[swarm.dim]Undo cancelled.[/swarm.dim]")
                            else:
                                shared.console.print(f"[bold yellow]Undo last edit to:[/bold yellow] {undo_path}")
                                shared.console.print(f"[dim]Edit history depth: {len(shared.state.edit_history)}[/dim]")
                                if shared._terminal_confirm("Restore previous content?", default=False):
                                    fp = _safe_path(undo_path)
                                    if fp:
                                        fp.write_text(undo_content, encoding="utf-8")
                                        shared.state.edit_history.pop()
                                        shared.console.print(f"[swarm.accent]Restored {undo_path} (history: {len(shared.state.edit_history)} remaining)[/swarm.accent]")
                                    else:
                                        shared.console.print("[swarm.error]Cannot restore: path outside project.[/swarm.error]")
                                else:
                                    shared.console.print("[swarm.dim]Undo cancelled.[/swarm.dim]")
                    elif cmd == "trust":
                        shared.state.trust_mode = not shared.state.trust_mode
                        trust_state = "ON" if shared.state.trust_mode else "OFF"
                        color = "bold green" if shared.state.trust_mode else "bold red"
                        shared.console.print(f"[{color}]Trust mode: {trust_state}[/{color}]")
                        if shared.state.trust_mode:
                            shared.console.print("[swarm.dim]Non-dangerous ops will be auto-approved. Shell + destructive git still gated.[/swarm.dim]")
                    elif cmd == "readonly":
                        shared.state.read_only = not shared.state.read_only
                        ro_state = "ON" if shared.state.read_only else "OFF"
                        color = "bold yellow" if shared.state.read_only else "bold green"
                        shared.console.print(f"[{color}]Read-only mode: {ro_state}[/{color}]")
                        if shared.state.read_only:
                            shared.console.print("[swarm.dim]All file-mutating tools are blocked. Use /readonly again to unlock.[/swarm.dim]")
                    elif cmd == "verbose":
                        shared.state.verbose_mode = not shared.state.verbose_mode
                        label = "full" if shared.state.verbose_mode else "compact"
                        shared.console.print(f"[swarm.accent]Output mode: [bold]{label}[/bold][/swarm.accent]")
                        shared.console.print("[dim]  compact = one-line status per tool round[/dim]")
                        shared.console.print("[dim]  full    = detailed tool names, args, timing[/dim]")
                    elif cmd == "project":
                        if not arg or arg == "list":
                            recents = _load_recent_projects()
                            if recents:
                                shared.console.print("[swarm.accent]Recent projects:[/swarm.accent]")
                                for i, rp in enumerate(recents, 1):
                                    marker = " [bold](current)[/bold]" if rp == str(shared.PROJECT_DIR.resolve()) else ""
                                    shared.console.print(f"  [bold]{i}.[/bold] {rp}{marker}")
                            else:
                                shared.console.print("[swarm.dim]No recent projects.[/swarm.dim]")
                        else:
                            target = arg.strip()
                            if target.isdigit():
                                recents = _load_recent_projects()
                                idx = int(target) - 1
                                if 0 <= idx < len(recents):
                                    target = recents[idx]
                                else:
                                    shared.console.print("[swarm.error]Invalid project number.[/swarm.error]")
                                    continue
                            if await _switch_project_async(target):
                                conversation[0] = {"role": "system", "content": shared.SYSTEM_PROMPT}
                    elif cmd == "doctor":
                        _run_doctor()
                    elif cmd == "dashboard":
                        dashboard()
                    elif cmd == "metrics":
                        metrics = get_bus().get_metrics()
                        shared.console.print()
                        shared.console.print("[swarm.accent]Session Metrics[/swarm.accent]")
                        shared.console.print(f"  [bold]Prompt Tokens:[/bold]     {metrics['prompt_tokens']:,}")
                        shared.console.print(f"  [bold]Completion Tokens:[/bold] {metrics['completion_tokens']:,}")
                        shared.console.print(f"  [bold]Total Tokens:[/bold]      {metrics['total_tokens']:,}")
                        shared.console.print()
                        shared.console.print("[swarm.accent]Project Totals (all sessions)[/swarm.accent]")
                        ptot = shared.state.project_prompt_tokens + shared.state.project_completion_tokens
                        shared.console.print(f"  [bold]Prompt Tokens:[/bold]     {shared.state.project_prompt_tokens:,}")
                        shared.console.print(f"  [bold]Completion Tokens:[/bold] {shared.state.project_completion_tokens:,}")
                        shared.console.print(f"  [bold]Total Tokens:[/bold]      {ptot:,}")
                        shared.console.print(f"  [bold]Total Cost:[/bold]        ${shared.state.project_cost_usd:.4f}")
                        shared.console.print()
                    elif cmd == "watch":
                        await _watch_agents(auto_exit=False)
                    elif cmd == "tell":
                        if not arg:
                            shared.console.print("[swarm.dim]Usage: /tell <agent_name> <message>[/swarm.dim]")
                        else:
                            tell_parts = arg.split(maxsplit=1)
                            if len(tell_parts) < 2:
                                shared.console.print("[swarm.dim]Usage: /tell <agent_name> <message>[/swarm.dim]")
                            else:
                                tell_target, tell_msg = tell_parts
                                get_bus().post("user", tell_msg, recipient=tell_target, kind="nudge")
                                shared.console.print(f"[swarm.accent]Sent nudge to '{tell_target}':[/swarm.accent] {tell_msg}")
                    elif cmd == "abort":
                        from grokswarm.commands import abort as abort_cmd
                        abort_cmd()
                    elif cmd == "clear-swarm":
                        shared.state.clear_swarm()
                        get_bus().clear()
                        shared.console.print("[bold green]Swarm state cleared:[/bold green] agents, bus messages, and background tasks reset.")
                    elif cmd == "agents":
                        shared.console.print(_list_agents_impl())
                    elif cmd == "peek":
                        if shared.state.agents:
                            targets = {}
                            if arg and arg.strip():
                                a = shared.state.get_agent(arg.strip())
                                if a is None:
                                    shared.console.print(f"[red]Agent '{arg.strip()}' not found.[/red]")
                                else:
                                    targets[arg.strip()] = a
                            else:
                                targets = dict(shared.state.agents)
                            for aname, agent in targets.items():
                                ptable = Table(title=f"[bold cyan]{aname}[/bold cyan] [dim]({agent.state.value})[/dim]", border_style="dim", width=80)
                                ptable.add_column("#", width=3, justify="right")
                                ptable.add_column("Status", width=12)
                                ptable.add_column("Step", width=58)
                                if agent.plan:
                                    _icons = {"pending": "[dim]\u25cb[/dim]", "in-progress": "[yellow]\u25b6[/yellow]", "done": "[green]\u2714[/green]", "skipped": "[dim]\u2013[/dim]"}
                                    for i, step in enumerate(agent.plan, 1):
                                        sicon = _icons.get(step["status"], " ")
                                        scolor = "green" if step["status"] == "done" else "yellow" if step["status"] == "in-progress" else "dim"
                                        ptable.add_row(str(i), f'{sicon} [{scolor}]{step["status"]}[/{scolor}]', step["step"])
                                else:
                                    ptable.add_row("", "", "[dim]No plan yet[/dim]")
                                shared.console.print(ptable)
                        else:
                            shared.console.print("[dim]No agents active.[/dim]")
                    elif cmd == "pause":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /pause <agent_name>[/swarm.warning]")
                        else:
                            agent = shared.state.get_agent(arg.strip())
                            if agent is None:
                                shared.console.print(f"[red]Agent '{arg.strip()}' not found.[/red]")
                            elif agent.state == AgentState.PAUSED:
                                shared.console.print(f"[yellow]Agent '{arg.strip()}' is already paused.[/yellow]")
                            else:
                                agent.pause_requested = True
                                agent.transition(AgentState.PAUSED)
                                shared.console.print(f"[yellow]Agent '{arg.strip()}' paused.[/yellow]")
                    elif cmd == "resume":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /resume <agent_name>[/swarm.warning]")
                        else:
                            agent = shared.state.get_agent(arg.strip())
                            if agent is None:
                                shared.console.print(f"[red]Agent '{arg.strip()}' not found.[/red]")
                            elif agent.state != AgentState.PAUSED:
                                shared.console.print(f"[yellow]Agent '{arg.strip()}' is not paused (state={agent.state.value}).[/yellow]")
                            else:
                                agent.pause_requested = False
                                agent.transition(AgentState.IDLE)
                                shared.console.print(f"[green]Agent '{arg.strip()}' resumed.[/green]")
                    elif cmd == "approve":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /approve <agent_name>[/swarm.warning]")
                        else:
                            agent = shared.state.get_agent(arg.strip())
                            if agent is None:
                                shared.console.print(f"[red]Agent '{arg.strip()}' not found.[/red]")
                            elif agent.phase != "planning":
                                shared.console.print(f"[yellow]Agent '{arg.strip()}' is not in planning phase (phase={agent.phase}).[/yellow]")
                            else:
                                PlanGate.transition_to_executing(agent)
                                get_bus().post("user", "Plan approved", recipient=arg.strip(), kind="nudge")
                                shared.console.print(f"[green]Agent '{arg.strip()}' plan approved -- now executing ({len(agent.plan)} steps).[/green]")
                    elif cmd == "reject":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /reject <agent_name> <feedback>[/swarm.warning]")
                        else:
                            reject_parts = arg.split(maxsplit=1)
                            target_name = reject_parts[0]
                            feedback = reject_parts[1] if len(reject_parts) > 1 else "Plan rejected. Revise your approach."
                            agent = shared.state.get_agent(target_name)
                            if agent is None:
                                shared.console.print(f"[red]Agent '{target_name}' not found.[/red]")
                            else:
                                agent.phase = "planning"
                                agent.approved_plan = None
                                get_bus().post("user", f"Plan rejected: {feedback}", recipient=target_name, kind="nudge")
                                shared.console.print(f"[yellow]Agent '{target_name}' plan rejected -- back to planning.[/yellow]")
                    elif cmd == "tasks":
                        dag = getattr(shared, '_current_dag', None)
                        if dag and dag.subtasks:
                            task_table = Table(title=f"[bold cyan]Task DAG[/bold cyan] -- {dag.goal[:60]}", border_style="dim", width=100)
                            task_table.add_column("ID", width=6)
                            task_table.add_column("Status", width=10)
                            task_table.add_column("Expert", width=14)
                            task_table.add_column("Description", ratio=1)
                            task_table.add_column("Depends", width=12)
                            _status_colors = {"pending": "dim", "running": "yellow", "done": "green", "failed": "red"}
                            for st in dag.subtasks:
                                color = _status_colors.get(st.status, "white")
                                deps = ", ".join(st.depends_on) if st.depends_on else "-"
                                task_table.add_row(st.id, f"[{color}]{st.status}[/{color}]", st.expert, st.description[:60], deps)
                            shared.console.print(task_table)
                        else:
                            shared.console.print("[dim]No task DAG active. Use /swarm to start one.[/dim]")
                    elif cmd == "self-improve":
                        if not arg:
                            shared.console.print("[swarm.warning]Usage: /self-improve <description of improvement>[/swarm.warning]")
                            continue
                        shadow_dir = shared.PROJECT_DIR / ".grokswarm" / "shadow"
                        shadow_dir.mkdir(parents=True, exist_ok=True)
                        shadow_file = shadow_dir / "main.py"
                        shutil.copy2(shared.PROJECT_DIR / "main.py", shadow_file)
                        shared.state.self_improve_active = True
                        shared.console.print(f"[swarm.accent]Shadow copy created:[/swarm.accent] [dim]{shadow_file.relative_to(shared.PROJECT_DIR)}[/dim]")
                        shared.console.print("[swarm.dim]The swarm will edit and test the shadow copy safely...[/swarm.dim]\n")
                        improve_prompt = f"""[SELF-IMPROVEMENT PROTOCOL]
    A shadow copy of main.py has been created at `.grokswarm/shadow/main.py`.

    TASK: {arg}

    RULES:
    1. ONLY modify `.grokswarm/shadow/main.py` using edit_file or write_file. If you need to create new files, create them in `.grokswarm/shadow/`.
    2. DO NOT edit `main.py` directly (this is mechanically blocked).
    3. After editing, verify it compiles: run_shell `python -m py_compile .grokswarm/shadow/main.py`
    4. If test_grokswarm.py exists, run tests: run_shell `python -m pytest test_grokswarm.py -v`
    5. When verified, stop and summarize your changes. I will handle promotion.

    CRITICAL: If you extract code into a new file, you MUST remove that code from `.grokswarm/shadow/main.py` and update the imports in `.grokswarm/shadow/main.py` to use the new file.
    CRITICAL: If you create a new file, you MUST use the `write_file` tool and provide the full path to the new file in the `.grokswarm/shadow/` directory."""
                        conversation.append({"role": "user", "content": improve_prompt})
                        conversation = await _trim_conversation(conversation)
                        full_response = await _stream_with_tools(conversation)
                        if session_name:
                            save_session(session_name, conversation)
                        shared.state.self_improve_active = False
                        shared.console.print("\n[bold yellow]Self-Improvement Complete.[/bold yellow]")
                        shared.console.print(f"[dim]Review changes: run_shell 'python -c \"import difflib,pathlib; a=pathlib.Path(\\\"main.py\\\").read_text().splitlines(); b=pathlib.Path(\\\".grokswarm/shadow/main.py\\\").read_text().splitlines(); print(chr(10).join(difflib.unified_diff(a,b,lineterm=\\\"\\\",n=3)))'[/dim]")
                        if shared._terminal_confirm("Promote shadow copy to main.py?", default=False):
                            check = subprocess.run(["python", "-m", "py_compile", str(shadow_file)], capture_output=True, text=True)
                            if check.returncode != 0:
                                shared.console.print("[bold red]Shadow copy has syntax errors \u2014 promotion blocked.[/bold red]")
                                shared.console.print(f"[dim]{check.stderr[:300]}[/dim]")
                            else:
                                test_file = shared.PROJECT_DIR / "test_grokswarm.py"
                                if test_file.exists():
                                    shared.console.print("[swarm.dim]Running test suite against shadow copy (isolated)...[/swarm.dim]")
                                    import tempfile as _tf
                                    with _tf.TemporaryDirectory(prefix="grokswarm_test_") as iso_dir:
                                        iso = Path(iso_dir)
                                        shutil.copy2(shadow_file, iso / "main.py")
                                        for f in shadow_dir.glob("*.py"):
                                            if f.name != "main.py":
                                                shutil.copy2(f, iso / f.name)
                                        shutil.copy2(test_file, iso / "test_grokswarm.py")
                                        for f in shared.PROJECT_DIR.glob("conftest*.py"):
                                            shutil.copy2(f, iso / f.name)
                                        test_check = subprocess.run(
                                            [sys.executable, "-m", "pytest", "test_grokswarm.py", "-x", "-q"],
                                            capture_output=True, text=True, cwd=str(iso), timeout=120
                                        )
                                    if test_check.returncode != 0:
                                        shared.console.print("[bold red]Tests failed \u2014 promotion blocked.[/bold red]")
                                        shared.console.print(f"[dim]{test_check.stdout[-500:] if test_check.stdout else test_check.stderr[:300]}[/dim]")
                                        shared.console.print("[swarm.dim]Shadow copy preserved. Fix tests before promoting.[/swarm.dim]")
                                    else:
                                        shared.console.print("[swarm.accent]Tests passed (isolated)![/swarm.accent]")
                                        shutil.copy2(shadow_file, shared.PROJECT_DIR / "main.py")
                                        for f in shadow_dir.glob("*.py"):
                                            if f.name != "main.py":
                                                shutil.copy2(f, shared.PROJECT_DIR / f.name)
                                        shared.console.print("[swarm.accent]main.py updated! Restart to load new features.[/swarm.accent]")
                                else:
                                    shutil.copy2(shadow_file, shared.PROJECT_DIR / "main.py")
                                    for f in shadow_dir.glob("*.py"):
                                        if f.name != "main.py":
                                            shutil.copy2(f, shared.PROJECT_DIR / f.name)
                                    shared.console.print("[swarm.accent]main.py updated! Restart to load new features.[/swarm.accent]")
                        else:
                            shared.console.print("[swarm.dim]Shadow copy preserved at .grokswarm/shadow/main.py[/swarm.dim]")
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
            shared._clear_status()
            if _toolbar_spinner_task:
                _toolbar_spinner_task.cancel()
            processor_task.cancel()


def experts_list_cmd():
    """List experts (for /experts slash command)."""
    table = Table(title="Expert Registry")
    table.add_column("Expert")
    for e in list_experts():
        table.add_row(e)
    shared.console.print(table)


def skills_list_cmd():
    """List skills (for /skills slash command)."""
    table = Table(title="Skill Registry")
    table.add_column("Skill")
    for s in list_skills():
        table.add_row(s)
    shared.console.print(table)
