"""Extracted slash-command handlers for the REPL.

Each handler is an async function with signature:
    async def handle_X(arg: str, ctx: CmdContext) -> None
"""

import os
import sys
import json
import shutil
import asyncio
import difflib
import subprocess
from pathlib import Path

import grokswarm.shared as shared
from grokswarm.cmd_dispatch import CmdContext
from grokswarm.tools_fs import list_dir, read_file, write_file, search_files, grep_files
from grokswarm.tools_shell import run_shell
from grokswarm.tools_test import run_tests
from grokswarm.tools_git import git_status, git_diff, git_log, git_branch
from grokswarm.tools_search import web_search, x_search
from grokswarm.tools_browser import fetch_page
from grokswarm.registry_helpers import list_experts, list_skills, list_memory, prune_memory
from grokswarm.agents import get_bus, _list_agents_impl
from grokswarm.guardrails import (
    PlanGate, drain_notifications,
    get_model_tiers, set_model_tier, reset_model_tiers,
)
from grokswarm.models import AgentState
from grokswarm.context import scan_project_context, _save_context_cache, build_system_prompt, _safe_path


# -- Simple handlers --

async def handle_quit(arg: str, ctx: CmdContext) -> None:
    if ctx.session_name:
        ctx.save_session(ctx.session_name, ctx.conversation)
        shared.console.print(f"[swarm.dim]Session '{ctx.session_name}' saved.[/swarm.dim]")
    shared.console.print("[swarm.dim]Goodbye.[/swarm.dim]")
    ctx.quit_flag = True


async def handle_help(arg: str, ctx: CmdContext) -> None:
    from grokswarm.cmd_dispatch import all_commands
    shared.console.print()
    shared.console.print("[swarm.accent]Commands[/swarm.accent]")
    for entry in all_commands().values():
        name = f"/{entry.name}"
        aliases = ""
        if entry.aliases:
            aliases = f" (/{'/'.join(entry.aliases)})"
        shared.console.print(f"  [bold]{name + aliases:<24}[/bold] [dim]{entry.description}[/dim]")
    shared.console.print()


async def handle_clear(arg: str, ctx: CmdContext) -> None:
    from grokswarm.repl import show_welcome
    ctx.conversation.clear()
    ctx.conversation.append({"role": "system", "content": shared.SYSTEM_PROMPT})
    os.system("cls" if os.name == "nt" else "clear")
    show_welcome()


async def handle_context(arg: str, ctx: CmdContext) -> None:
    from grokswarm.repl import _show_context
    _show_context(arg)
    if arg == "refresh":
        shared.console.print("[swarm.dim]  refreshing context...[/swarm.dim]")
        shared.PROJECT_CONTEXT = await asyncio.to_thread(scan_project_context, shared.PROJECT_DIR)
        await asyncio.to_thread(_save_context_cache, shared.PROJECT_DIR, shared.PROJECT_CONTEXT)
        shared.SYSTEM_PROMPT = build_system_prompt(shared.PROJECT_CONTEXT)
        ctx.conversation[0] = {"role": "system", "content": shared.SYSTEM_PROMPT}


async def handle_session(arg: str, ctx: CmdContext) -> None:
    from grokswarm.repl import _handle_session_command
    result = _handle_session_command(arg, ctx.conversation, ctx.session_name)
    if result:
        ctx.new_session_name = result


async def handle_list(arg: str, ctx: CmdContext) -> None:
    shared.console.print(list_dir(arg or "."))


async def handle_read(arg: str, ctx: CmdContext) -> None:
    if not arg:
        shared.console.print("[swarm.warning]Usage: /read <file>[/swarm.warning]")
    else:
        shared.console.print(read_file(arg))


async def handle_write(arg: str, ctx: CmdContext) -> None:
    if not arg:
        shared.console.print("[swarm.warning]Usage: /write <file>[/swarm.warning]")
    else:
        file_path = arg.split(maxsplit=1)[0]
        shared.console.print("[swarm.dim]Enter content (type END on a new line to finish):[/swarm.dim]")
        _lines = []
        while True:
            line = await ctx.session.prompt_async("  ")
            if line.strip() == "END":
                break
            _lines.append(line)
        shared.console.print(write_file(file_path, "\n".join(_lines)))


async def handle_run(arg: str, ctx: CmdContext) -> None:
    if not arg:
        shared.console.print("[swarm.warning]Usage: /run <command>[/swarm.warning]")
    else:
        shared.console.print(run_shell(arg))


async def handle_search(arg: str, ctx: CmdContext) -> None:
    if not arg:
        shared.console.print("[swarm.warning]Usage: /search <query>[/swarm.warning]")
    else:
        shared.console.print(search_files(arg))


async def handle_grep(arg: str, ctx: CmdContext) -> None:
    if not arg:
        shared.console.print("[swarm.warning]Usage: /grep <pattern> [path]  (quote multi-word patterns)[/swarm.warning]")
        return
    if arg.startswith('"') or arg.startswith("'"):
        quote = arg[0]
        end = arg.find(quote, 1)
        if end > 0:
            grep_pattern = arg[1:end]
            rest = arg[end + 1:].strip()
            grep_path = rest if rest else "."
        else:
            grep_pattern = arg[1:]
            grep_path = "."
    else:
        grep_parts = arg.split(maxsplit=1)
        grep_pattern = grep_parts[0]
        grep_path = grep_parts[1] if len(grep_parts) > 1 else "."
    shared.console.print(grep_files(grep_pattern, grep_path))


async def handle_swarm(arg: str, ctx: CmdContext) -> None:
    if not arg:
        shared.console.print("[swarm.warning]Usage: /swarm <task>[/swarm.warning]")
    else:
        from grokswarm.repl import _swarm_async
        await _swarm_async(arg)


async def handle_experts(arg: str, ctx: CmdContext) -> None:
    from rich.table import Table
    table = Table(title="Expert Registry")
    table.add_column("Expert")
    for e in list_experts():
        table.add_row(e)
    shared.console.print(table)


async def handle_skills(arg: str, ctx: CmdContext) -> None:
    from rich.table import Table
    table = Table(title="Skill Registry")
    table.add_column("Skill")
    for s in list_skills():
        table.add_row(s)
    shared.console.print(table)


async def handle_git(arg: str, ctx: CmdContext) -> None:
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


async def handle_web(arg: str, ctx: CmdContext) -> None:
    if not arg:
        shared.console.print("[swarm.warning]Usage: /web <query>[/swarm.warning]")
    else:
        shared.console.print(f"[swarm.dim]Searching web: {arg}...[/swarm.dim]")
        shared.console.print(web_search(arg))


async def handle_x(arg: str, ctx: CmdContext) -> None:
    if not arg:
        shared.console.print("[swarm.warning]Usage: /x <query>[/swarm.warning]")
    else:
        shared.console.print(f"[swarm.dim]Searching X: {arg}...[/swarm.dim]")
        shared.console.print(x_search(arg))


async def handle_browse(arg: str, ctx: CmdContext) -> None:
    if not arg:
        shared.console.print("[swarm.warning]Usage: /browse <url>[/swarm.warning]")
    else:
        shared.console.print(f"[swarm.dim]Fetching {arg}...[/swarm.dim]")
        shared.console.print(fetch_page(arg))


async def handle_test(arg: str, ctx: CmdContext) -> None:
    shared.console.print(run_tests(arg if arg else None))


async def handle_undo(arg: str, ctx: CmdContext) -> None:
    if not shared.state.edit_history:
        shared.console.print("[swarm.warning]Nothing to undo. No edit history.[/swarm.warning]")
        return
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
        # Show inline diff preview
        fp_preview = _safe_path(undo_path)
        if fp_preview and fp_preview.is_file():
            try:
                current_text = fp_preview.read_text(encoding="utf-8")
                diff_lines = list(difflib.unified_diff(
                    current_text.splitlines(), undo_content.splitlines(),
                    fromfile="current", tofile="previous", lineterm=""))
                if diff_lines:
                    from rich.syntax import Syntax
                    display_lines = diff_lines[:50]
                    extra = len(diff_lines) - 50
                    diff_text = "\n".join(display_lines)
                    if extra > 0:
                        diff_text += f"\n... (+{extra} more lines)"
                    shared.console.print(Syntax(diff_text, "diff", theme="monokai"))
            except Exception:
                pass
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


async def handle_trust(arg: str, ctx: CmdContext) -> None:
    shared.state.trust_mode = not shared.state.trust_mode
    trust_state = "ON" if shared.state.trust_mode else "OFF"
    color = "bold green" if shared.state.trust_mode else "bold red"
    shared.console.print(f"[{color}]Trust mode: {trust_state}[/{color}]")
    if shared.state.trust_mode:
        shared.console.print("[swarm.dim]Non-dangerous ops will be auto-approved. Shell + destructive git still gated.[/swarm.dim]")


async def handle_readonly(arg: str, ctx: CmdContext) -> None:
    shared.state.read_only = not shared.state.read_only
    ro_state = "ON" if shared.state.read_only else "OFF"
    color = "bold yellow" if shared.state.read_only else "bold green"
    shared.console.print(f"[{color}]Read-only mode: {ro_state}[/{color}]")
    if shared.state.read_only:
        shared.console.print("[swarm.dim]All file-mutating tools are blocked. Use /readonly again to unlock.[/swarm.dim]")


async def handle_verbose(arg: str, ctx: CmdContext) -> None:
    shared.state.verbose_mode = not shared.state.verbose_mode
    label = "full" if shared.state.verbose_mode else "compact"
    shared.console.print(f"[swarm.accent]Output mode: [bold]{label}[/bold][/swarm.accent]")
    shared.console.print("[dim]  compact = one-line status per tool round[/dim]")
    shared.console.print("[dim]  full    = detailed tool names, args, timing[/dim]")


async def handle_project(arg: str, ctx: CmdContext) -> None:
    from grokswarm.repl import _load_recent_projects, _switch_project_async
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
                return
        if await _switch_project_async(target):
            ctx.conversation[0] = {"role": "system", "content": shared.SYSTEM_PROMPT}


async def handle_doctor(arg: str, ctx: CmdContext) -> None:
    from grokswarm.repl import _run_doctor
    _run_doctor()


async def handle_dashboard(arg: str, ctx: CmdContext) -> None:
    from grokswarm.dashboard import dashboard
    dashboard()


async def handle_metrics(arg: str, ctx: CmdContext) -> None:
    metrics = get_bus().get_metrics()
    shared.console.print()
    shared.console.print("[swarm.accent]Session Metrics[/swarm.accent]")
    shared.console.print(f"  [bold]Prompt Tokens:[/bold]     {metrics['prompt_tokens']:,}")
    shared.console.print(f"  [bold]Completion Tokens:[/bold] {metrics['completion_tokens']:,}")
    shared.console.print(f"  [bold]Cached Tokens:[/bold]     {metrics.get('cached_tokens', 0):,}")
    shared.console.print(f"  [bold]Total Tokens:[/bold]      {metrics['total_tokens']:,}")
    s_cached = metrics.get('cached_tokens', 0)
    s_prompt = metrics['prompt_tokens']
    if s_prompt > 0 and s_cached > 0:
        cache_pct = (s_cached / s_prompt) * 100
        shared.console.print(f"  [bold]Cache Hit Rate:[/bold]   {cache_pct:.1f}%")
    shared.console.print()
    shared.console.print("[swarm.accent]Project Totals (all sessions)[/swarm.accent]")
    ptot = shared.state.project_prompt_tokens + shared.state.project_completion_tokens
    shared.console.print(f"  [bold]Prompt Tokens:[/bold]     {shared.state.project_prompt_tokens:,}")
    shared.console.print(f"  [bold]Completion Tokens:[/bold] {shared.state.project_completion_tokens:,}")
    shared.console.print(f"  [bold]Cached Tokens:[/bold]     {shared.state.project_cached_tokens:,}")
    shared.console.print(f"  [bold]Total Tokens:[/bold]      {ptot:,}")
    shared.console.print(f"  [bold]Total Cost:[/bold]        ${shared.state.project_cost_usd:.4f}")
    p_cached = shared.state.project_cached_tokens
    if p_cached > 0:
        savings = (p_cached / 1_000_000.0) * (0.20 - 0.05)
        shared.console.print(f"  [bold green]Cache Savings:[/bold green]   ~${savings:.4f}")
        cache_pct = (p_cached / shared.state.project_prompt_tokens) * 100 if shared.state.project_prompt_tokens > 0 else 0
        shared.console.print(f"  [bold]Cache Hit Rate:[/bold]   {cache_pct:.1f}%")
    shared.console.print()


async def handle_watch(arg: str, ctx: CmdContext) -> None:
    from grokswarm.dashboard import _watch_agents
    await _watch_agents(auto_exit=False)


async def handle_tell(arg: str, ctx: CmdContext) -> None:
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


async def handle_abort(arg: str, ctx: CmdContext) -> None:
    from grokswarm.commands import abort as abort_cmd
    abort_cmd()


async def handle_clear_swarm(arg: str, ctx: CmdContext) -> None:
    shared.state.clear_swarm()
    get_bus().clear()
    shared.console.print("[bold green]Swarm state cleared:[/bold green] agents, bus messages, and background tasks reset.")


async def handle_agents(arg: str, ctx: CmdContext) -> None:
    shared.console.print(_list_agents_impl())


async def handle_peek(arg: str, ctx: CmdContext) -> None:
    from rich.table import Table
    if shared.state.agents:
        targets = {}
        if arg and arg.strip():
            a = shared.state.get_agent(arg.strip())
            if a is None:
                shared.console.print(f"[red]Agent '{arg.strip()}' not found.[/red]")
                return
            targets[arg.strip()] = a
        else:
            targets = dict(shared.state.agents)
        for aname, agent in targets.items():
            model_short = (agent.current_model or "?").split("-")[-1][:20]
            cache_pct = f"{(agent.cached_tokens_total / max(agent.tokens_used, 1)) * 100:.0f}%" if agent.tokens_used else "n/a"
            shared.console.print(f"\n[bold cyan]{aname}[/bold cyan] [dim]({agent.state.value})[/dim]  "
                                 f"phase={agent.phase}  model={model_short}  "
                                 f"cost=${agent.cost_usd:.4f}  tokens={agent.tokens_used:,}  cache={cache_pct}")
            ptable = Table(border_style="dim", width=80, show_header=True)
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
            if agent.tool_call_log:
                shared.console.print("[dim]Recent tool calls:[/dim]")
                for entry in agent.tool_call_log[-5:]:
                    result_short = entry['result'][:80]
                    shared.console.print(f"  [dim]R{entry['round']}[/dim] [bold]{entry['tool']}[/bold] {entry['args']}  [dim]{result_short}[/dim]")
    else:
        shared.console.print("[dim]No agents active.[/dim]")


async def handle_pause(arg: str, ctx: CmdContext) -> None:
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


async def handle_resume(arg: str, ctx: CmdContext) -> None:
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


async def handle_approve(arg: str, ctx: CmdContext) -> None:
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


async def handle_reject(arg: str, ctx: CmdContext) -> None:
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


async def handle_tasks(arg: str, ctx: CmdContext) -> None:
    from rich.table import Table
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


async def handle_budget(arg: str, ctx: CmdContext) -> None:
    from grokswarm.guardrails import _cost_guard
    if arg and arg.strip():
        try:
            budget_val = float(arg.strip().lstrip("$"))
            _cost_guard.set_budget(budget_val)
            shared.state.session_cost_budget_usd = budget_val
            shared.console.print(f"[swarm.accent]Session budget set to ${budget_val:.2f}[/swarm.accent]")
        except ValueError:
            shared.console.print("[swarm.warning]Usage: /budget <amount> (e.g., /budget 5.00)[/swarm.warning]")
    else:
        current = _cost_guard.session_budget_usd
        rate = _cost_guard.get_rate_per_min()
        shared.console.print(f"  [bold]Session budget:[/bold]  {'$' + f'{current:.2f}' if current > 0 else 'unlimited'}")
        shared.console.print(f"  [bold]Session spent:[/bold]   ${shared.state.global_cost_usd:.4f}")
        shared.console.print(f"  [bold]Spending rate:[/bold]   ${rate:.4f}/min")
        if current > 0:
            remaining = max(0, current - shared.state.global_cost_usd)
            shared.console.print(f"  [bold]Remaining:[/bold]       ${remaining:.4f}")


async def handle_model(arg: str, ctx: CmdContext) -> None:
    from rich.table import Table
    model_parts = arg.split() if arg else []
    if not model_parts or model_parts[0] == "list":
        tiers = get_model_tiers()
        tbl = Table(title="Model Tiers", show_header=True, header_style="bold")
        tbl.add_column("Tier", style="bold cyan")
        tbl.add_column("Model")
        tbl.add_column("Input/1M", justify="right")
        tbl.add_column("Cached/1M", justify="right")
        tbl.add_column("Output/1M", justify="right")
        for tier, model_name in tiers.items():
            inp, cached, out = shared._get_pricing(model_name)
            tbl.add_row(tier, model_name, f"${inp:.2f}", f"${cached:.2f}", f"${out:.2f}")
        shared.console.print()
        shared.console.print(tbl)
        shared.console.print()
    elif model_parts[0] == "reset":
        reset_model_tiers()
        shared.console.print("[swarm.accent]Model tiers restored to defaults.[/swarm.accent]")
    elif len(model_parts) == 2:
        m_tier, m_name = model_parts
        try:
            set_model_tier(m_tier, m_name)
            shared.console.print(f"[swarm.accent]Tier '[bold]{m_tier}[/bold]' \u2192 {m_name}[/swarm.accent]")
        except ValueError as e:
            shared.console.print(f"[swarm.warning]{e}[/swarm.warning]")
    else:
        shared.console.print("[swarm.dim]Usage: /model [list|reset|<tier> <model-name>][/swarm.dim]")


async def handle_bugs(arg: str, ctx: CmdContext) -> None:
    from rich.table import Table
    from grokswarm.bugs import get_self_tracker, get_project_tracker
    bug_parts = arg.split() if arg else []
    scope = "project"
    if bug_parts and bug_parts[0] in ("self", "project"):
        scope = bug_parts.pop(0)
    tracker = get_self_tracker() if scope == "self" else get_project_tracker()
    action = bug_parts[0] if bug_parts else "list"
    if action == "list" or not bug_parts:
        bugs = tracker.list()
        if not bugs:
            shared.console.print(f"[swarm.dim]No bugs in {scope} tracker.[/swarm.dim]")
        else:
            tbl = Table(title=f"{scope.title()} Bugs ({len(bugs)})", show_header=True, header_style="bold")
            tbl.add_column("#", style="bold", width=4)
            tbl.add_column("Sev", width=8)
            tbl.add_column("Status", width=11)
            tbl.add_column("Title")
            tbl.add_column("Source", width=6)
            for b in bugs:
                sev_color = {"critical": "red", "high": "yellow", "medium": "cyan", "low": "dim"}.get(b.severity, "")
                tbl.add_row(str(b.id), f"[{sev_color}]{b.severity}[/{sev_color}]", b.status, b.title, b.source)
            shared.console.print()
            shared.console.print(tbl)
            shared.console.print()
    elif action == "add":
        title = " ".join(bug_parts[1:]) if len(bug_parts) > 1 else ""
        if not title:
            shared.console.print("[swarm.dim]Usage: /bugs [self|project] add <title>[/swarm.dim]")
        else:
            bug = tracker.log(title, "", "medium", "user")
            shared.console.print(f"[swarm.accent]Bug #{bug.id} logged: {title}[/swarm.accent]")
    elif action == "fix" and len(bug_parts) >= 2:
        try:
            bid = int(bug_parts[1])
            updated = tracker.update(bid, status="fixed")
            if updated:
                shared.console.print(f"[swarm.accent]Bug #{bid} marked as fixed.[/swarm.accent]")
            else:
                shared.console.print(f"[swarm.warning]Bug #{bid} not found.[/swarm.warning]")
        except ValueError:
            shared.console.print("[swarm.dim]Usage: /bugs fix <id>[/swarm.dim]")
    elif action == "show" and len(bug_parts) >= 2:
        try:
            bid = int(bug_parts[1])
            bug = tracker.get(bid)
            if bug:
                shared.console.print(f"\n[bold]Bug #{bug.id}[/bold] [{bug.severity}] [{bug.status}]")
                shared.console.print(f"[bold]Title:[/bold] {bug.title}")
                shared.console.print(f"[bold]Source:[/bold] {bug.source}")
                shared.console.print(f"[bold]Created:[/bold] {bug.created}")
                if bug.description:
                    shared.console.print(f"[bold]Description:[/bold]\n{bug.description[:500]}")
                if bug.context:
                    shared.console.print(f"[bold]Context:[/bold] {json.dumps(bug.context, indent=2)[:300]}")
                shared.console.print()
            else:
                shared.console.print(f"[swarm.warning]Bug #{bid} not found.[/swarm.warning]")
        except ValueError:
            shared.console.print("[swarm.dim]Usage: /bugs show <id>[/swarm.dim]")
    else:
        shared.console.print("[swarm.dim]Usage: /bugs [self|project] [list|add <title>|show <id>|fix <id>][/swarm.dim]")


async def handle_memory(arg: str, ctx: CmdContext) -> None:
    from rich.table import Table
    mem_parts = arg.split() if arg else []
    action = mem_parts[0] if mem_parts else "list"
    if action == "list" or not mem_parts:
        entries = list_memory()
        if not entries:
            shared.console.print("[swarm.dim]No memory files.[/swarm.dim]")
        else:
            tbl = Table(title=f"Agent Memory ({len(entries)} files)")
            tbl.add_column("Key", ratio=1)
            tbl.add_column("Timestamp", width=20)
            tbl.add_column("Size", width=8, justify="right")
            for e in entries:
                ts = e["timestamp"][:19].replace("T", " ") if e["timestamp"] != "?" else "?"
                tbl.add_row(e["key"], ts, f"{e['size']:,}")
            shared.console.print()
            shared.console.print(tbl)
            shared.console.print()
    elif action == "prune":
        days = 30
        if len(mem_parts) > 1 and mem_parts[1].isdigit():
            days = int(mem_parts[1])
        deleted = prune_memory(days)
        shared.console.print(f"[swarm.accent]Pruned {deleted} memory file(s) older than {days} days.[/swarm.accent]")
    else:
        shared.console.print("[swarm.dim]Usage: /memory [list|prune [days]][/swarm.dim]")


async def handle_eval(arg: str, ctx: CmdContext) -> None:
    eval_script = shared.PROJECT_DIR / "eval_grokswarm.py"
    if not eval_script.exists():
        shared.console.print("[swarm.warning]eval_grokswarm.py not found in project directory.[/swarm.warning]")
    else:
        eval_args = [sys.executable, str(eval_script)]
        if arg:
            eval_args.extend(arg.split())
        else:
            eval_args.append("--live")
        shared.console.print(f"[swarm.dim]Running: {' '.join(eval_args)}[/swarm.dim]")
        proc = await asyncio.to_thread(
            subprocess.run, eval_args,
            capture_output=True, text=True, timeout=600,
            cwd=str(shared.PROJECT_DIR),
        )
        if proc.stdout:
            shared.console.print(proc.stdout[-3000:])
        if proc.returncode != 0 and proc.stderr:
            shared.console.print(f"[swarm.error]{proc.stderr[-1000:]}[/swarm.error]")


async def handle_self_improve(arg: str, ctx: CmdContext) -> None:
    from grokswarm.engine import _stream_with_tools, _trim_conversation
    from grokswarm.repl import save_session

    if not arg:
        shared.console.print("[swarm.warning]Usage: /self-improve <description of improvement>[/swarm.warning]")
        return
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
    ctx.conversation.append({"role": "user", "content": improve_prompt})
    ctx.conversation[:] = await _trim_conversation(ctx.conversation)
    await _stream_with_tools(ctx.conversation)
    if ctx.session_name:
        save_session(ctx.session_name, ctx.conversation)
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


async def handle_self_eval(arg: str, ctx: CmdContext) -> None:
    from grokswarm.self_eval import run_self_eval_loop
    parts = arg.split() if arg else []
    category = parts[0] if parts else "all"
    max_rounds = 3
    if len(parts) >= 2 and parts[1].isdigit():
        max_rounds = int(parts[1])
    result = await run_self_eval_loop(category=category, max_rounds=max_rounds)
    shared.console.print(result)


async def handle_history(arg: str, ctx: CmdContext) -> None:
    from rich.table import Table
    history_file = Path("~/.grokswarm/history.txt").expanduser()
    if not history_file.exists():
        shared.console.print("[swarm.dim]No history file found.[/swarm.dim]")
        return
    # Parse prompt_toolkit history format: lines starting with '+' are content
    entries: list[str] = []
    current: list[str] = []
    for line in history_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("+"):
            current.append(line[1:])
        else:
            if current:
                entries.append("\n".join(current))
                current = []
    if current:
        entries.append("\n".join(current))
    query = arg.strip().lower() if arg else ""
    if query:
        entries = [e for e in entries if query in e.lower()]
    shown = entries[-25:]
    if not shown:
        shared.console.print(f"[swarm.dim]No history entries{'matching ' + repr(arg.strip()) if query else ''}.[/swarm.dim]")
        return
    tbl = Table(title=f"History ({len(shown)}/{len(entries)})", show_header=True, border_style="dim")
    tbl.add_column("#", width=5, justify="right")
    tbl.add_column("Input", ratio=1)
    offset = len(entries) - len(shown)
    for i, entry in enumerate(shown, offset + 1):
        display = entry[:120] + ("..." if len(entry) > 120 else "")
        tbl.add_row(str(i), display)
    shared.console.print()
    shared.console.print(tbl)
    shared.console.print()
    # Fuzzy replay picker
    if shown:
        lookup = {}
        for entry in shown:
            key = entry[:80].replace("\n", " ").strip()
            if key:
                lookup[key] = entry
        if lookup:
            try:
                from prompt_toolkit import prompt as pt_prompt
                from prompt_toolkit.completion import FuzzyWordCompleter
                pick = pt_prompt("replay> ", completer=FuzzyWordCompleter(list(lookup.keys()))).strip()
            except (EOFError, KeyboardInterrupt):
                return
            if pick and pick in lookup:
                shared._saved_prompt_text = lookup[pick]
                shared.console.print("[swarm.dim]Selected entry will appear in next prompt.[/swarm.dim]")
            elif pick:
                # Try exact match against full entries
                for entry in shown:
                    if pick in entry:
                        shared._saved_prompt_text = entry
                        shared.console.print("[swarm.dim]Selected entry will appear in next prompt.[/swarm.dim]")
                        break


async def handle_vim(arg: str, ctx: CmdContext) -> None:
    from prompt_toolkit.enums import EditingMode
    shared.state.vi_mode = not shared.state.vi_mode
    if shared.state.vi_mode:
        ctx.session.editing_mode = EditingMode.VI
        shared.console.print("[swarm.accent]Vi mode [bold]ON[/bold] — ESC for normal mode, i for insert[/swarm.accent]")
    else:
        ctx.session.editing_mode = EditingMode.EMACS
        shared.console.print("[swarm.accent]Vi mode [bold]OFF[/bold] — standard editing restored[/swarm.accent]")


async def handle_claude(arg: str, ctx: CmdContext) -> None:
    import shutil
    subcmd = arg.strip().lower() if arg else ""

    if subcmd == "dualhead":
        if not shutil.which("claude"):
            shared.console.print("[swarm.error]Claude Code CLI not found in PATH.[/swarm.error]")
            return
        shared.state.dualhead_mode = not shared.state.dualhead_mode
        status = "ON" if shared.state.dualhead_mode else "OFF"
        shared.console.print(
            f"[magenta]Dualhead mode [bold]{status}[/bold] — "
            f"{'Grok plans, Claude reviews before execution' if shared.state.dualhead_mode else 'deliberation disabled'}[/magenta]"
        )
        return

    # Original toggle behavior (no subcommand)
    if not shared.state.claude_mode and not shutil.which("claude"):
        shared.console.print("[swarm.error]Claude Code CLI not found in PATH. Install it first: https://docs.anthropic.com/en/docs/claude-code[/swarm.error]")
        return
    shared.state.claude_mode = not shared.state.claude_mode
    status = "ON" if shared.state.claude_mode else "OFF"
    shared.console.print(f"[magenta]Claude Code mode [bold]{status}[/bold] — {'experts route through claude -p' if shared.state.claude_mode else 'back to native GrokSwarm'}[/magenta]")


async def handle_daemon(arg: str, ctx: CmdContext) -> None:
    from grokswarm.daemon import start_daemon, stop_daemon, daemon_status, daemon_log, add_watch_pattern
    parts = arg.split() if arg else []
    subcmd = parts[0] if parts else "status"
    if subcmd == "start":
        result = await start_daemon()
        shared.console.print(f"[swarm.accent]{result}[/swarm.accent]")
    elif subcmd == "stop":
        result = await stop_daemon()
        shared.console.print(f"[swarm.accent]{result}[/swarm.accent]")
    elif subcmd == "status":
        shared.console.print(f"[swarm.accent]Daemon Status[/swarm.accent]")
        shared.console.print(daemon_status())
    elif subcmd == "log":
        shared.console.print(daemon_log(20))
    elif subcmd == "add" and len(parts) >= 2:
        result = add_watch_pattern(parts[1])
        shared.console.print(f"[swarm.accent]{result}[/swarm.accent]")
    else:
        shared.console.print("[swarm.dim]Usage: /daemon [start|stop|status|log|add <pattern>][/swarm.dim]")


async def handle_self_scores(arg: str, ctx: CmdContext) -> None:
    from rich.table import Table
    from pathlib import Path as _Path

    scores_path = _Path(".grokswarm") / "eval_scores.json"
    scores: dict = {}
    if scores_path.exists():
        try:
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Build task catalog from all eval modules
    all_tasks: dict[str, dict] = {}
    try:
        from eval_grokswarm import EVAL_TASKS as BASIC_TASKS
        for t in BASIC_TASKS:
            all_tasks[t.id] = {"category": t.category, "description": t.description}
    except ImportError:
        pass
    try:
        from eval_deep import DEEP_EVAL_TASKS
        for t in DEEP_EVAL_TASKS:
            all_tasks[t.id] = {"category": t.category, "description": t.description}
    except ImportError:
        pass
    try:
        from eval_deep_v2 import V2_EVAL_TASKS
        for t in V2_EVAL_TASKS:
            all_tasks[t.id] = {"category": t.category, "description": t.description}
    except ImportError:
        pass

    # Include any tasks in scores that aren't in the catalog
    for tid, data in scores.items():
        if tid not in all_tasks:
            all_tasks[tid] = {"category": data.get("category", "?"), "description": data.get("description", "")}

    task_id = arg.strip() if arg else ""

    if task_id:
        # Detailed view for a single task
        if task_id not in scores:
            info = all_tasks.get(task_id)
            if info:
                shared.console.print(f"[swarm.dim]Task {task_id} ({info['description']}) — not yet run.[/swarm.dim]")
            else:
                shared.console.print(f"[swarm.warning]Unknown task: {task_id}[/swarm.warning]")
            return
        d = scores[task_id]
        s_over = d.get("single_overall", 0) or 0
        w_over = d.get("swarm_overall", 0) or 0
        has_swarm = d.get("swarm_quality", 0) > 0 or d.get("swarm_cost_score", 0) > 0
        if not has_swarm:
            verdict = "Single Only"
        elif s_over - w_over > 0.05:
            verdict = "Single Better"
        elif w_over - s_over > 0.05:
            verdict = "Swarm Better"
        else:
            verdict = "Tie"
        shared.console.print(f"\n[bold cyan]{task_id}[/bold cyan] — {d.get('description', '')}")
        shared.console.print(f"  [bold]Category:[/bold]  {d.get('category', '?')}")
        shared.console.print(f"  [bold]Verdict:[/bold]   {verdict} (S.Overall={s_over:.0%} W.Overall={w_over:.0%})")
        shared.console.print(f"  [bold]Updated:[/bold]   {d.get('updated', '?')}")
        shared.console.print()
        shared.console.print(f"  [bold]Single:[/bold]  quality={d.get('single_quality', 0):.0%}  "
                             f"cost=${d.get('single_cost_usd', 0):.4f}  time={d.get('single_time_s', 0):.1f}s")
        shared.console.print(f"           cost$={d.get('single_cost_score', 0):.0%}  "
                             f"time$={d.get('single_time_score', 0):.0%}  "
                             f"overall={d.get('single_overall', 0):.0%}")
        if d.get("swarm_quality", 0) > 0 or d.get("swarm_cost_score", 0) > 0:
            shared.console.print(f"  [bold]Swarm:[/bold]   quality={d.get('swarm_quality', 0):.0%}  "
                                 f"cost=${d.get('swarm_cost_usd', 0):.4f}  time={d.get('swarm_time_s', 0):.1f}s")
            shared.console.print(f"           cost$={d.get('swarm_cost_score', 0):.0%}  "
                                 f"time$={d.get('swarm_time_score', 0):.0%}  "
                                 f"overall={d.get('swarm_overall', 0):.0%}")
        # Check details
        for label, key in [("Single", "single_checks"), ("Swarm", "swarm_checks")]:
            checks = d.get(key, [])
            if checks:
                shared.console.print(f"\n  [{label}] Checks:")
                for c in checks:
                    status = "[green]PASS[/green]" if c.get("passed") else "[red]FAIL[/red]"
                    shared.console.print(f"    {status} {c.get('check', '?')} "
                                         f"({c.get('category', '?')} w={c.get('weight', 0):.1f}): "
                                         f"{str(c.get('message', ''))[:60]}")
        # Improvement notes
        notes = d.get("notes", {})
        if notes.get("strengths") or notes.get("weaknesses") or notes.get("suggestions"):
            shared.console.print()
            if notes.get("strengths"):
                shared.console.print("  [bold green]Strengths:[/bold green]")
                for s in notes["strengths"]:
                    shared.console.print(f"    - {s}")
            if notes.get("weaknesses"):
                shared.console.print("  [bold red]Weaknesses:[/bold red]")
                for w in notes["weaknesses"]:
                    shared.console.print(f"    - {w}")
            if notes.get("suggestions"):
                shared.console.print("  [bold yellow]Suggestions:[/bold yellow]")
                for s in notes["suggestions"]:
                    shared.console.print(f"    - {s}")
        shared.console.print()
        return

    # Summary table — all known tasks
    updated = ""
    for d in scores.values():
        u = d.get("updated", "")
        if u > updated:
            updated = u
    title = "GROKSWARM EVAL SCORES"
    if updated:
        title += f" (last updated: {updated[:10]})"

    tbl = Table(title=title, show_header=True, header_style="bold", border_style="dim")
    tbl.add_column("Task", width=6)
    tbl.add_column("Cat", width=4)
    tbl.add_column("Description", ratio=1)
    tbl.add_column("S.Qual", width=7, justify="right")
    tbl.add_column("S.Cost", width=8, justify="right")
    tbl.add_column("S.Time", width=8, justify="right")
    tbl.add_column("S.Over", width=7, justify="right")
    tbl.add_column("|", width=1)
    tbl.add_column("W.Qual", width=7, justify="right")
    tbl.add_column("W.Cost", width=8, justify="right")
    tbl.add_column("W.Time", width=8, justify="right")
    tbl.add_column("W.Over", width=7, justify="right")
    tbl.add_column("Verdict", width=12)
    tbl.add_column("Notes", width=8)

    for tid in sorted(all_tasks.keys()):
        info = all_tasks[tid]
        d = scores.get(tid)
        if d:
            def _pct(v): return f"{v:.0%}" if isinstance(v, (int, float)) else "--"
            def _cost(v): return f"${v:.3f}" if isinstance(v, (int, float)) else "--"
            def _time(v): return f"{v:.1f}s" if isinstance(v, (int, float)) else "--"
            has_swarm = d.get("swarm_quality", 0) > 0 or d.get("swarm_cost_score", 0) > 0
            # Derive verdict from overall scores
            s_over = d.get("single_overall", 0) or 0
            w_over = d.get("swarm_overall", 0) or 0
            if not has_swarm:
                verdict = "Single Only"
            elif s_over - w_over > 0.05:
                verdict = "Single Better"
            elif w_over - s_over > 0.05:
                verdict = "Swarm Better"
            else:
                verdict = "Tie"
            # Notes summary: count weaknesses and suggestions
            notes = d.get("notes", {})
            n_w = len(notes.get("weaknesses", []))
            n_s = len(notes.get("suggestions", []))
            notes_str = f"{n_w}W {n_s}S" if (n_w or n_s) else ""
            tbl.add_row(
                tid, info["category"], info["description"][:30],
                _pct(d.get("single_quality")), _cost(d.get("single_cost_usd")),
                _time(d.get("single_time_s")), _pct(d.get("single_overall")),
                "|",
                _pct(d.get("swarm_quality")) if has_swarm else "--",
                _cost(d.get("swarm_cost_usd")) if has_swarm else "--",
                _time(d.get("swarm_time_s")) if has_swarm else "--",
                _pct(d.get("swarm_overall")) if has_swarm else "--",
                verdict, notes_str,
            )
        else:
            tbl.add_row(
                tid, info["category"], info["description"][:30],
                "--", "--", "--", "--", "|",
                "--", "--", "--", "--", "(not run)", "",
            )

    shared.console.print()
    shared.console.print(tbl)
    shared.console.print()


async def handle_diff(arg: str, ctx: CmdContext) -> None:
    """Show all file changes in this session (edit_history) as a diff."""
    from rich.syntax import Syntax
    if not shared.state.edit_history:
        shared.console.print("[swarm.dim]No file edits in this session.[/swarm.dim]")
        return
    # Collect unique files from edit history
    seen_files: dict[str, str | None] = {}  # path -> original content (first snapshot)
    for path, content in shared.state.edit_history:
        if path not in seen_files:
            seen_files[path] = content  # first recorded pre-edit content
    total_lines = 0
    for path, original in seen_files.items():
        fp = Path(path)
        if not fp.is_file():
            if original is not None:
                shared.console.print(f"[bold red]Deleted:[/bold red] {path}")
            continue
        current = fp.read_text(encoding="utf-8", errors="ignore")
        if original is None:
            # File was created in this session
            lines = current.splitlines()
            shared.console.print(f"[bold green]New file:[/bold green] {path} (+{len(lines)} lines)")
            total_lines += len(lines)
        else:
            diff_lines = list(difflib.unified_diff(
                original.splitlines(), current.splitlines(),
                fromfile=f"a/{Path(path).name}", tofile=f"b/{Path(path).name}", lineterm=""))
            if diff_lines:
                display = diff_lines[:80]
                extra = len(diff_lines) - 80
                diff_text = "\n".join(display)
                if extra > 0:
                    diff_text += f"\n... (+{extra} more lines)"
                shared.console.print(Syntax(diff_text, "diff", theme="monokai"))
                total_lines += len(diff_lines)
            else:
                shared.console.print(f"[dim]{path}: no net changes[/dim]")
    shared.console.print(f"\n[swarm.dim]{len(seen_files)} file(s) touched, ~{total_lines} diff lines[/swarm.dim]")


async def handle_copy(arg: str, ctx: CmdContext) -> None:
    """Copy last assistant response to clipboard."""
    # Find last assistant message
    last_text = None
    for msg in reversed(ctx.conversation):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                last_text = content
                break
    if not last_text:
        shared.console.print("[swarm.dim]No assistant response to copy.[/swarm.dim]")
        return
    try:
        if os.name == "nt":
            proc = subprocess.Popen(["clip.exe"], stdin=subprocess.PIPE)
            proc.communicate(last_text.encode("utf-8"))
        elif sys.platform == "darwin":
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(last_text.encode("utf-8"))
        else:
            proc = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
            proc.communicate(last_text.encode("utf-8"))
        if proc.returncode != 0:
            shared.console.print(f"[swarm.error]Clipboard tool exited with code {proc.returncode}[/swarm.error]")
        else:
            shared.console.print(f"[swarm.accent]Copied {len(last_text)} chars to clipboard.[/swarm.accent]")
    except FileNotFoundError:
        shared.console.print("[swarm.error]Clipboard tool not found (clip.exe/pbcopy/xclip).[/swarm.error]")
    except Exception as e:
        shared.console.print(f"[swarm.error]Clipboard error: {e}[/swarm.error]")
