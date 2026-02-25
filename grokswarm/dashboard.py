"""_build_dashboard, _build_swarm_monitor, _build_swarm_feed, _build_swarm_view, _watch_agents, dashboard command."""

import asyncio
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live

import grokswarm.shared as shared
from grokswarm.models import AgentState
from grokswarm.agents import get_bus


def _build_swarm_monitor(task_description: str = "") -> Table:
    _state_icons = {
        AgentState.IDLE: ("\u23f8", "dim"),
        AgentState.THINKING: ("\U0001f9e0", "yellow"),
        AgentState.WORKING: ("\u26a1", "green"),
        AgentState.PAUSED: ("\u23f8", "red"),
        AgentState.DONE: ("\u2714", "cyan"),
        AgentState.ERROR: ("\u2718", "bold red"),
    }
    _plan_icons = {
        "pending": "[dim]  [/dim]",
        "in-progress": "[yellow]>[/yellow]",
        "done": "[green]x[/green]",
        "skipped": "[dim]-[/dim]",
    }
    agent_table = Table(
        show_header=True, header_style="bold", border_style="cyan",
        title=f"[bold]Swarm Monitor[/bold]" + (f" \u2014 {task_description[:60]}" if task_description else ""),
        title_style="bold cyan",
        width=110,
    )
    agent_table.add_column("Agent", style="cyan", width=18)
    agent_table.add_column("", width=3)
    agent_table.add_column("State", width=10)
    agent_table.add_column("Progress", width=40)
    agent_table.add_column("Tokens", justify="right", width=10)
    agent_table.add_column("Cost", justify="right", width=8)

    if shared.state.agents:
        for name, agent in shared.state.agents.items():
            icon, color = _state_icons.get(agent.state, ("?", "white"))
            tool = agent.current_tool or ""
            if agent.plan:
                done = sum(1 for s in agent.plan if s["status"] == "done")
                total = len(agent.plan)
                current = next((s for s in agent.plan if s["status"] == "in-progress"), None)
                if not current:
                    current = next((s for s in agent.plan if s["status"] == "pending"), None)
                step_text = current["step"][:30] if current else ""
                progress = f"[{done}/{total}] {step_text}"
            elif tool:
                progress = f"[dim]{tool}[/dim]"
            else:
                progress = ""
            agent_table.add_row(
                name, icon,
                f"[{color}]{agent.state.value}[/{color}]",
                progress,
                f"{agent.tokens_used:,}",
                f"${agent.cost_usd:.4f}",
            )
            for step in agent.plan:
                step_icon = _plan_icons.get(step["status"], " ")
                step_text = step["step"][:55]
                step_style = "green" if step["status"] == "done" else "yellow" if step["status"] == "in-progress" else "dim"
                agent_table.add_row(
                    "", "", "",
                    f"  {step_icon} [{step_style}]{step_text}[/{step_style}]",
                    "", "",
                )
    else:
        agent_table.add_row("[dim](no agents)[/dim]", "", "", "", "", "")

    return agent_table


def _build_swarm_feed() -> Panel:
    bus = get_bus()
    msgs = bus.read(limit=20)
    feed_lines = []
    for m in msgs:
        ts = m['ts'].split()[-1] if ' ' in m['ts'] else m['ts']
        body = m['body'].replace('\n', ' ')
        if len(body) > 90:
            body = body[:87] + "..."
        kind = m['kind']
        if kind == 'plan':
            feed_lines.append(f"[dim]{ts}[/dim] [bold cyan]Plan:[/bold cyan] {body}")
        elif kind == 'abort':
            feed_lines.append(f"[dim]{ts}[/dim] [bold red]ABORT[/bold red]")
        elif kind == 'status':
            feed_lines.append(f"[dim]{ts}[/dim] [yellow]\u2139 {m['sender']}:[/yellow] {body}")
        elif kind == 'error':
            feed_lines.append(f"[dim]{ts}[/dim] [bold red]\u2718 {m['sender']}:[/bold red] {body}")
        elif kind == 'result':
            feed_lines.append(f"[dim]{ts}[/dim] [green]\u2714 {m['sender']}:[/green] {body}")
        else:
            feed_lines.append(f"[dim]{ts}[/dim] [bold]{m['sender']}[/bold]\u2192{m['recipient']}: {body}")
    return Panel(
        "\n".join(feed_lines[-12:]) or "[dim](waiting for messages...)[/dim]",
        title="Live Feed", border_style="dim", width=110,
    )


def _build_swarm_view(task_description: str = ""):
    from rich.console import Group
    metrics = get_bus().get_metrics()
    total = metrics['total_tokens']
    footer = f"  [dim]Tokens: {total:,}  |  Session: ${shared.state.global_cost_usd:.4f}  |  Project: ${shared.state.project_cost_usd:.4f}  |  Press [bold]q[/bold]/Escape to exit monitor[/dim]"
    return Group(
        _build_swarm_monitor(task_description),
        _build_swarm_feed(),
        footer,
    )


async def _watch_agents(task_description: str = "", auto_exit: bool = True):
    import threading, sys
    _exit = threading.Event()

    def _key_listener():
        if sys.platform == "win32":
            import msvcrt
            while not _exit.is_set():
                try:
                    ch = msvcrt.getch()
                except Exception:
                    return
                if ch == b'\xe0' or ch == b'\x00':
                    try: msvcrt.getch()
                    except Exception: pass
                    continue
                if ch in (b'q', b'Q', b'\x1b'):
                    _exit.set()
                    return
        else:
            try:
                import tty, termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while not _exit.is_set():
                        import select
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            ch = sys.stdin.read(1)
                            if ch in ('q', 'Q', '\x1b'):
                                _exit.set()
                                return
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                _exit.wait()

    listener = threading.Thread(target=_key_listener, daemon=True)
    listener.start()

    try:
        with Live(_build_swarm_view(task_description), console=shared.console, refresh_per_second=2, screen=True) as live:
            while not _exit.is_set():
                if auto_exit and shared.state.agents:
                    all_done = all(
                        a.state in (AgentState.DONE, AgentState.ERROR)
                        for a in shared.state.agents.values()
                    )
                    if all_done and not any(not t.done() for t in shared._background_tasks.values()):
                        live.update(_build_swarm_view(task_description))
                        await asyncio.sleep(1)
                        break
                await asyncio.sleep(0.5)
                if not _exit.is_set():
                    live.update(_build_swarm_view(task_description))
    except KeyboardInterrupt:
        pass
    finally:
        _exit.set()


def _build_dashboard() -> Layout:
    from rich.tree import Tree

    bus = get_bus()
    metrics = bus.get_metrics()
    total_tokens = metrics["total_tokens"]
    proj_lines = [
        f"[bold]Directory:[/bold]  {shared.PROJECT_DIR}",
        f"[bold]Model:[/bold]      {shared.MODEL}",
        f"[bold]Version:[/bold]    {shared.VERSION}",
        f"[bold]Trust:[/bold]      {'[green]ON[/green]' if shared.state.trust_mode else '[red]OFF[/red]'}",
        f"[bold]Read-only:[/bold]  {'[yellow]ON[/yellow]' if shared.state.read_only else 'OFF'}",
    ]
    project_panel = Panel("\n".join(proj_lines), title="Project", border_style="cyan")

    metrics_lines = [
        f"[bold]Prompt Tokens:[/bold]     {metrics['prompt_tokens']:,}",
        f"[bold]Completion Tokens:[/bold] {metrics['completion_tokens']:,}",
        f"[bold]Total Tokens:[/bold]      {total_tokens:,}",
        f"[bold]Global Cost:[/bold]       ${shared.state.global_cost_usd:.4f}",
    ]
    if shared.state.global_cost_budget_usd > 0:
        metrics_lines.append(f"[bold]Cost Budget:[/bold]      ${shared.state.global_cost_budget_usd:.4f}")
    metrics_panel = Panel("\n".join(metrics_lines), title="Session Metrics", border_style="magenta")

    _state_colors = {
        AgentState.IDLE: "dim",
        AgentState.THINKING: "yellow",
        AgentState.WORKING: "green",
        AgentState.PAUSED: "red",
        AgentState.DONE: "cyan",
        AgentState.ERROR: "bold red",
    }
    agent_tree = Tree("[bold]Agent Swarm[/bold]")
    if shared.state.agents:
        children_map: dict[str | None, list[str]] = {}
        for aname, ainfo in shared.state.agents.items():
            children_map.setdefault(ainfo.parent, []).append(aname)

        def _add_children(tree_node, parent_name):
            for child_name in children_map.get(parent_name, []):
                child = shared.state.agents[child_name]
                color = _state_colors.get(child.state, "white")
                budget_str = ""
                if child.token_budget > 0:
                    pct = min(100, int(child.tokens_used / child.token_budget * 100))
                    budget_str = f" [{pct}% budget]"
                bg_str = ""
                if child_name in shared._background_tasks:
                    t = shared._background_tasks[child_name]
                    bg_str = " \u23f3" if not t.done() else " \u2713"
                label = f"[{color}]{child_name}[/{color}] ({child.expert}) [{color}]{child.state.value}[/{color}]{budget_str}{bg_str}"
                child_node = tree_node.add(label)
                _add_children(child_node, child_name)

        _add_children(agent_tree, None)
    else:
        agent_tree.add("[dim](no active agents)[/dim]")
    agents_panel = Panel(agent_tree, title=f"Agents ({len(shared.state.agents)})", border_style="yellow")

    expert_files = sorted(shared.EXPERTS_DIR.glob("*.yaml"))
    expert_names = [f.stem for f in expert_files] or ["(none)"]
    experts_panel = Panel("\n".join(expert_names), title=f"Experts ({len(expert_files)})", border_style="green")

    skill_files = sorted(shared.SKILLS_DIR.glob("*.yaml"))
    skill_names = [f.stem for f in skill_files] or ["(none)"]
    skills_panel = Panel("\n".join(skill_names), title=f"Skills ({len(skill_files)})", border_style="green")

    sess_files = sorted(shared.SESSIONS_DIR.glob("*.json"))[-10:]
    sess_names = [f.stem for f in sess_files] or ["(none)"]
    sessions_panel = Panel("\n".join(sess_names), title=f"Sessions ({len(list(shared.SESSIONS_DIR.glob('*.json')))})", border_style="yellow")

    recent = shared.state.edit_history[-5:] if shared.state.edit_history else []
    edit_lines = []
    for path, content in reversed(recent):
        label = "new" if content is None else "edit"
        edit_lines.append(f"[dim]{label}[/dim]  {path}")
    edits_panel = Panel("\n".join(edit_lines) or "(no edits yet)", title="Recent Edits", border_style="magenta")

    team_files = sorted(shared.TEAMS_DIR.glob("*.yaml"))
    team_names = [f.stem for f in team_files] or ["(none)"]
    teams_panel = Panel("\n".join(team_names), title=f"Teams ({len(team_files)})", border_style="blue")

    msgs = bus.read(limit=15)
    feed_lines = []
    for m in msgs:
        ts = m['ts'].split()[-1]
        body = m['body'].replace('\n', ' ')
        if len(body) > 100:
            body = body[:97] + "..."
        if m['kind'] == 'plan':
            feed_lines.append(f"[dim]{ts}[/dim] [bold cyan]Plan:[/bold cyan] {body}")
        elif m['kind'] == 'abort':
            feed_lines.append(f"[dim]{ts}[/dim] [bold red]ABORT SIGNAL[/bold red]")
        elif m['kind'] == 'status':
            feed_lines.append(f"[dim]{ts}[/dim] [yellow]\u2139 {m['sender']}:[/yellow] {body}")
        elif m['kind'] == 'error':
            feed_lines.append(f"[dim]{ts}[/dim] [bold red]\u2718 {m['sender']}:[/bold red] {body}")
        else:
            feed_lines.append(f"[dim]{ts}[/dim] [[bold green]{m['sender']}[/bold green]\u2192{m['recipient']}] {body}")
    feed_panel = Panel("\n".join(feed_lines) or "(no active swarm messages)", title="Live Swarm Feed", border_style="cyan")

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=7),
        Layout(name="agents_row", size=10),
        Layout(name="middle", size=7),
        Layout(feed_panel, name="feed"),
        Layout(name="bottom", size=8),
    )
    layout["header"].split_row(
        Layout(project_panel, name="project"),
        Layout(metrics_panel, name="metrics"),
    )
    layout["agents_row"].split_row(
        Layout(agents_panel, name="agents"),
        Layout(edits_panel, name="edits"),
    )
    layout["middle"].split_row(
        Layout(experts_panel, name="experts"),
        Layout(skills_panel, name="skills"),
        Layout(teams_panel, name="teams"),
    )
    layout["bottom"].split_row(
        Layout(sessions_panel, name="sessions"),
    )
    return layout


def _check_quit_key() -> bool:
    import sys
    if sys.platform == "win32":
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch == b'\xe0':
                msvcrt.getch()
                return False
            return ch in (b'q', b'Q', b'\x1b')
    else:
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            return ch in ('q', 'Q', '\x1b')
    return False


@shared.app.command()
def dashboard():
    """Live TUI dashboard -- shows project state, experts, skills, sessions."""
    import threading, sys
    _dashboard_exit = threading.Event()

    def _key_listener():
        if sys.platform == "win32":
            import msvcrt
            while not _dashboard_exit.is_set():
                try:
                    ch = msvcrt.getch()
                except Exception:
                    return
                if ch == b'\xe0' or ch == b'\x00':
                    try:
                        msvcrt.getch()
                    except Exception:
                        pass
                    continue
                if ch in (b'q', b'Q', b'\x1b'):
                    _dashboard_exit.set()
                    return
        else:
            try:
                import tty, termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while not _dashboard_exit.is_set():
                        import select
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            ch = sys.stdin.read(1)
                            if ch in ('q', 'Q', '\x1b'):
                                _dashboard_exit.set()
                                return
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                _dashboard_exit.wait()

    listener = threading.Thread(target=_key_listener, daemon=True)
    listener.start()
    try:
        with Live(_build_dashboard(), console=shared.console, refresh_per_second=2, screen=True) as live:
            while not _dashboard_exit.is_set():
                _dashboard_exit.wait(0.5)
                if not _dashboard_exit.is_set():
                    live.update(_build_dashboard())
    except KeyboardInterrupt:
        pass
    finally:
        _dashboard_exit.set()
