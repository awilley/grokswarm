"""Typer CLI commands: swarm, team-*, task, expert, skills/experts list, create_*, abort."""

import asyncio
import yaml
from rich.table import Table

import grokswarm.shared as shared
from grokswarm.registry_helpers import list_experts, list_skills
from grokswarm.agents import get_bus, run_expert


@shared.app.command()
def swarm(description: str):
    from grokswarm.repl import _swarm_async
    asyncio.run(_swarm_async(description))


@shared.app.command("team-save")
def team_save(name: str):
    from datetime import datetime
    team = {"name": name, "experts": list_experts(), "created": datetime.now().isoformat()}
    (shared.TEAMS_DIR / f"{name.lower()}.yaml").write_text(yaml.dump(team))
    shared.console.print(f"[green]Team '{name}' saved.[/green]")


@shared.app.command("team-list")
def team_list():
    table = Table(title="Saved Teams")
    table.add_column("Team Name")
    for t in shared.TEAMS_DIR.glob("*.yaml"):
        table.add_row(t.stem)
    shared.console.print(table)


@shared.app.command("team-run")
def team_run(name: str, task_desc: str):
    team_file = shared.TEAMS_DIR / f"{name.lower()}.yaml"
    if not team_file.exists():
        shared.console.print(f"[red]Team {name} not found.[/red]")
        return
    data = yaml.safe_load(team_file.read_text())
    if not isinstance(data, dict) or "name" not in data:
        shared.console.print(f"[red]Invalid team file {team_file}: must have 'name' field.[/red]")
        return
    shared.console.print(f"[bold cyan]Running Team:[/bold cyan] {data['name']}")
    bus = get_bus()
    bus.clear()

    async def _run_team():
        for expert in data.get("experts", []):
            if bus.check_abort():
                shared.console.print("[swarm.warning]Team run aborted by dashboard.[/swarm.warning]")
                break
            await run_expert(expert, task_desc, bus=bus)
    asyncio.run(_run_team())


@shared.app.command()
def task(description: str):
    shared.console.print(f"[bold green]Executing task:[/bold green] {description}")
    asyncio.run(run_expert("assistant", description))


@shared.app.command()
def expert(name: str, task_desc: str):
    asyncio.run(run_expert(name, task_desc))


@shared.app.command("skills-list")
def skills_list():
    table = Table(title="Skill Registry")
    table.add_column("Skill")
    for s in list_skills():
        table.add_row(s)
    shared.console.print(table)


@shared.app.command("experts-list")
def experts_list():
    table = Table(title="Expert Registry")
    table.add_column("Expert")
    for e in list_experts():
        table.add_row(e)
    shared.console.print(table)


@shared.app.command()
def create_skill(name: str, description: str):
    skill = {"name": name, "description": description, "version": "1.0"}
    (shared.SKILLS_DIR / f"{name.lower()}.yaml").write_text(yaml.dump(skill))
    shared.console.print(f"[green]Skill '{name}' created.[/green]")


@shared.app.command()
def create_expert(name: str, mindset: str):
    expert_data = {"name": name, "mindset": mindset, "objectives": []}
    (shared.EXPERTS_DIR / f"{name.lower()}.yaml").write_text(yaml.dump(expert_data))
    shared.console.print(f"[green]Expert '{name}' created.[/green]")


@shared.app.command()
def abort():
    """Send an abort signal to stop any currently running swarm."""
    bus = get_bus()
    bus.post("user", "Abort requested", kind="abort")
    shared.console.print("[bold red]Abort signal sent to SwarmBus.[/bold red]")
