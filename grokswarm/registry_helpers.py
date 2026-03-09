"""Registry helpers — expert/skill proposals, listing, persistence."""

import json
import yaml
from datetime import datetime

import grokswarm.shared as shared


def seed_defaults():
    if not any(shared.EXPERTS_DIR.iterdir()):
        defaults = {
            "researcher": {"name": "Researcher", "mindset": "Thorough, source-critical, always cites latest data.", "objectives": ["Deliver comprehensive, up-to-date summaries"]},
            "coder": {"name": "Coder", "mindset": "Clean, efficient, testable Python-first.", "objectives": ["Produce production-ready code"]},
            "assistant": {"name": "Personal_Assistant", "mindset": "Organized, efficient, does not waste time.", "objectives": ["Handle tasks efficiently"]},
            "finance": {"name": "Finance_Optimizer", "mindset": "Conservative growth, tax-aware, long-term focused.", "objectives": ["Optimize financial outcomes"]}
        }
        for name, data in defaults.items():
            (shared.EXPERTS_DIR / f"{name}.yaml").write_text(yaml.dump(data))


def list_experts():
    return [f.stem for f in shared.EXPERTS_DIR.glob("*.yaml")]


def list_skills():
    return [f.stem for f in shared.SKILLS_DIR.glob("*.yaml")]


def save_memory(key: str, content: str):
    entry = {"timestamp": datetime.now().isoformat(), "content": content}
    (shared.MEMORY_DIR / f"{key}.json").write_text(json.dumps(entry, indent=2))


def list_memory() -> list[dict]:
    """Return all memory entries sorted newest-first."""
    entries = []
    for f in shared.MEMORY_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            entries.append({
                "key": f.stem,
                "timestamp": data.get("timestamp", ""),
                "size": len(data.get("content", "")),
            })
        except Exception:
            entries.append({"key": f.stem, "timestamp": "?", "size": 0})
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries


def prune_memory(max_age_days: int = 30) -> int:
    """Delete memory files older than *max_age_days*. Returns count deleted."""
    cutoff = datetime.now().timestamp() - (max_age_days * 86400)
    deleted = 0
    for f in shared.MEMORY_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(data["timestamp"]).timestamp()
            if ts < cutoff:
                f.unlink()
                deleted += 1
        except Exception:
            continue
    return deleted


def propose_expert(name: str, mindset: str, objectives: list[str]) -> str:
    from rich.panel import Panel
    safe_name = name.lower().replace(" ", "_")
    expert_file = shared.EXPERTS_DIR / f"{safe_name}.yaml"
    if expert_file.exists():
        return f"Expert '{safe_name}' already exists. Use a different name."
    shared.console.print()
    shared.console.print(Panel(
        f"[bold]Name:[/bold] {name}\n"
        f"[bold]Mindset:[/bold] {mindset}\n"
        f"[bold]Objectives:[/bold]\n" + "\n".join(f"  * {o}" for o in objectives),
        title="[swarm.accent]> New Expert Proposal[/swarm.accent]",
        border_style="bright_green",
        padding=(1, 2),
    ))
    if shared._auto_approve("[bold yellow]Approve this expert?[/bold yellow]"):
        data = {"name": name, "mindset": mindset, "objectives": objectives}
        expert_file.write_text(yaml.dump(data, default_flow_style=False))
        return f"Expert '{name}' created and saved to experts/{safe_name}.yaml"
    return "Expert creation cancelled by user."


def propose_skill(name: str, description: str, steps: list[str] | None = None) -> str:
    from rich.panel import Panel
    from grokswarm.tools_registry import _register_skill_tool
    safe_name = name.lower().replace(" ", "_")
    skill_file = shared.SKILLS_DIR / f"{safe_name}.yaml"
    if skill_file.exists():
        return f"Skill '{safe_name}' already exists. Use a different name."
    shared.console.print()
    body = f"[bold]Name:[/bold] {name}\n[bold]Description:[/bold] {description}"
    if steps:
        body += "\n[bold]Steps:[/bold]\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
    shared.console.print(Panel(
        body,
        title="[swarm.accent]> New Skill Proposal[/swarm.accent]",
        border_style="bright_green",
        padding=(1, 2),
    ))
    if shared._auto_approve("[bold yellow]Approve this skill?[/bold yellow]"):
        data = {"name": name, "description": description, "version": "1.0"}
        if steps:
            data["steps"] = steps
        skill_file.write_text(yaml.dump(data, default_flow_style=False))
        _register_skill_tool(safe_name, description)
        return f"Skill '{name}' created and saved to skills/{safe_name}.yaml (registered as tool: skill_{safe_name})"
    return "Skill creation cancelled by user."


def get_registry() -> str:
    lines = ["Experts:"]
    for f in sorted(shared.EXPERTS_DIR.glob("*.yaml")):
        data = yaml.safe_load(f.read_text())
        lines.append(f"  * {data.get('name', f.stem)} -- {data.get('mindset', '(no mindset)')}")
    if len(lines) == 1:
        lines.append("  (none)")
    lines.append("\nSkills:")
    for f in sorted(shared.SKILLS_DIR.glob("*.yaml")):
        data = yaml.safe_load(f.read_text())
        lines.append(f"  * {data.get('name', f.stem)} -- {data.get('description', '(no description)')}")
    if lines[-1] == "\nSkills:":
        lines.append("  (none)")
    return "\n".join(lines)
