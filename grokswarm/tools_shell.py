"""Shell safety checks and run_shell."""

import re
import asyncio
import subprocess

from rich.panel import Panel

import grokswarm.shared as shared

DANGEROUS_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*)?\s*-[a-zA-Z]*r",
    r"\brm\s+-[a-zA-Z]*f",
    r"\bcurl\b.*\|\s*(ba)?sh",
    r"\bwget\b.*\|\s*(ba)?sh",
    r"\bmkfs\b",
    r"\bdd\b\s+if=",
    r":\(\)\{\s*:\|",
    r"\b(poweroff|reboot|shutdown)\b",
    r"\bformat\s+[a-zA-Z]:",
    r"> /dev/(sd|null|zero)",
    r"\bgit\s+push\s+.*--force",
    r"\bgit\s+reset\s+--hard",
]
_DANGEROUS_RX = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]


def _is_dangerous_command(command: str) -> bool:
    return any(rx.search(command) for rx in _DANGEROUS_RX)


async def _explain_command_safety(command: str) -> str:
    try:
        response = await shared._api_call_with_retry(
            lambda: shared.client.chat.completions.create(
                model=shared.MODEL,
                messages=[
                    {"role": "system", "content": """You are a command safety analyst. Given a shell command, explain:
1. What the command does (plain English)
2. What files/resources it accesses or modifies
3. Safety assessment: SAFE, CAUTION, or DANGEROUS
4. Any risks or side effects

Be concise — 3-5 lines max. Working directory is: """ + str(shared.PROJECT_DIR)},
                    {"role": "user", "content": f"Analyze this command:\n```\n{command}\n```"}
                ],
                max_tokens=300,
            ),
            label="SafetyCheck"
        )
        return response.choices[0].message.content or "(no analysis available)"
    except Exception as e:
        return f"(safety analysis failed: {e})"


def _approval_prompt(command: str, is_dangerous: bool = False) -> str:
    if is_dangerous:
        label = "[bold red]CONFIRM dangerous command? [y/n/i/trust][/bold red] "
    else:
        label = "Approve command? [y/n/i/trust] "
    while True:
        shared.console.print(f"  [dim](y=yes, n=no, i=info/explain, trust=approve all remaining)[/dim]")
        answer = shared.console.input(label).strip().lower()
        if answer in ("y", "yes"):
            return "y"
        elif answer in ("n", "no", ""):
            return "n"
        elif answer in ("i", "info"):
            return "i"
        elif answer == "trust":
            return "trust"
        else:
            shared.console.print("  [dim]Please enter y, n, i, or trust[/dim]")


def run_shell(command: str):
    if shared.state.agent_mode == 0 and shared.state.verbose_mode:
        shared.console.print(f"[bold yellow]About to EXECUTE:[/bold yellow] {command}")
        shared.console.print(f"[dim]Working directory: {shared.PROJECT_DIR}[/dim]")
    is_dangerous = _is_dangerous_command(command)

    if is_dangerous:
        shared.console.print("[bold red][DANGEROUS COMMAND DETECTED][/bold red]")
        shared.console.print("[bold red]This command matches a dangerous pattern. Review carefully.[/bold red]")
        while True:
            choice = _approval_prompt(command, is_dangerous=True)
            if choice == "y":
                break
            elif choice == "n":
                return "Cancelled: dangerous command rejected."
            elif choice == "i":
                shared.console.print("[swarm.dim]  Analyzing command safety...[/swarm.dim]")
                explanation = asyncio.run(_explain_command_safety(command))
                shared.console.print()
                shared.console.print(Panel(explanation, title="Command Safety Analysis", border_style="cyan"))
                shared.console.print()
            elif choice == "trust":
                break
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=shared.PROJECT_DIR, timeout=120)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Error: {e}"
