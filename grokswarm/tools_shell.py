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
    from grokswarm import llm
    try:
        chat = llm.create_chat(shared.MODEL, max_tokens=300)
        llm.populate_chat(chat, [
            {"role": "system", "content": """You are a command safety analyst. Given a shell command, explain:
1. What the command does (plain English)
2. What files/resources it accesses or modifies
3. Safety assessment: SAFE, CAUTION, or DANGEROUS
4. Any risks or side effects

Be concise — 3-5 lines max. Working directory is: """ + str(shared.PROJECT_DIR)},
            {"role": "user", "content": f"Analyze this command:\n```\n{command}\n```"}
        ])
        response = await shared._api_call_with_retry(
            lambda: chat.sample(),
            label="SafetyCheck"
        )
        return response.content or "(no analysis available)"
    except Exception as e:
        return f"(safety analysis failed: {e})"


def _approval_prompt(command: str, is_dangerous: bool = False) -> str:
    import sys
    out = sys.__stdout__
    sep_color = "\033[31m" if is_dangerous else "\033[33m"  # red vs yellow
    reset = "\033[0m"
    dim = "\033[2m"
    bold = "\033[1m"
    label = "Confirm dangerous command?" if is_dangerous else "Approve command?"

    # Non-TTY fallback
    try:
        is_tty = sys.__stdin__.isatty()
    except Exception:
        is_tty = False
    if not is_tty:
        return "n"

    out.write(
        f"\n{sep_color}{'─' * 50}{reset}\n"
        f"  {bold}{label}{reset}\n"
        f"  {dim} y {reset} approve"
        f"   {dim} n {reset} reject"
        f"   {dim} i {reset} explain"
        f"   {dim} t {reset} trust all\n"
    )
    out.flush()

    while True:
        try:
            ch = shared._read_single_key()
        except (EOFError, KeyboardInterrupt):
            out.write(f"  {dim}> Rejected{reset}\n")
            out.flush()
            return "n"
        key = ch.lower()
        if key == "y":
            out.write(f"  {dim}> Approved{reset}\n")
            out.flush()
            return "y"
        elif key in ("n", "\x1b", "\x03"):  # n, Esc, Ctrl+C
            out.write(f"  {dim}> Rejected{reset}\n")
            out.flush()
            return "n"
        elif key == "i":
            out.write(f"  {dim}> Explaining...{reset}\n")
            out.flush()
            return "i"
        elif key == "t":
            out.write(f"  {dim}> Trusting all{reset}\n")
            out.flush()
            return "trust"
        # Ignore any other key


def run_shell(command: str):
    work_dir = shared.get_project_dir()
    if shared.state.agent_mode == 0 and shared.state.verbose_mode:
        shared.console.print(f"[bold yellow]About to EXECUTE:[/bold yellow] {command}")
        shared.console.print(f"[dim]Working directory: {work_dir}[/dim]")
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
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=work_dir, timeout=120)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Error: {e}"
