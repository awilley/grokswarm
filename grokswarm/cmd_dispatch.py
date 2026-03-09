"""Registry-based command dispatch for the REPL slash commands.

Each command is registered with metadata (description, whether it can run
while busy, aliases) and a handler coroutine/function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class CmdEntry:
    """Metadata + handler for one slash command."""
    name: str
    description: str
    handler: Callable  # async def handler(arg: str, ctx: CmdContext) -> str | None
    allow_while_busy: bool = True
    aliases: list[str] = field(default_factory=list)


@dataclass
class CmdContext:
    """Mutable state bag passed to every command handler."""
    conversation: list[dict]
    session_name: str | None
    session: Any  # PromptSession
    # Callbacks that the handler can use to modify REPL state:
    save_session: Callable | None = None
    quit_flag: bool = False  # handler sets to True to exit REPL
    new_session_name: str | None = None  # handler sets to change session_name


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_COMMANDS: dict[str, CmdEntry] = {}
_ALIAS_MAP: dict[str, str] = {}  # alias -> canonical name


def register(
    name: str,
    description: str,
    *,
    allow_while_busy: bool = True,
    aliases: list[str] | None = None,
):
    """Decorator to register a command handler."""
    def decorator(fn: Callable) -> Callable:
        entry = CmdEntry(
            name=name,
            description=description,
            handler=fn,
            allow_while_busy=allow_while_busy,
            aliases=aliases or [],
        )
        _COMMANDS[name] = entry
        for alias in entry.aliases:
            _ALIAS_MAP[alias] = name
        return fn
    return decorator


def get_command(cmd: str) -> CmdEntry | None:
    """Look up a command by name or alias."""
    canonical = _ALIAS_MAP.get(cmd, cmd)
    return _COMMANDS.get(canonical)


def all_commands() -> dict[str, CmdEntry]:
    """Return the full command registry."""
    return _COMMANDS


def busy_allowed_set() -> set[str]:
    """Return the set of command names allowed while processing_busy."""
    names: set[str] = set()
    for entry in _COMMANDS.values():
        if entry.allow_while_busy:
            names.add(entry.name)
            names.update(entry.aliases)
    return names
