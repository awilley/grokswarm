"""Mutable globals, console, app, client, constants, and utility functions."""

import os
import re
import sys
import asyncio
import atexit
import concurrent.futures
import contextvars
import typer
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.theme import Theme

from dotenv import load_dotenv
from prompt_toolkit.history import FileHistory

from grokswarm.models import SwarmState

load_dotenv()

# Ensure UTF-8 output on Windows to prevent UnicodeEncodeError with emoji/symbols
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# -- Theme & Console --
SWARM_THEME = Theme({
    "swarm.accent": "bold bright_green",
    "swarm.dim": "dim white",
    "swarm.user": "bold bright_cyan",
    "swarm.ai": "bold bright_green",
    "swarm.warning": "bold yellow",
    "swarm.error": "bold red",
})
console = Console(theme=SWARM_THEME)
app = typer.Typer(rich_markup_mode="rich", help="Grok Swarm -- your local persistent AI workhorse", invoke_without_command=True)

THINKING_FRAMES = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]

# -- Module-level toolbar spinner state --
_toolbar_status = ""
_toolbar_spinner_idx = 0
_toolbar_app_ref = None
_toolbar_suspended = False
_toolbar_status_expires: float = 0.0

# State for safely pausing the main prompt loop
_prompt_suspend_event = asyncio.Event()
_prompt_resume_event = asyncio.Event()
_saved_prompt_text = ""
_is_prompt_suspended = False
_suspend_lock = asyncio.Lock()

# Escape-key cancellation flag (checked by streaming/tool loops)
_cancel_event = asyncio.Event()

# Pending clipboard images (base64 data URIs) queued via Alt+V
_pending_images: list[str] = []

# Pending approval prompt (toolbar-based confirmation)
# {
#   "prompt": str,                                     # Question text
#   "choices": [{"label": str, "value": any}, ...],    # Options
#   "selected": int,                                   # Currently highlighted index
#   "future": concurrent.futures.Future,               # Blocks caller thread
# }
_pending_approval: dict | None = None


def _set_status(text: str):
    global _toolbar_status, _toolbar_spinner_idx
    _toolbar_status = text
    _toolbar_spinner_idx = 0
    if _toolbar_app_ref and not _toolbar_suspended and not _is_prompt_suspended:
        try:
            _toolbar_app_ref.invalidate()
        except Exception:
            pass


def _clear_status():
    global _toolbar_status
    _toolbar_status = ""
    if _toolbar_app_ref and not _toolbar_suspended and not _is_prompt_suspended:
        try:
            _toolbar_app_ref.invalidate()
        except Exception:
            pass


# -- API Key & Client --
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    console.print("[swarm.error]Error: XAI_API_KEY not found in .env[/swarm.error]")
    raise typer.Exit(1)

from grokswarm import llm as _llm
_llm.init_client(XAI_API_KEY)

# -- Constants --
VERSION = "0.30.0"
MODEL = "grok-4-1-fast-reasoning"
BASE_URL = "https://api.x.ai/v1"
MAX_TOKENS = 16384
CODE_MODEL: str | None = None

# -- Pricing (per 1M tokens: input, cached_input, output) --
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    # NOTE: Order matters — longer prefixes MUST come before shorter ones
    # because _get_pricing uses startswith() prefix matching.
    #
    # Tier 1: Fast (non-reasoning) — exploration, simple tool calls
    "grok-4-1-fast-non-reasoning": (0.20, 0.05,  0.50),
    "grok-4-1-fast-reasoning":     (0.20, 0.05,  0.50),
    "grok-4-1-fast":               (0.20, 0.05,  0.50),
    # grok-4-fast variants (same pricing as 4-1-fast)
    "grok-4-fast-reasoning":       (0.20, 0.05,  0.50),
    "grok-4-fast-non-reasoning":   (0.20, 0.05,  0.50),
    # Tier 3: Hardcore — complex planning, decomposition (xAI: $2.00/$6.00, cached $0.20)
    "grok-4.20-multi-agent":       (2.00, 0.20,  6.00),
    "grok-4.20":                   (2.00, 0.20,  6.00),
    # grok-4 variants (longer prefixes first)
    "grok-4-0709":                 (3.00, 0.75, 15.00),
    "grok-4":                      (3.00, 0.75, 15.00),
    "grok-code-fast-1":            (0.20, 0.02,  1.50),
    # grok-3 variants (longer prefixes first)
    "grok-3-mini":                 (0.30, 0.07,  0.50),
    "grok-3-fast":                 (0.60, 0.15,  3.00),
    "grok-3":                      (3.00, 0.75, 15.00),
    # Legacy
    "grok-2":                      (2.00, 0.50, 10.00),
}
_DEFAULT_PRICING = (0.20, 0.05, 0.50)

# -- Context window sizes (tokens) --
MODEL_CONTEXT_WINDOW: dict[str, int] = {
    "grok-4": 131072,
    "grok-3": 131072,
    "grok-2": 131072,
}
_DEFAULT_CONTEXT_WINDOW = 131072


def _get_context_window(model: str) -> int:
    m = model.lower()
    for prefix, size in MODEL_CONTEXT_WINDOW.items():
        if m.startswith(prefix):
            return size
    return _DEFAULT_CONTEXT_WINDOW


def _get_pricing(model: str) -> tuple[float, float, float]:
    """Returns (input_rate, cached_input_rate, output_rate) per 1M tokens."""
    m = model.lower()
    for prefix, rates in MODEL_PRICING.items():
        if m.startswith(prefix):
            return rates
    return _DEFAULT_PRICING


# -- State --
state = SwarmState()

# -- Input Queue --
_input_queue: asyncio.Queue[str | None] = asyncio.Queue()


def _drain_input_queue() -> list[str]:
    items: list[str] = []
    while True:
        try:
            items.append(_input_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


# -- Session Log --
_session_log_file = None


def _open_session_log():
    global _session_log_file
    log_dir = PROJECT_DIR / ".grokswarm"
    log_dir.mkdir(parents=True, exist_ok=True)
    _session_log_file = open(log_dir / "session.log", "a", encoding="utf-8")


def _log(msg: str):
    global _session_log_file
    if _session_log_file is None:
        try:
            _open_session_log()
        except OSError:
            return
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        _session_log_file.write(f"[{ts}] {msg}\n")
        _session_log_file.flush()
    except OSError:
        pass


# -- Directories --
GROKSWARM_HOME = Path(os.environ.get("GROKSWARM_HOME", Path(__file__).resolve().parent.parent)).resolve()
PROJECT_DIR = Path.cwd().resolve()
PROJECT_CONTEXT: dict = {}
SYSTEM_PROMPT: str = ""

SKILLS_DIR = Path("skills")
EXPERTS_DIR = Path("experts")
TEAMS_DIR = Path("teams")
MEMORY_DIR = Path("memory")
SESSIONS_DIR = Path.home() / ".grokswarm" / "sessions"
CONTEXT_CACHE_DIR = Path.home() / ".grokswarm" / "cache"
PLUGINS_DIR = Path.home() / ".grokswarm" / "plugins"
_RECENT_PROJECTS_FILE = Path.home() / ".grokswarm" / "recent_projects.json"
SKILLS_DIR.mkdir(exist_ok=True)
EXPERTS_DIR.mkdir(exist_ok=True)
TEAMS_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True, parents=True)
CONTEXT_CACHE_DIR.mkdir(exist_ok=True, parents=True)
PLUGINS_DIR.mkdir(exist_ok=True, parents=True)


# -- Workspace override (per-agent branch isolation via contextvars) --
# When an agent runs in a git worktree, this overrides PROJECT_DIR for that task.
# contextvars are automatically copied per asyncio.Task, so concurrent agents
# each see their own workspace without interfering with each other.
_workspace_override: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    '_workspace_override', default=None
)


def get_project_dir() -> Path:
    """Return the effective project directory for the current execution context.

    Returns the agent's worktree path if running in branch-isolated mode,
    otherwise falls back to the global PROJECT_DIR.
    """
    ws = _workspace_override.get()
    return ws if ws is not None else PROJECT_DIR


# -- Background tasks --
_agent_counter = 0
_background_tasks: dict[str, asyncio.Task] = {}

# -- Bus instance --
_bus_instance = None

# -- Orchestrator DAG (for /tasks display) --
_current_dag = None
_last_planning_time: float = 0.0  # seconds spent in Orchestrator decomposition


# -- Secret patterns --
_SECRET_PATTERNS = [
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),
    re.compile(r'xai-[a-zA-Z0-9]{20,}'),
    re.compile(r'-----BEGIN[A-Z ]*PRIVATE KEY-----'),
    re.compile(r'eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}'),
    re.compile(r'(?i)(?:api[_-]?key|secret|token|password)\s*[=:]\s*["\']?[a-zA-Z0-9_/+=-]{16,}'),
    re.compile(r'(?i)bearer\s+[a-zA-Z0-9_./-]{20,}'),
]


def _redact_secrets(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub('[REDACTED]', text)
    return text


def _sanitize_surrogates(text: str) -> str:
    return text.encode('utf-8', errors='replace').decode('utf-8')


class SafeFileHistory(FileHistory):
    """FileHistory that strips surrogate characters before writing (Windows clipboard fix)."""
    def store_string(self, string: str) -> None:
        super().store_string(string.encode('utf-8', errors='replace').decode('utf-8'))


def _read_single_key() -> str:
    """Read a single keypress without requiring Enter."""
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        # Windows special keys send \x00 or \xe0 prefix + second byte — consume both
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()  # discard scan code
            return ""
        return ch
    else:
        import tty
        import termios
        fd = sys.__stdin__.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.__stdin__.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch


def _inline_confirm(prompt_text: str, default: bool = True) -> bool:
    """Ask y/n via single-keypress with ANSI-styled hint bar (inline/stdout path).

    Used when no REPL is running (tests, scripts, between prompts).
    Bypasses Rich and prompt_toolkit to avoid spinner-character leakage.
    """
    out = sys.__stdout__
    sep_color = "\033[90m" if default else "\033[33m"  # gray vs yellow
    reset = "\033[0m"
    dim = "\033[2m"
    bold = "\033[1m"

    # Check for non-TTY (piped input, CI, etc.)
    try:
        is_tty = sys.__stdin__.isatty()
    except Exception:
        is_tty = False
    if not is_tty:
        return default

    def_hint_y = " (default)" if default else ""
    def_hint_n = " (default)" if not default else ""
    out.write(
        f"\n{sep_color}{'─' * 50}{reset}\n"
        f"  {bold}{prompt_text}{reset}\n"
        f"  {dim} y {reset} approve{def_hint_y}"
        f"   {dim} n {reset} reject{def_hint_n}"
        f"   {dim} esc {reset} cancel\n"
    )
    out.flush()

    while True:
        try:
            ch = _read_single_key()
        except (EOFError, KeyboardInterrupt):
            out.write(f"  {dim}> Cancelled{reset}\n")
            out.flush()
            return False
        key = ch.lower()
        if key == "y":
            out.write(f"  {dim}> Approved{reset}\n")
            out.flush()
            return True
        elif key == "n":
            out.write(f"  {dim}> Rejected{reset}\n")
            out.flush()
            return False
        elif key == "\x1b":  # Esc
            out.write(f"  {dim}> Cancelled{reset}\n")
            out.flush()
            return False
        elif key == "\r":  # Enter → use default
            label = "Approved" if default else "Rejected"
            out.write(f"  {dim}> {label}{reset}\n")
            out.flush()
            return default
        elif key == "\x03":  # Ctrl+C
            out.write(f"  {dim}> Cancelled{reset}\n")
            out.flush()
            return False
        # Ignore any other key (including empty from special keys)


def _toolbar_confirm(prompt_text: str, choices: list[dict] | None = None, default_index: int = 0):
    """Ask via prompt_toolkit arrow-key selector.

    Creates a Future, sets _pending_approval with choices, and blocks until resolved.
    Returns the ``value`` of the chosen option, or False on timeout/cancel.

    If *choices* is None, generates a simple Yes/No list.
    """
    global _pending_approval
    if choices is None:
        choices = [
            {"label": "Yes", "value": True},
            {"label": "No", "value": False},
        ]
    default_index = max(0, min(default_index, len(choices) - 1))
    future = concurrent.futures.Future()
    _pending_approval = {
        "prompt": prompt_text,
        "choices": choices,
        "selected": default_index,
        "future": future,
    }
    app = _toolbar_app_ref
    if app:
        try:
            app.invalidate()
        except Exception:
            pass

    try:
        result = future.result(timeout=300)
    except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
        result = False
    finally:
        _pending_approval = None
        if app:
            try:
                app.invalidate()
            except Exception:
                pass

    # Feedback via Rich console (goes through patch_stdout, no rendering fights)
    if result is True:
        console.print("  [dim]> Approved[/dim]")
    elif result is False:
        console.print("  [dim]> Rejected[/dim]")
    else:
        console.print(f"  [dim]> {result}[/dim]")
    return result


def _terminal_confirm(prompt_text: str, default: bool = True) -> bool:
    """Ask y/n — routes to toolbar selector or inline depending on REPL state."""
    clean = re.sub(r'\[/?[^\]]*\]', '', prompt_text)
    app = _toolbar_app_ref
    if app and getattr(app, 'is_running', False) and not _is_prompt_suspended:
        result = _toolbar_confirm(clean, default_index=0 if default else 1)
        return result is True  # only True counts as approval
    return _inline_confirm(clean, default)


def _auto_approve(prompt: str, default: bool = True) -> bool:
    """Auto-approve in trust mode or agent mode, otherwise prompt with trust option."""
    if state.trust_mode or state.agent_mode > 0:
        return True
    app = _toolbar_app_ref
    if app and getattr(app, 'is_running', False) and not _is_prompt_suspended:
        choices = [
            {"label": "Yes", "value": True},
            {"label": "Yes, allow all this session", "value": "trust"},
            {"label": "No", "value": False},
        ]
        result = _toolbar_confirm(
            re.sub(r'\[/?[^\]]*\]', '', prompt),
            choices=choices,
            default_index=0 if default else 2,
        )
        if result == "trust":
            state.trust_mode = True
            return True
        return result is True
    return _terminal_confirm(prompt, default)


# -- Auto-Checkpoint Constants --
AUTO_CHECKPOINT_THRESHOLD = 5
MAX_EDIT_HISTORY = 20


# -- API retry --
MAX_API_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]


async def _api_call_with_retry(call_fn, label: str = "API call"):
    last_error = None
    for attempt in range(MAX_API_RETRIES):
        try:
            return await call_fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if any(k in err_str for k in ("401", "403", "invalid_api_key", "authentication")):
                raise
            if attempt < MAX_API_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                try:
                    console.print(f"  [swarm.warning]WARNING: {label} failed (attempt {attempt + 1}/{MAX_API_RETRIES}): {type(e).__name__}: {str(e)[:100]}[/swarm.warning]")
                    console.print(f"  [swarm.dim]  retrying in {wait}s...[/swarm.dim]")
                except UnicodeEncodeError:
                    pass
                await asyncio.sleep(wait)
            else:
                try:
                    console.print(f"  [swarm.error]FAILED: {label} failed after {MAX_API_RETRIES} attempts: {e}[/swarm.error]")
                except UnicodeEncodeError:
                    pass
                raise
    raise last_error
