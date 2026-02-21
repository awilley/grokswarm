import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text
from rich.rule import Rule
from rich.layout import Layout
from rich.theme import Theme
from typer import Context
import os
from dotenv import load_dotenv
from pathlib import Path
import yaml
import json
from datetime import datetime
import subprocess
import re
import ast
import fnmatch
import shutil
import base64
import atexit
import time
import sys
import sqlite3
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

load_dotenv()

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

XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    console.print("[swarm.error]Error: XAI_API_KEY not found in .env[/swarm.error]")
    raise typer.Exit(1)

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
)

VERSION = "0.25.0"
MODEL = "grok-4-1-fast-reasoning"
BASE_URL = "https://api.x.ai/v1"
MAX_TOKENS = 16384  # default max output tokens per response


# -- L3: Centralized session state (prevents global-declaration bugs) --
@dataclass
class SwarmState:
    """All mutable session state in one place."""
    trust_mode: bool = False
    read_only: bool = False
    self_improve_active: bool = False
    edit_history: list = field(default_factory=list)  # (path, content|None) tuples
    pending_write_count: int = 0
    last_edited_file: str | None = None
    test_fix_state: dict = field(default_factory=lambda: {"cmd": None, "attempts": 0})

    def reset_project_state(self):
        """Clear cross-project state when switching projects."""
        self.edit_history.clear()
        self.test_fix_state["cmd"] = None
        self.test_fix_state["attempts"] = 0
        self.pending_write_count = 0

state = SwarmState()


# GrokSwarm's own install directory (read-only self-knowledge when working on other projects)
GROKSWARM_HOME = Path(os.environ.get("GROKSWARM_HOME", Path(__file__).resolve().parent)).resolve()

BASE_SYSTEM_PROMPT = """You are Grok Swarm, an expert AI assistant for Aaron. You are concise, direct, and helpful.

FILESYSTEM TOOLS:
- list_directory: List files and subdirectories with sizes.
- read_file: Read file contents (supports start_line/end_line for partial reads).
- write_file: Create or overwrite a file with full content. Use for NEW files.
- edit_file: Surgically edit a file by replacing specific text. Use for MODIFYING existing files. ALWAYS prefer this over write_file for edits.
- search_files: Find files by name.
- grep_files: Search inside file contents for text patterns (like grep).
- run_shell: Execute a shell command.

When modifying existing files, ALWAYS use edit_file instead of write_file -- it's safer and preserves unchanged content.
For multiple changes in the same file, use edit_file with the "edits" array parameter to apply all changes in one call (faster, atomic).
When creating new files, use write_file. When you need to inspect code, use read_file with line ranges for large files.

GIT TOOLS: git_status, git_diff, git_log, git_commit, git_checkout, git_branch, git_show_file, git_blame, git_stash, git_init.
- BEFORE making risky file changes, use git_commit to create a checkpoint.
- The system tracks consecutive file mutations. After 5+ edits without a commit, you'll see an [AUTO-CHECKPOINT] reminder. Take the hint and commit.
- After edits, use git_status and git_diff to verify changes.
- Use git_checkout to undo changes to specific files.
- Use git_branch to create feature branches for experimental work.
- Use git_show_file to view a file at a specific commit or branch.
- Use git_blame to see who changed each line and when.
- Use git_stash to temporarily save/restore work-in-progress (push/pop/list/drop).
- Use git_init to initialize a new repo for projects without one.

SELF-EXTENSION TOOLS:
- list_registry: See existing experts and skills.
- create_expert / create_skill: Propose new agents or skills (user must approve).
When a task would benefit from a specialist that doesn't exist yet, proactively propose creating one.

SEARCH TOOLS (xAI server-side, fast and reliable):
- web_search: Search the web in real-time. Returns summarized results with source URLs. Use for current events, documentation, facts, research.
- x_search: Search X (Twitter) posts in real-time. Returns summarized posts with links. Use for opinions, trending topics, social media sentiment.
Prefer web_search/x_search over fetch_page when you need to FIND information. Use fetch_page only when you need the full content of a SPECIFIC URL.

BROWSER TOOLS (Playwright, for direct page control):
- fetch_page: Read the full text content of a specific URL.
- screenshot_page: Capture visual snapshots (saved to project).
- extract_links: Discover links on a page.

TESTING TOOLS:
- run_tests: Auto-detect and run project tests (pytest, jest, go test, etc). Always run tests after code changes to verify correctness. If tests fail, analyze the output and fix the code.

CODE INTELLIGENCE TOOLS:
- find_symbol: Find where a symbol (class, function, variable) is defined across the project. Uses AST for Python, regex for other languages. Returns file:line for each definition.
- find_references: Find all files that import or reference a given module or symbol. Useful for understanding dependencies and impact of changes.

TEST-FIX CYCLE (automatic):
- When run_tests fails, the system enters a test-fix cycle and remembers the test command.
- After each successful edit_file, the system AUTOMATICALLY re-runs the failed tests to verify your fix.
- You will see [AUTO-RETEST PASSED] or [AUTO-RETEST FAILED] in the tool result.
- If [AUTO-RETEST FAILED]: analyze the new error output and make another edit. The system will auto-retest again (up to 3 times).
- If [AUTO-RETEST PASSED]: the cycle ends and you may proceed to other tasks.
- This means you do NOT need to call run_tests after fixing a test failure — the system does it for you.
- Focus on reading the error output carefully and making precise, targeted fixes.

ERROR RECOVERY:
- After edit_file or write_file, the system automatically checks for syntax errors. If lint errors are reported, you MUST fix them immediately using edit_file before doing anything else.
- If a tool call fails (e.g. edit_file can't find old_text), read the error message carefully, use read_file to inspect the current file state, and retry with corrected arguments.
- Never leave a file in a broken state. If your edit introduced errors, fix them before responding to the user.

Format responses in markdown when appropriate. Be specific and actionable."""

# -- Project Context Awareness --
CONTEXT_FILES = [
    "README.md", "readme.md", "README.rst",
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "Cargo.toml", "go.mod", "Makefile",
    "docker-compose.yml", "Dockerfile",
    ".env.example", "GOALS.md",
]
MAX_CONTEXT_FILE_SIZE = 8000  # chars per file
MAX_TREE_DEPTH = 4
MAX_TREE_FILES = 150
MAX_SCAN_FILES = 500  # max files to inspect during context scan
IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info", "chroma_db", ".next", "target",
}
_IGNORE_PATTERNS = [p for p in IGNORE_DIRS if "*" in p]
_IGNORE_LITERALS = IGNORE_DIRS - set(_IGNORE_PATTERNS)

def _should_ignore(name: str) -> bool:
    """Check if a directory name matches IGNORE_DIRS (literal or glob pattern)."""
    if name in _IGNORE_LITERALS:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in _IGNORE_PATTERNS)

CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb"}


def _iter_project_files(project_dir: Path, extensions: set[str] | None = None, max_files: int = MAX_SCAN_FILES):
    """Walk project files, pruning ignored/hidden dirs at traversal time. Yields (path, rel_path)."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(project_dir):
        # Prune ignored and hidden directories in-place so os.walk never descends into them
        dirnames[:] = sorted([d for d in dirnames if not _should_ignore(d) and not d.startswith(".")])
        for fname in filenames:
            if extensions:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in extensions:
                    continue
            count += 1
            if count > max_files:
                return
            p = Path(dirpath) / fname
            rel = p.relative_to(project_dir)
            yield p, rel


# S4: Patterns for secret redaction in session saves
_SECRET_PATTERNS = [
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),
    re.compile(r'xai-[a-zA-Z0-9]{20,}'),
    re.compile(r'-----BEGIN[A-Z ]*PRIVATE KEY-----'),
    re.compile(r'eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}'),
    re.compile(r'(?i)(?:api[_-]?key|secret|token|password)\s*[=:]\s*["\']?[a-zA-Z0-9_/+=-]{16,}'),
    re.compile(r'(?i)bearer\s+[a-zA-Z0-9_./-]{20,}'),
]

def _redact_secrets(text: str) -> str:
    """Replace sensitive patterns with [REDACTED]."""
    for pat in _SECRET_PATTERNS:
        text = pat.sub('[REDACTED]', text)
    return text


def _detect_language_stats(project_dir: Path) -> dict[str, int]:
    """Count files by language extension."""
    stats: dict[str, int] = {}
    for p, rel in _iter_project_files(project_dir, max_files=MAX_SCAN_FILES):
        ext = p.suffix.lower()
        if ext:
            stats[ext] = stats.get(ext, 0) + 1
    return dict(sorted(stats.items(), key=lambda x: -x[1])[:15])


def _scan_tree(root: Path, prefix: str = "", depth: int = 0) -> list[str]:
    """Build a compact directory tree string."""
    if depth > MAX_TREE_DEPTH:
        return []
    lines = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return []
    dirs = [e for e in entries if e.is_dir() and not _should_ignore(e.name) and not e.name.startswith(".")]
    files = [e for e in entries if e.is_file()]
    items = dirs + files
    for i, entry in enumerate(items):
        if len(lines) > MAX_TREE_FILES:
            lines.append(f"{prefix}... (truncated)")
            break
        is_last = i == len(items) - 1
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            extension = "    " if is_last else "\u2502   "
            lines.extend(_scan_tree(entry, prefix + extension, depth + 1))
        else:
            lines.append(f"{prefix}{connector}{entry.name}")
    return lines

def _read_context_file(path: Path) -> str | None:
    """Read a context file, truncating if too large."""
    if not path.exists() or not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > MAX_CONTEXT_FILE_SIZE:
            content = content[:MAX_CONTEXT_FILE_SIZE] + "\n... (truncated)"
        return content
    except Exception:
        return None


# -- Code Intelligence (AST-powered) --

def _build_python_symbol_index(file_path: Path) -> list[dict]:
    """Parse a Python file with ast and extract all symbols with full detail."""
    symbols = []
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        console.print(f"  [swarm.dim]AST: syntax error in {file_path.name}: {e.msg} line {e.lineno}[/swarm.dim]")
        return symbols
    except Exception:
        return symbols

    # Collect IDs of functions that are methods (direct children of a ClassDef)
    method_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_ids.add(id(item))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            symbols.append({
                "name": node.name, "type": "class", "line": node.lineno,
                "methods": methods,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip methods (they're inside ClassDef)
            if id(node) not in method_ids:
                args = [a.arg for a in node.args.args if a.arg != "self"]
                symbols.append({
                    "name": node.name, "type": "function", "line": node.lineno,
                    "args": args[:8],
                })
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.col_offset == 0:
                    symbols.append({
                        "name": target.id, "type": "variable", "line": node.lineno,
                    })
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.col_offset == 0:
                symbols.append({
                    "name": node.target.id, "type": "variable", "line": node.lineno,
                })
    return symbols


def _extract_python_imports(file_path: Path) -> list[str]:
    """Extract all import targets from a Python file using AST."""
    imports = []
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, Exception):
        return imports
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)
    return imports


# Regex patterns for symbol definitions in non-Python languages
_SYMBOL_PATTERNS = {
    ".js": re.compile(r"^(?:export\s+)?(?:default\s+)?(?:(?:async\s+)?function\s+(\w+)|class\s+(\w+)|(?:const|let|var)\s+(\w+))"),
    ".jsx": re.compile(r"^(?:export\s+)?(?:default\s+)?(?:(?:async\s+)?function\s+(\w+)|class\s+(\w+)|(?:const|let|var)\s+(\w+))"),
    ".ts": re.compile(r"^(?:export\s+)?(?:(?:async\s+)?function\s+(\w+)|class\s+(\w+)|interface\s+(\w+)|type\s+(\w+)|enum\s+(\w+)|(?:const|let|var)\s+(\w+))"),
    ".tsx": re.compile(r"^(?:export\s+)?(?:(?:async\s+)?function\s+(\w+)|class\s+(\w+)|interface\s+(\w+)|type\s+(\w+)|enum\s+(\w+)|(?:const|let|var)\s+(\w+))"),
    ".go": re.compile(r"^(?:func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)|type\s+(\w+))"),
    ".rs": re.compile(r"^(?:pub\s+)?(?:fn\s+(\w+)|struct\s+(\w+)|enum\s+(\w+)|trait\s+(\w+)|impl\s+(\w+))"),
    ".java": re.compile(r"^(?:public|private|protected)?\s*(?:static\s+)?(?:class\s+(\w+)|interface\s+(\w+)|enum\s+(\w+))"),
    ".rb": re.compile(r"^(?:class\s+(\w+)|def\s+(\w+)|module\s+(\w+))"),
}


def _find_symbol_in_file(file_path: Path, symbol_name: str, rel: str) -> list[dict]:
    """Find definitions of symbol_name in a single file. Works for all supported languages."""
    results = []
    ext = file_path.suffix.lower()

    if ext == ".py":
        for sym in _build_python_symbol_index(file_path):
            if sym["name"] == symbol_name:
                results.append({"file": rel, "line": sym["line"], "type": sym["type"],
                                **({"methods": sym["methods"]} if sym.get("methods") else {}),
                                **({"args": sym["args"]} if sym.get("args") else {})})
    else:
        pattern = _SYMBOL_PATTERNS.get(ext)
        if not pattern:
            return results
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return results
        for i, line in enumerate(text.splitlines(), 1):
            m = pattern.match(line.rstrip())
            if m:
                # Pattern has multiple groups; the first non-None is the name
                name = next((g for g in m.groups() if g), None)
                if name == symbol_name:
                    results.append({"file": rel, "line": i, "type": "definition"})
    return results


def find_symbol(name: str) -> str:
    """Find where a symbol is defined across the project."""
    results = []
    for p, rel in _iter_project_files(PROJECT_DIR, extensions=CODE_EXTENSIONS):
        hits = _find_symbol_in_file(p, name, str(rel))
        results.extend(hits)
        if len(results) >= 50:
            break
    if not results:
        return f"No definitions found for '{name}'."
    lines = []
    for r in results:
        detail = f"  {r['file']}:{r['line']} ({r['type']})"
        if r.get("methods"):
            detail += f" methods: {', '.join(r['methods'][:10])}"
        if r.get("args"):
            detail += f" args: ({', '.join(r['args'])})"
        lines.append(detail)
    return f"Found {len(results)} definition(s) for '{name}':\n" + "\n".join(lines)


def find_references(name: str) -> str:
    """Find all files that import or reference a given module/symbol."""
    results = []
    name_lower = name.lower()
    for p, rel in _iter_project_files(PROJECT_DIR, extensions=CODE_EXTENSIONS):
        ext = p.suffix.lower()
        if ext == ".py":
            imports = _extract_python_imports(p)
            # Check if any import matches (module or submodule)
            for imp in imports:
                if name_lower in imp.lower().split("."):
                    results.append(f"  {rel} (import {imp})")
                    break
            else:
                # Also check for direct usage via grep
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(text.splitlines(), 1):
                        if re.search(r'\b' + re.escape(name) + r'\b', line):
                            results.append(f"  {rel}:{i}: {line.rstrip()[:100]}")
                            break  # one hit per file
                except Exception:
                    pass
        else:
            # Non-Python: simple word-boundary grep
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    if re.search(r'\b' + re.escape(name) + r'\b', line):
                        results.append(f"  {rel}:{i}: {line.rstrip()[:100]}")
                        break
            except Exception:
                pass
        if len(results) >= 50:
            results.append("  ... (max 50 results)")
            break
    if not results:
        return f"No references found for '{name}'."
    return f"Found {len(results)} reference(s) for '{name}':\n" + "\n".join(results)


def _build_import_graph(project_dir: Path) -> dict[str, list[str]]:
    """Build a project-level import dependency graph for Python files."""
    graph: dict[str, list[str]] = {}
    py_modules: set[str] = set()
    # Collect all files once (reuse list for both passes)
    py_files: list[tuple[Path, Path]] = list(_iter_project_files(project_dir, extensions={".py"}))
    # First pass: collect all local module names
    for p, rel in py_files:
        module_name = str(rel.with_suffix("")).replace(os.sep, ".")
        py_modules.add(module_name)
        for part in rel.parent.parts:
            py_modules.add(part)
    # Second pass: build the graph
    for p, rel in py_files:
        imports = _extract_python_imports(p)
        local_deps = []
        for imp in imports:
            root_module = imp.split(".")[0]
            if root_module in py_modules or imp in py_modules:
                local_deps.append(imp)
        if local_deps:
            graph[str(rel)] = sorted(set(local_deps))
    return graph


def _build_deep_symbol_index(project_dir: Path) -> dict[str, list[str]]:
    """Build a rich symbol index: classes with methods, functions with args, variables."""
    index: dict[str, list[str]] = {}
    for p, rel in _iter_project_files(project_dir, extensions=CODE_EXTENSIONS):
        ext = p.suffix.lower()
        defs = []
        if ext == ".py":
            for sym in _build_python_symbol_index(p):
                entry = f"  L{sym['line']}: {sym['type']} {sym['name']}"
                if sym.get("methods"):
                    entry += f" [{', '.join(sym['methods'][:8])}]"
                if sym.get("args"):
                    entry += f"({', '.join(sym['args'])})"
                defs.append(entry)
                if len(defs) >= 15:
                    defs.append("  ... (truncated)")
                    break
        else:
            pattern = _SYMBOL_PATTERNS.get(ext)
            if not pattern:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                m = pattern.match(line.rstrip())
                if m:
                    name = next((g for g in m.groups() if g), None)
                    if name:
                        defs.append(f"  L{i}: {name}")
                if len(defs) >= 15:
                    defs.append("  ... (truncated)")
                    break
        if defs:
            index[str(rel)] = defs
    return index


def scan_project_context(project_dir: Path) -> dict:
    """Scan a directory and build project context."""
    context = {
        "project_dir": str(project_dir),
        "project_name": project_dir.name,
        "tree": "",
        "key_files": {},
        "code_structure": {},
        "language_stats": {},
    }
    # Build tree
    tree_lines = _scan_tree(project_dir)
    context["tree"] = f"{project_dir.name}/\n" + "\n".join(tree_lines) if tree_lines else "(empty)"

    # Read key files
    for fname in CONTEXT_FILES:
        fpath = project_dir / fname
        content = _read_context_file(fpath)
        if content:
            context["key_files"][fname] = content

    # Scan code structure (classes, functions) -- use deep AST index for Python
    context["code_structure"] = _build_deep_symbol_index(project_dir)

    # Language stats
    context["language_stats"] = _detect_language_stats(project_dir)

    # Import dependency graph (Python)
    context["import_graph"] = _build_import_graph(project_dir)

    return context


def _context_cache_key(project_dir: Path) -> Path:
    """Get the cache file path for a project based on its resolved path."""
    import hashlib
    h = hashlib.sha256(str(project_dir.resolve()).encode()).hexdigest()[:16]
    return CONTEXT_CACHE_DIR / f"{project_dir.name}_{h}.json"


def _project_mtime(project_dir: Path) -> float:
    """Get the most recent mtime of tracked files in the project (fast check)."""
    latest = 0.0
    try:
        for p, _ in _iter_project_files(project_dir, max_files=10000):
            try:
                mt = p.stat().st_mtime
                if mt > latest:
                    latest = mt
            except OSError:
                pass
    except Exception:
        pass
    return latest


def _load_cached_context(project_dir: Path) -> dict | None:
    """Perf-P1: Load context from cache if project hasn't changed."""
    cache_file = _context_cache_key(project_dir)
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        cached_mtime = data.get("_mtime", 0)
        if _project_mtime(project_dir) <= cached_mtime:
            data.pop("_mtime", None)
            return data
    except Exception:
        pass
    return None


def _save_context_cache(project_dir: Path, context: dict):
    """Save project context to cache with mtime stamp."""
    try:
        data = dict(context)
        data["_mtime"] = _project_mtime(project_dir)
        _context_cache_key(project_dir).write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def scan_project_context_cached(project_dir: Path) -> dict:
    """Scan project context, using cache when nothing has changed."""
    cached = _load_cached_context(project_dir)
    if cached:
        console.print("[swarm.dim]  context loaded from cache[/swarm.dim]")
        return cached
    context = scan_project_context(project_dir)
    _save_context_cache(project_dir, context)
    return context

def format_context_for_prompt(ctx: dict) -> str:
    """Format project context as a system prompt section."""
    parts = [f"\n--- PROJECT CONTEXT ---"]
    parts.append(f"Project: {ctx['project_name']}")
    parts.append(f"Directory: {ctx['project_dir']}")

    # Language stats
    lang_stats = ctx.get("language_stats", {})
    if lang_stats:
        stats_str = ", ".join(f"{ext}: {count}" for ext, count in list(lang_stats.items())[:10])
        parts.append(f"\nLanguages: {stats_str}")

    parts.append(f"\nFile tree:\n```\n{ctx['tree']}\n```")

    for fname, content in ctx["key_files"].items():
        parts.append(f"\n{fname}:\n```\n{content}\n```")

    # Code structure map (classes, functions, methods)
    code_struct = ctx.get("code_structure", {})
    if code_struct:
        parts.append("\nCode structure (symbols):")
        for filepath, defs in list(code_struct.items())[:20]:
            parts.append(f"\n{filepath}:")
            parts.extend(defs)

    # Import dependency graph
    import_graph = ctx.get("import_graph", {})
    if import_graph:
        parts.append("\nImport graph (local dependencies):")
        for filepath, deps in list(import_graph.items())[:30]:
            parts.append(f"  {filepath} -> {', '.join(deps)}")

    parts.append("--- END PROJECT CONTEXT ---")
    return "\n".join(parts)

def build_system_prompt(project_context: dict | None = None) -> str:
    """Build the full system prompt with optional project context."""
    prompt = BASE_SYSTEM_PROMPT
    if project_context:
        prompt += "\n" + format_context_for_prompt(project_context)
    # Add self-knowledge section when working on an external project
    if PROJECT_DIR.resolve() != GROKSWARM_HOME:
        prompt += ("\n\n--- SELF-KNOWLEDGE ---\n"
                   "You are GrokSwarm. Your own source code is accessible read-only via the @grokswarm/ prefix.\n"
                   "  - read_file with path '@grokswarm/main.py' to read your own source\n"
                   "  - list_directory with path '@grokswarm/' to see your own files\n"
                   "Use this for self-reference — understanding your own capabilities, reviewing your implementation, "
                   "or when asked about how you work. All @grokswarm/ access is read-only.\n"
                   "--- END SELF-KNOWLEDGE ---")
    return prompt

# Detect project on startup from cwd (re-scanned in main() if --project-dir given)
PROJECT_DIR = Path.cwd().resolve()
PROJECT_CONTEXT: dict = {}
SYSTEM_PROMPT: str = ""


def _incremental_context_refresh(file_path_str: str):
    """Cheaply refresh project context for one edited file (symbols + imports)."""
    global SYSTEM_PROMPT
    try:
        fp = PROJECT_DIR / file_path_str
        if not fp.is_file():
            return
        rel = str(fp.relative_to(PROJECT_DIR))
        if fp.suffix.lower() == ".py":
            syms = _build_python_symbol_index(fp)
            defs = []
            for s in syms:
                e = f"  L{s['line']}: {s['type']} {s['name']}"
                if s.get("methods"):
                    e += f" [{', '.join(s['methods'][:8])}]"
                if s.get("args"):
                    e += f"({', '.join(s['args'])})"
                defs.append(e)
                if len(defs) >= 15:
                    defs.append("  ... (truncated)")
                    break
            cs = PROJECT_CONTEXT.setdefault("code_structure", {})
            if defs:
                cs[rel] = defs
            elif rel in cs:
                del cs[rel]
        SYSTEM_PROMPT = build_system_prompt(PROJECT_CONTEXT)
    except Exception:
        pass

def _safe_path(path: str) -> Path | None:
    """Resolve a path and verify it's inside the project directory (symlink-safe)."""
    candidate = PROJECT_DIR / path
    try:
        # S2: walk each component — reject any symlink in the chain inside the project
        cur = PROJECT_DIR
        for part in candidate.relative_to(PROJECT_DIR).parts:
            cur = cur / part
            if cur.is_symlink():
                # Symlink found — only allow if its target stays inside the project
                real_target = cur.resolve()
                if not real_target.is_relative_to(PROJECT_DIR.resolve()):
                    return None
        full = candidate.resolve()
    except (ValueError, OSError):
        return None
    if not full.is_relative_to(PROJECT_DIR.resolve()):
        return None
    return full

def _grokswarm_read_path(path: str) -> Path | None:
    """Resolve a @grokswarm/ prefixed path for read-only access to GrokSwarm's own source."""
    rel = path[len("@grokswarm/"):]
    candidate = (GROKSWARM_HOME / rel).resolve()
    if not candidate.is_relative_to(GROKSWARM_HOME):
        return None
    return candidate

# -- Tools --
def list_dir(path: str = "."):
    # Allow read-only listing of @grokswarm/ paths
    if path.startswith("@grokswarm/"):
        full_path = _grokswarm_read_path(path)
        if not full_path:
            return "Access denied: outside grokswarm directory."
    else:
        full_path = _safe_path(path)
        if not full_path:
            return "Access denied: outside project directory."
    if not full_path.exists():
        return "Path not found."
    entries = sorted(full_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = []
    for p in entries:
        if p.is_dir():
            lines.append(f"  {p.name}/")
        else:
            size = p.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024*1024):.1f} MB"
            lines.append(f"  {p.name:<35} {size_str}")
    return "\n".join(lines) if lines else "(empty directory)"

def read_file(path: str, start_line: int | None = None, end_line: int | None = None):
    # Allow read-only access to @grokswarm/ paths
    if path.startswith("@grokswarm/"):
        full_path = _grokswarm_read_path(path)
        if not full_path:
            return "Access denied: outside grokswarm directory."
        if not (full_path.exists() and full_path.is_file()):
            return "File not found in grokswarm directory."
    else:
        full_path = _safe_path(path)
        if not full_path:
            return "Access denied: outside project directory."
        if not (full_path.exists() and full_path.is_file()):
            return "File not found."
    # C6: warn on large files when no line range specified
    if start_line is None and end_line is None:
        try:
            fsize = full_path.stat().st_size
            if fsize > 1_048_576:
                return f"Warning: {path} is {fsize:,} bytes ({fsize / 1048576:.1f} MB). Use read_file with start_line/end_line for partial reads, or grep_files to search for specific content."
        except OSError:
            pass
    text = full_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    total = len(lines)
    if start_line is not None or end_line is not None:
        s = max(1, start_line or 1)
        e = min(total, end_line or total)
        selected = lines[s-1:e]
        numbered = [f"{i:>4} | {line}" for i, line in enumerate(selected, s)]
        header = f"[{path} lines {s}-{e} of {total}]"
        return header + "\n" + "\n".join(numbered)
    return text

def write_file(path: str, content: str):
    full_path = _safe_path(path)
    if not full_path:
        return "Access denied: outside project directory."
    console.print(f"[bold yellow]About to WRITE to:[/bold yellow] {full_path}")
    preview = content[:300]
    if len(content) > 300:
        preview += "..."
    console.print(f"[dim]Preview:[/dim]\n{preview}")
    if _auto_approve("Approve write?", default=False):
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"Written: {full_path}"
    return "Cancelled."


def _apply_single_edit(full_path: Path, content: str, old_text: str, new_text: str, show_preview: bool = True) -> tuple[str | None, str]:
    """Apply one search/replace to content. Returns (new_content, message) or (None, error)."""
    count = content.count(old_text)
    if count == 0:
        return None, f"Error: old_text not found in {full_path.name}. The text to replace must match exactly (including whitespace and indentation). Use read_file to verify the current content."
    if count > 1:
        return None, f"Error: old_text found {count} times in {full_path.name}. Include more surrounding context to make it unique."
    if show_preview:
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()
        console.print(f"[dim]  Replacing {len(old_lines)} line(s) with {len(new_lines)} line(s):[/dim]")
        for line in old_lines[:8]:
            console.print(f"    [red]- {line}[/red]")
        if len(old_lines) > 8:
            console.print(f"    [dim]  ... ({len(old_lines) - 8} more lines)[/dim]")
        for line in new_lines[:8]:
            console.print(f"    [green]+ {line}[/green]")
        if len(new_lines) > 8:
            console.print(f"    [dim]  ... ({len(new_lines) - 8} more lines)[/dim]")
    return content.replace(old_text, new_text, 1), f"{len(old_text.splitlines())} lines -> {len(new_text.splitlines())} lines"


def edit_file(path: str, old_text: str = "", new_text: str = "", edits: list | None = None) -> str:
    """Surgically edit a file. Supports single edit (old_text/new_text) or multi-edit (edits array)."""
    full_path = _safe_path(path)
    if not full_path:
        return "Access denied: outside project directory."
    if not full_path.exists() or not full_path.is_file():
        return f"File not found: {path}"
    content = full_path.read_text(encoding="utf-8", errors="ignore")

    # Normalize: if edits array provided, use that; otherwise use single old_text/new_text
    edit_list = edits if edits else [{"old_text": old_text, "new_text": new_text}]
    if not edit_list or (len(edit_list) == 1 and not edit_list[0].get("old_text")):
        return "Error: no edits provided. Supply old_text/new_text or an edits array."

    console.print(f"[bold yellow]About to EDIT:[/bold yellow] {full_path} ({len(edit_list)} edit{'s' if len(edit_list) != 1 else ''})")

    # Validate all edits before applying any
    for i, edit in enumerate(edit_list):
        ot = edit.get("old_text", "")
        if not ot:
            return f"Error: edit #{i+1} has empty old_text."
        cnt = content.count(ot)
        if cnt == 0:
            return f"Error: edit #{i+1} old_text not found in {path}. Use read_file to verify current content."
        if cnt > 1:
            return f"Error: edit #{i+1} old_text found {cnt} times in {path}. Include more surrounding context to make it unique."

    # Show previews
    for i, edit in enumerate(edit_list):
        if len(edit_list) > 1:
            console.print(f"  [swarm.accent]edit {i+1}/{len(edit_list)}:[/swarm.accent]")
        _apply_single_edit(full_path, content, edit["old_text"], edit.get("new_text", ""), show_preview=True)

    if _auto_approve("Approve edit?"):
        # Apply all edits sequentially
        results = []
        for i, edit in enumerate(edit_list):
            new_content, msg = _apply_single_edit(full_path, content, edit["old_text"], edit.get("new_text", ""), show_preview=False)
            if new_content is None:
                return f"Error applying edit #{i+1}: {msg}"
            content = new_content
            results.append(msg)
        full_path.write_text(content, encoding="utf-8")
        summary = "; ".join(results)
        return f"Edited: {full_path} ({len(edit_list)} edit{'s' if len(edit_list) != 1 else ''}: {summary})"
    return "Edit cancelled."


def search_files(query: str):
    results = []
    for p, rel in _iter_project_files(PROJECT_DIR):
        if query.lower() in p.name.lower():
            results.append(f"  {rel}")
    # Also check directories via os.walk
    for dirpath, dirnames, _ in os.walk(PROJECT_DIR):
        dirnames[:] = [d for d in dirnames if not _should_ignore(d) and not d.startswith(".")]
        dp = Path(dirpath)
        if query.lower() in dp.name.lower() and dp != PROJECT_DIR:
            results.append(f"  {dp.relative_to(PROJECT_DIR)}/")
    return "\n".join(results) if results else "No matches found."


def grep_files(pattern: str, path: str = ".", is_regex: bool = False, context_lines: int = 0) -> str:
    """Search file contents for a text pattern. Returns matching lines with file:line references.
    Supports regex mode and context lines (like grep -C)."""
    search_root = _safe_path(path)
    if not search_root:
        return "Access denied: outside project directory."
    if not search_root.exists():
        return "Path not found."

    # Perf-P2: try ripgrep first (much faster on large repos)
    if shutil.which("rg"):
        try:
            rg_args = ["rg", "--no-heading", "--line-number", "--color=never",
                       "-i", "--max-count=200"]
            if context_lines:
                rg_args += [f"-C{min(context_lines, 10)}"]
            if not is_regex:
                rg_args.append("--fixed-strings")
            rg_args += ["--", pattern, str(search_root)]
            rg = subprocess.run(rg_args, capture_output=True, text=True, cwd=PROJECT_DIR, timeout=15)
            if rg.returncode in (0, 1):  # 0=matches, 1=no matches
                rg_out = rg.stdout.strip()
                return rg_out if rg_out else "No matches found."
        except Exception:
            pass  # fall through to Python implementation

    results = []
    # Compile regex if requested
    if is_regex:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex: {e}"
    else:
        search_lower = pattern.lower()
    ctx = min(max(context_lines, 0), 10)  # clamp 0-10
    if search_root.is_dir():
        targets = _iter_project_files(search_root, max_files=2000)
    else:
        targets = [(search_root, search_root.relative_to(PROJECT_DIR))]
    for p, rel in targets:
        # Skip binary files by extension
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2",
                                  ".ttf", ".eot", ".zip", ".tar", ".gz", ".exe", ".dll",
                                  ".so", ".dylib", ".pyc", ".pyo", ".db", ".sqlite"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        file_lines = text.splitlines()
        matched_ranges: set[int] = set()
        for i, line in enumerate(file_lines):
            match = rx.search(line) if is_regex else (search_lower in line.lower())
            if match:
                for offset in range(-ctx, ctx + 1):
                    matched_ranges.add(i + offset)
        if not matched_ranges:
            continue
        prev_idx = -2
        for idx in sorted(matched_ranges):
            if idx < 0 or idx >= len(file_lines):
                continue
            if prev_idx >= 0 and idx > prev_idx + 1:
                results.append("  --")
            line_num = idx + 1
            is_match = (rx.search(file_lines[idx]) if is_regex else (search_lower in file_lines[idx].lower()))
            marker = ":" if is_match else "-"
            results.append(f"  {rel}{marker}{line_num}: {file_lines[idx].rstrip()[:120]}")
            prev_idx = idx
            if len(results) >= 200:
                results.append("  ... (max 200 matches reached)")
                return "\n".join(results)
    return "\n".join(results) if results else "No matches found."

DANGEROUS_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*)?\s*-[a-zA-Z]*r",     # rm -rf, rm -r
    r"\brm\s+-[a-zA-Z]*f",                        # rm -f
    r"\bsudo\b",
    r"\bcurl\b.*\|\s*(ba)?sh",                   # curl | sh, curl | bash
    r"\bwget\b.*\|\s*(ba)?sh",
    r"\bchmod\s+(-[a-zA-Z]*)?\s*-[a-zA-Z]*R",    # chmod -R
    r"\bchown\s+(-[a-zA-Z]*)?\s*-[a-zA-Z]*R",    # chown -R
    r"\bmkfs\b",
    r"\bdd\b\s+if=",
    r":\(\)\{\s*:\|",                             # fork bomb
    r"\b(poweroff|reboot|shutdown)\b",
    r"\bformat\s+[a-zA-Z]:",                       # Windows format
    r"> /dev/(sd|null|zero)",
    r"\bgit\s+push\s+.*--force",
    r"\bgit\s+reset\s+--hard",
]
_DANGEROUS_RX = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]

def _is_dangerous_command(command: str) -> bool:
    return any(rx.search(command) for rx in _DANGEROUS_RX)

def run_shell(command: str):
    console.print(f"[bold yellow]About to EXECUTE:[/bold yellow] {command}")
    console.print(f"[dim]Working directory: {PROJECT_DIR}[/dim]")
    if _is_dangerous_command(command):
        console.print("[bold red][DANGEROUS COMMAND DETECTED][/bold red]")
        console.print(f"[bold red]This command matches a dangerous pattern. Review carefully before approving.[/bold red]")
        if not Confirm.ask("[bold red]CONFIRM dangerous command?[/bold red]", default=False):
            return "Cancelled: dangerous command rejected."
    elif not Confirm.ask("Approve command?", default=False):
        return "Cancelled."
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=PROJECT_DIR, timeout=120)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Error: {e}"


# -- Testing Tools --
TEST_COMMANDS = {
    "pytest": {"detect": ["pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py"], "cmd": "python -m pytest -v"},
    "unittest": {"detect": [], "cmd": "python -m unittest discover -v"},
    "jest": {"detect": ["jest.config.js", "jest.config.ts"], "cmd": "npx jest --verbose"},
    "mocha": {"detect": [".mocharc.yml", ".mocharc.json"], "cmd": "npx mocha"},
    "go": {"detect": ["go.mod"], "cmd": "go test ./... -v"},
    "cargo": {"detect": ["Cargo.toml"], "cmd": "cargo test"},
}


def _detect_test_framework() -> str | None:
    """Auto-detect the test framework from project files."""
    for name, info in TEST_COMMANDS.items():
        for detect_file in info["detect"]:
            if (PROJECT_DIR / detect_file).exists():
                return name
    # Fallback: check if any test_*.py files exist -> pytest
    test_files = [p for p, _ in _iter_project_files(PROJECT_DIR, extensions={".py"}, max_files=200)
                  if p.name.startswith("test_") or p.name.endswith("_test.py")]
    if test_files:
        return "pytest"
    if (PROJECT_DIR / "package.json").exists():
        try:
            pkg = json.loads((PROJECT_DIR / "package.json").read_text())
            if "jest" in pkg.get("devDependencies", {}) or "jest" in pkg.get("dependencies", {}):
                return "jest"
        except Exception:
            pass
    return None


# -- Test-Fix Cycle State --
# Tracks the last failed test command so auto-retest can run after edits
MAX_TEST_FIX_ATTEMPTS = 3  # auto-retest up to 3 times per fix cycle


def _run_tests_raw(command: str, timeout: int = 120) -> str:
    """Run a test command and return formatted output. No approval gate."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=PROJECT_DIR, timeout=timeout
        )
        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"\nSTDOUT:\n{result.stdout}"
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if len(output) > 6000:
            output = output[:6000] + "\n... (truncated)"
        if result.returncode == 0:
            return "[PASS] TESTS PASSED\n" + output
        else:
            return "[FAIL] TESTS FAILED\n" + output
    except subprocess.TimeoutExpired:
        return "[FAIL] TESTS FAILED\nError: test command timed out (120s limit)."
    except Exception as e:
        return f"Error: {e}"


def run_tests(command: str | None = None, pattern: str | None = None) -> str:
    """Run project tests. Auto-detects framework if no command given."""
    if not command:
        framework = _detect_test_framework()
        if not framework:
            return "No test framework detected. Use the 'command' parameter to specify a test command."
        command = TEST_COMMANDS[framework]["cmd"]
        console.print(f"[swarm.accent]Detected: {framework}[/swarm.accent]")
    if pattern:
        command += f" -k {pattern}" if "pytest" in command else f" {pattern}"
    console.print(f"[bold yellow]About to RUN TESTS:[/bold yellow] {command}")
    if _auto_approve("Approve test run?"):
        output = _run_tests_raw(command)
        return output
    return "Test run cancelled."


# -- Auto-Lint After Edits --
LINT_COMMANDS: dict[str, list[str]] = {
    ".py": ["python", "-m", "py_compile"],
    ".js": ["node", "--check"],
    ".ts": ["npx", "tsc", "--noEmit", "--pretty"],
}


def _lint_file(path: Path) -> str | None:
    """Run a syntax check on a file. Returns error string or None if clean."""
    ext = path.suffix.lower()
    base_cmd = LINT_COMMANDS.get(ext)
    if not base_cmd:
        return None
    try:
        if ext == ".py":
            # py_compile writes to __pycache__ so use compile() directly
            import py_compile
            py_compile.compile(str(path), doraise=True)
            return None
        else:
            result = subprocess.run(
                base_cmd + [str(path)],
                capture_output=True, text=True, cwd=PROJECT_DIR, timeout=15
            )
            if result.returncode != 0:
                err = (result.stderr or result.stdout).strip()
                return err[:1500] if err else f"Lint failed (exit {result.returncode})"
            return None
    except py_compile.PyCompileError as e:
        return str(e)[:1500]
    except FileNotFoundError:
        return None  # linter not installed, skip silently
    except subprocess.TimeoutExpired:
        return None  # don't block on slow linters
    except Exception:
        return None


# -- Git Tools --
def _run_git(*args: str) -> str:
    """Run a git command in the project directory and return output."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True, text=True, cwd=PROJECT_DIR, timeout=15
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"Error (exit {result.returncode}): {err}" if err else f"Error (exit {result.returncode})"
        return output if output else "(no output)"
    except FileNotFoundError:
        return "Error: git is not installed or not in PATH."
    except subprocess.TimeoutExpired:
        return "Error: git command timed out."


def git_status() -> str:
    return _run_git("status", "--short", "--branch")


def git_diff(path: str | None = None, staged: bool = False) -> str:
    args = ["diff"]
    if staged:
        args.append("--staged")
    args.append("--stat")  # summary first
    if path:
        safe = _safe_path(path)
        if not safe:
            return "Access denied: outside project directory."
        args.append(path)
    summary = _run_git(*args)
    # Also get the actual diff (truncated)
    detail_args = ["diff"]
    if staged:
        detail_args.append("--staged")
    if path:
        detail_args.append(path)
    detail = _run_git(*detail_args)
    if len(detail) > 3000:
        detail = detail[:3000] + "\n... (truncated)"
    return f"{summary}\n\n{detail}"


def git_log(count: int = 10) -> str:
    count = min(max(count, 1), 50)
    return _run_git("log", f"--oneline", f"-{count}", "--decorate")


def git_commit(message: str) -> str:
    console.print(f"[bold yellow]About to COMMIT:[/bold yellow] {message}")
    # Show what would be committed
    status = _run_git("status", "--short")
    console.print(f"[dim]{status}[/dim]")
    if _auto_approve("Approve commit? (will stage all changes)"):
        _run_git("add", "-A")
        return _run_git("commit", "-m", message)
    return "Commit cancelled by user."


def git_checkout(target: str) -> str:
    """Checkout a file, branch, or commit. Requires approval for destructive ops."""
    console.print(f"[bold yellow]About to CHECKOUT:[/bold yellow] {target}")
    # Determine if it's a file restore or branch switch
    safe = _safe_path(target)
    if safe and safe.exists():
        console.print("[dim]This will discard uncommitted changes to this file.[/dim]")
        if Confirm.ask("Approve file restore?", default=False):
            return _run_git("checkout", "--", target)
        return "Checkout cancelled by user."
    else:
        console.print(f"[dim]Switching to branch/commit: {target}[/dim]")
        if _auto_approve("Approve branch switch?"):
            return _run_git("checkout", target)
        return "Checkout cancelled by user."


def git_branch(name: str | None = None, delete: bool = False) -> str:
    if not name:
        return _run_git("branch", "-a", "--no-color")
    if delete:
        console.print(f"[bold yellow]About to DELETE branch:[/bold yellow] {name}")
        if Confirm.ask("Approve branch deletion?", default=False):
            return _run_git("branch", "-d", name)
        return "Branch deletion cancelled by user."
    return _run_git("branch", name)


def git_show_file(path: str, ref: str = "HEAD") -> str:
    """Show file contents at a specific git ref (commit, branch, tag)."""
    safe = _safe_path(path)
    if not safe:
        return "Access denied: outside project directory."
    return _run_git("show", f"{ref}:{path}")


def git_blame(path: str) -> str:
    """Show git blame for a file -- who changed each line and when."""
    safe = _safe_path(path)
    if not safe:
        return "Access denied: outside project directory."
    result = _run_git("blame", "--date=short", path)
    if len(result) > 5000:
        result = result[:5000] + "\n... (truncated)"
    return result


def git_stash(action: str = "list", message: str | None = None) -> str:
    """Manage git stash: list, push, pop, drop."""
    action = action.lower()
    if action == "list":
        return _run_git("stash", "list") or "(no stashes)"
    elif action == "push":
        console.print(f"[bold yellow]About to STASH changes:[/bold yellow] {message or '(no message)'}")
        if _auto_approve("Approve stash?"):
            args = ["stash", "push"]
            if message:
                args.extend(["-m", message])
            return _run_git(*args)
        return "Stash cancelled by user."
    elif action == "pop":
        console.print("[bold yellow]About to POP stash[/bold yellow] (apply + drop top stash)")
        if _auto_approve("Approve stash pop?"):
            return _run_git("stash", "pop")
        return "Stash pop cancelled by user."
    elif action == "drop":
        console.print("[bold yellow]About to DROP top stash[/bold yellow] (permanently removes it)")
        if Confirm.ask("Approve stash drop?", default=False):
            return _run_git("stash", "drop")
        return "Stash drop cancelled by user."
    else:
        return f"Unknown stash action: '{action}'. Use: list, push, pop, drop."


def git_init() -> str:
    """Initialize a new git repository in the project directory."""
    git_dir = PROJECT_DIR / ".git"
    if git_dir.exists():
        return "Already a git repository."
    console.print("[bold yellow]About to INIT git repo[/bold yellow]")
    if _auto_approve("Initialize a new git repository here?"):
        return _run_git("init")
    return "Git init cancelled by user."


# -- Auto-Checkpoint Constants --
AUTO_CHECKPOINT_THRESHOLD = 5  # suggest checkpoint after this many file mutations
MAX_EDIT_HISTORY = 20  # keep at most this many undo entries


def _auto_approve(prompt: str, default: bool = True) -> bool:
    """Auto-approve in trust mode, otherwise prompt the user."""
    if state.trust_mode:
        console.print(f"[swarm.dim]  (auto-approved: trust mode)[/swarm.dim]")
        return True
    return Confirm.ask(prompt, default=default)


# -- Browser Tools (Playwright) --
_browser_instance = None
_playwright_instance = None


def _get_browser():
    """Lazy-init a headless Chromium browser."""
    global _browser_instance, _playwright_instance
    if _browser_instance is None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            # G2: offer to install playwright on first use
            console.print("[swarm.warning]Playwright is not installed (needed for browser tools).[/swarm.warning]")
            if Confirm.ask("Install playwright + chromium now?", default=True):
                try:
                    subprocess.run(["pip", "install", "playwright"], check=True, timeout=120)
                    subprocess.run(["playwright", "install", "chromium"], check=True, timeout=180)
                    console.print("[swarm.accent]Playwright installed! Retrying...[/swarm.accent]")
                    from playwright.sync_api import sync_playwright
                except Exception as e:
                    console.print(f"[swarm.error]Installation failed: {e}[/swarm.error]")
                    return None
            else:
                return None
        _playwright_instance = sync_playwright().start()
        _browser_instance = _playwright_instance.chromium.launch(headless=True)
    return _browser_instance


def _close_browser():
    """Clean up browser resources."""
    global _browser_instance, _playwright_instance
    if _browser_instance:
        try:
            _browser_instance.close()
        except Exception:
            pass
        _browser_instance = None
    if _playwright_instance:
        try:
            _playwright_instance.stop()
        except Exception:
            pass
        _playwright_instance = None


def _atexit_close_browser():
    """A2: atexit wrapper with hard timeout to prevent hang on exit."""
    import threading
    t = threading.Thread(target=_close_browser, daemon=True)
    t.start()
    t.join(timeout=3)  # give Playwright 3 seconds max


atexit.register(_atexit_close_browser)


# -- Search Tools (xAI server-side) --
def _xai_search(query: str, tool_type: str) -> str:
    """Call xAI Responses API with a built-in search tool and return the text result."""
    try:
        resp = httpx.post(
            f"{BASE_URL.replace('/v1', '')}/v1/responses",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "input": query,
                "tools": [{"type": tool_type}],
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract text from output messages
        parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        parts.append(block["text"])
        text = "\n".join(parts) if parts else "No results found."
        # Append citations
        citations = data.get("citations", [])
        if citations:
            text += "\n\nSources:\n" + "\n".join(f"  - {url}" for url in citations[:15])
        if len(text) > 8000:
            text = text[:8000] + "\n... (truncated)"
        return text
    except httpx.HTTPStatusError as e:
        return f"Error: search API returned {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return f"Error: search failed: {e}"


def web_search(query: str) -> str:
    """Search the web using xAI's server-side web search."""
    return _xai_search(query, "web_search")


def x_search(query: str) -> str:
    """Search X (Twitter) posts using xAI's server-side X search."""
    return _xai_search(query, "x_search")


# S3: SSRF guard — block local/internal URLs
_SSRF_BLOCKED = re.compile(
    r"^https?://"
    r"(localhost|127\.\d+\.\d+\.\d+|\[::1\]|0\.0\.0\.0"
    r"|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+"
    r"|169\.254\.\d+\.\d+)"
    r"(:\d+)?",
    re.IGNORECASE,
)

def _check_ssrf(url: str) -> str | None:
    """Return error string if URL targets a blocked internal address."""
    if not url.startswith(("http://", "https://")):
        return f"Blocked: unsupported URL scheme in '{url[:60]}'. Only http/https allowed."
    if _SSRF_BLOCKED.match(url):
        return f"Blocked: requests to internal/local addresses are not allowed."
    return None


def fetch_page(url: str) -> str:
    """Fetch a URL and return its readable text content."""
    ssrf = _check_ssrf(url)
    if ssrf:
        return ssrf
    browser = _get_browser()
    if browser is None:
        return "Error: playwright is not installed. Run: pip install playwright && playwright install chromium"
    page = browser.new_page()
    try:
        page.goto(url, timeout=15000, wait_until="domcontentloaded")
        title = page.title()
        text = page.inner_text("body")
        if len(text) > 8000:
            text = text[:8000] + "\n... (truncated)"
        return f"Title: {title}\n\n{text}"
    except Exception as e:
        return f"Error fetching page: {e}"
    finally:
        page.close()


def screenshot_page(url: str, save_path: str = "screenshot.png") -> str:
    """Take a screenshot of a URL and save it to the project. Requires approval."""
    ssrf = _check_ssrf(url)
    if ssrf:
        return ssrf
    full_path = _safe_path(save_path)
    if not full_path:
        return "Access denied: outside project directory."
    console.print(f"[bold yellow]About to SCREENSHOT:[/bold yellow] {url}")
    console.print(f"[dim]Save to: {full_path}[/dim]")
    if not _auto_approve("Approve screenshot?"):
        return "Screenshot cancelled by user."
    browser = _get_browser()
    if browser is None:
        return "Error: playwright is not installed. Run: pip install playwright && playwright install chromium"
    page = browser.new_page()
    try:
        page.goto(url, timeout=15000, wait_until="domcontentloaded")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(full_path), full_page=False)
        return f"Screenshot saved to {full_path}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        page.close()


def extract_links(url: str) -> str:
    """Extract all links from a web page."""
    ssrf = _check_ssrf(url)
    if ssrf:
        return ssrf
    browser = _get_browser()
    if browser is None:
        return "Error: playwright is not installed. Run: pip install playwright && playwright install chromium"
    page = browser.new_page()
    try:
        page.goto(url, timeout=15000, wait_until="domcontentloaded")
        links = page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({text: e.innerText.trim().substring(0, 80), href: e.href})).filter(l => l.href.startsWith('http'))"
        )
        seen = set()
        lines = []
        for link in links:
            if link["href"] not in seen:
                seen.add(link["href"])
                text = link["text"][:60] if link["text"] else "(no text)"
                lines.append(f"  {text} \u2192 {link['href']}")
        if len(lines) > 50:
            lines = lines[:50] + [f"  ... and {len(lines) - 50} more"]
        return "\n".join(lines) if lines else "No links found."
    except Exception as e:
        return f"Error: {e}"
    finally:
        page.close()


# -- Tool Schemas for LLM Function Calling --
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in a project directory. Returns formatted listing with sizes. Use @grokswarm/ prefix to list GrokSwarm's own source files (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to list. Defaults to project root."}
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the project. Supports reading specific line ranges for large files. Use @grokswarm/ prefix to read GrokSwarm's own source files (read-only self-knowledge).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "start_line": {"type": "integer", "description": "Optional: 1-based start line number. Omit to read entire file."},
                    "end_line": {"type": "integer", "description": "Optional: 1-based end line number. Omit to read to end."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the project. User will be prompted to approve. Use ONLY for creating new files. For modifying existing files, use edit_file instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path for the file."},
                    "content": {"type": "string", "description": "Full content to write."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Surgically edit a file by finding and replacing specific text. Supports SINGLE edit (old_text/new_text) or MULTI-EDIT (edits array) for multiple changes in one call. Each old_text must match exactly ONE occurrence. ALWAYS prefer this over write_file for modifications. Use multi-edit when you need to change multiple places in the same file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file to edit."},
                    "old_text": {"type": "string", "description": "The exact text to find and replace (for single edit). Must match exactly once. Include 2-3 lines of surrounding context for uniqueness."},
                    "new_text": {"type": "string", "description": "The replacement text (for single edit)."},
                    "edits": {
                        "type": "array",
                        "description": "For multi-edit: array of {old_text, new_text} objects. All edits are validated before any are applied. Use this when changing multiple places in the same file.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {"type": "string", "description": "Text to find and replace."},
                                "new_text": {"type": "string", "description": "Replacement text."}
                            },
                            "required": ["old_text", "new_text"]
                        }
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files by name in the project directory tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search string to match against file names (case-insensitive)."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": "Search inside file contents for a text pattern (like grep). Returns matching lines with file:line references. Supports regex and context lines (like grep -C). Use this to find code, config values, or any text across the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text pattern to search for (case-insensitive). Supports regex when is_regex=true."},
                    "path": {"type": "string", "description": "Optional: relative path to limit search scope (file or directory). Defaults to project root."},
                    "is_regex": {"type": "boolean", "description": "If true, treat pattern as a Python regex. Default false."},
                    "context_lines": {"type": "integer", "description": "Number of context lines to show around each match (like grep -C). Default 0, max 10."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command in the project directory. User will be prompted to approve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_registry",
            "description": "List all registered experts and skills with their mindsets/descriptions.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_expert",
            "description": "Propose a new expert agent. User will be prompted to approve before saving. Use when a task needs a specialist that doesn't exist yet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Display name for the expert (e.g. 'DevOps_Engineer')."},
                    "mindset": {"type": "string", "description": "The expert's permanent personality and approach (1-2 sentences)."},
                    "objectives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of core objectives the expert should pursue."
                    }
                },
                "required": ["name", "mindset", "objectives"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Propose a new reusable skill. User will be prompted to approve before saving. Skills capture repeatable workflows or knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Display name for the skill (e.g. 'code_review')."},
                    "description": {"type": "string", "description": "What this skill does and when to use it."},
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional ordered steps for executing this skill."
                    }
                },
                "required": ["name", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show the current git status (branch, staged/unstaged/untracked files).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show git diff of changes. Can diff a specific file or all changes. Use staged=true for staged changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional: specific file to diff."},
                    "staged": {"type": "boolean", "description": "If true, show staged changes instead of unstaged. Default false."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent git commit history (oneline format with decorations).",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of commits to show (default 10, max 50)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage all changes and create a git commit. User will be prompted to approve. Use this to create checkpoints before risky changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message."}
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_checkout",
            "description": "Restore a file to its last committed state, or switch branches. User will be prompted to approve. Use to undo changes to specific files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "File path to restore, or branch/commit to switch to."}
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_branch",
            "description": "List branches, create a new branch, or delete a branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Branch name to create. Omit to list all branches."},
                    "delete": {"type": "boolean", "description": "If true, delete the named branch instead of creating it."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_show_file",
            "description": "Show the contents of a file at a specific git ref (commit SHA, branch name, or tag). Useful for comparing current version to an older one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "ref": {"type": "string", "description": "Git ref: commit SHA, branch name, or tag. Default: HEAD."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_blame",
            "description": "Show git blame for a file -- who changed each line and when. Useful for understanding code history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_stash",
            "description": "Manage git stash. Actions: 'list' (default, show stashes), 'push' (stash current changes), 'pop' (apply+drop top stash), 'drop' (discard top stash).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Stash action: list, push, pop, drop. Default: list.", "enum": ["list", "push", "pop", "drop"]},
                    "message": {"type": "string", "description": "Optional message for 'push' action."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_init",
            "description": "Initialize a new git repository in the project directory. Only needed if no .git exists.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run project tests. Auto-detects the test framework (pytest, jest, go test, etc) if no command is given. Use after code changes to verify correctness.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Optional: custom test command. If omitted, auto-detects framework."},
                    "pattern": {"type": "string", "description": "Optional: filter pattern to run specific tests (e.g. test name or file)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch a web page and return its text content. Use for reading documentation, articles, web content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "screenshot_page",
            "description": "Take a screenshot of a web page and save it to a file in the project. User will be prompted to approve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to screenshot."},
                    "save_path": {"type": "string", "description": "Relative path to save the screenshot (default: screenshot.png)."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_links",
            "description": "Extract all links from a web page. Useful for finding resources, documentation links, or navigating site structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to extract links from."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": "Find where a symbol (class, function, variable) is defined in the project. Uses AST for Python (full detail: methods, args), regex for other languages. Like go-to-definition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The symbol name to look up (e.g. 'MyClass', 'parse_config', 'MAX_RETRIES')."}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_references",
            "description": "Find all files that import or reference a given module or symbol. Uses AST import analysis for Python, word-boundary search for other languages. Like find-all-references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The module or symbol name to search for references to."}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web in real-time using xAI server-side search. Returns summarized results with source URLs. Use for current events, documentation, research, facts. Faster and more reliable than fetch_page for finding information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "x_search",
            "description": "Search X (Twitter) posts in real-time using xAI server-side search. Returns summarized posts with links. Use for opinions, trending topics, social media sentiment, real-time reactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    },
]


def analyze_image(path: str, question: str = "Describe this image in detail.") -> str:
    """Analyze an image using Grok's vision capability."""
    full_path = _safe_path(path)
    if not full_path:
        return "Access denied: outside project directory."
    if not full_path.exists() or not full_path.is_file():
        return f"File not found: {path}"
    ext = full_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(ext)
    if not mime:
        return f"Unsupported image format: {ext}. Supported: png, jpg, jpeg, gif, webp."
    data = full_path.read_bytes()
    if len(data) > 20 * 1024 * 1024:
        return "Error: image too large (max 20 MB)."
    b64 = base64.b64encode(data).decode("ascii")
    try:
        resp = _api_call_with_retry(
            lambda: client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]}],
                max_tokens=1024,
            ),
            label="Vision"
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Vision error: {e}"


TOOL_SCHEMAS = TOOL_SCHEMAS + [
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze an image file using vision AI. Can describe contents, read text/diagrams, answer questions about the image. Supports png, jpg, gif, webp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the image file."},
                    "question": {"type": "string", "description": "Optional question about the image. Default: describe the image."}
                },
                "required": ["path"]
            }
        }
    },
]

TOOL_DISPATCH = {
    "list_directory": lambda args: list_dir(args.get("path", ".")),
    "read_file": lambda args: read_file(args["path"], args.get("start_line"), args.get("end_line")),
    "write_file": lambda args: write_file(args["path"], args["content"]),
    "edit_file": lambda args: edit_file(args["path"], args.get("old_text", ""), args.get("new_text", ""), args.get("edits")),
    "search_files": lambda args: search_files(args["query"]),
    "grep_files": lambda args: grep_files(args["pattern"], args.get("path", "."), args.get("is_regex", False), args.get("context_lines", 0)),
    "run_shell": lambda args: run_shell(args["command"]),
    "run_tests": lambda args: run_tests(args.get("command"), args.get("pattern")),
    "list_registry": lambda args: get_registry(),
    "create_expert": lambda args: propose_expert(args["name"], args["mindset"], args["objectives"]),
    "create_skill": lambda args: propose_skill(args["name"], args["description"], args.get("steps")),
    "git_status": lambda args: git_status(),
    "git_diff": lambda args: git_diff(args.get("path"), args.get("staged", False)),
    "git_log": lambda args: git_log(args.get("count", 10)),
    "git_commit": lambda args: git_commit(args["message"]),
    "git_checkout": lambda args: git_checkout(args["target"]),
    "git_branch": lambda args: git_branch(args.get("name"), args.get("delete", False)),
    "git_show_file": lambda args: git_show_file(args["path"], args.get("ref", "HEAD")),
    "git_blame": lambda args: git_blame(args["path"]),
    "git_stash": lambda args: git_stash(args.get("action", "list"), args.get("message")),
    "git_init": lambda args: git_init(),
    "fetch_page": lambda args: fetch_page(args["url"]),
    "screenshot_page": lambda args: screenshot_page(args["url"], args.get("save_path", "screenshot.png")),
    "extract_links": lambda args: extract_links(args["url"]),
    "find_symbol": lambda args: find_symbol(args["name"]),
    "find_references": lambda args: find_references(args["name"]),
    "web_search": lambda args: web_search(args["query"]),
    "x_search": lambda args: x_search(args["query"]),
}
TOOL_DISPATCH["analyze_image"] = lambda args: analyze_image(args["path"], args.get("question", "Describe this image in detail."))


# -- A3: Dynamic Skill Tool Registration --
def _invoke_skill(skill_name: str, context: str = "") -> str:
    """Invoke a registered skill by loading its YAML and returning the workflow."""
    path = SKILLS_DIR / f"{skill_name}.yaml"
    if not path.exists():
        return f"Skill '{skill_name}' not found."
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = f"=== Skill: {data.get('name', skill_name)} ===\n"
    result += f"Description: {data.get('description', '')}\n"
    steps = data.get("steps", [])
    if steps:
        result += "\nSteps:\n"
        for i, step in enumerate(steps, 1):
            result += f"  {i}. {step}\n"
    if context:
        result += f"\nApply to: {context}\n"
    result += "\nFollow these steps to complete the task."
    return result


def _register_skill_tool(safe_name: str, description: str):
    """Register a skill as a callable LLM tool (schema + dispatch)."""
    tool_name = f"skill_{safe_name}"
    # Skip if already registered
    if any(t["function"]["name"] == tool_name for t in TOOL_SCHEMAS if "function" in t):
        return
    TOOL_SCHEMAS.append({
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"Invoke the '{safe_name}' skill: {description}",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "What to apply this skill to (file path, topic, etc)."}
                },
            }
        }
    })
    TOOL_DISPATCH[tool_name] = lambda args, _sn=safe_name: _invoke_skill(_sn, args.get("context", ""))
    READ_ONLY_TOOLS.add(tool_name)


def _load_skill_tools():
    """Scan skills/ directory and register each as a callable tool."""
    for f in sorted(SKILLS_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            _register_skill_tool(f.stem, data.get("description", "(no description)"))
        except Exception:
            pass


# -- Registries & Helpers --
SKILLS_DIR = Path("skills")
EXPERTS_DIR = Path("experts")
TEAMS_DIR = Path("teams")
MEMORY_DIR = Path("memory")
SESSIONS_DIR = Path.home() / ".grokswarm" / "sessions"
CONTEXT_CACHE_DIR = Path.home() / ".grokswarm" / "cache"
_RECENT_PROJECTS_FILE = Path.home() / ".grokswarm" / "recent_projects.json"
SKILLS_DIR.mkdir(exist_ok=True)
EXPERTS_DIR.mkdir(exist_ok=True)
TEAMS_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True, parents=True)
CONTEXT_CACHE_DIR.mkdir(exist_ok=True, parents=True)

_load_skill_tools()  # A3: register existing skills as callable tools

def seed_defaults():
    if not any(EXPERTS_DIR.iterdir()):
        defaults = {
            "researcher": {"name": "Researcher", "mindset": "Thorough, source-critical, always cites latest data.", "objectives": ["Deliver comprehensive, up-to-date summaries"]},
            "coder": {"name": "Coder", "mindset": "Clean, efficient, testable Python-first.", "objectives": ["Produce production-ready code"]},
            "assistant": {"name": "Personal_Assistant", "mindset": "Proactive, organized, respects Aaron's time.", "objectives": ["Handle daily tasks efficiently"]},
            "finance": {"name": "Finance_Optimizer", "mindset": "Conservative growth, tax-aware, long-term focused.", "objectives": ["Optimize financial outcomes"]}
        }
        for name, data in defaults.items():
            (EXPERTS_DIR / f"{name}.yaml").write_text(yaml.dump(data))

seed_defaults()

def list_experts():
    return [f.stem for f in EXPERTS_DIR.glob("*.yaml")]

def list_skills():
    return [f.stem for f in SKILLS_DIR.glob("*.yaml")]

def save_memory(key: str, content: str):
    entry = {"timestamp": datetime.now().isoformat(), "content": content}
    (MEMORY_DIR / f"{key}.json").write_text(json.dumps(entry, indent=2))


# -- Self-Extension Tools --
def propose_expert(name: str, mindset: str, objectives: list[str]) -> str:
    """Propose a new expert. Requires user approval before saving."""
    safe_name = name.lower().replace(" ", "_")
    expert_file = EXPERTS_DIR / f"{safe_name}.yaml"
    if expert_file.exists():
        return f"Expert '{safe_name}' already exists. Use a different name."
    console.print()
    console.print(Panel(
        f"[bold]Name:[/bold] {name}\n"
        f"[bold]Mindset:[/bold] {mindset}\n"
        f"[bold]Objectives:[/bold]\n" + "\n".join(f"  * {o}" for o in objectives),
        title="[swarm.accent]> New Expert Proposal[/swarm.accent]",
        border_style="bright_green",
        padding=(1, 2),
    ))
    if _auto_approve("[bold yellow]Approve this expert?[/bold yellow]"):
        data = {"name": name, "mindset": mindset, "objectives": objectives}
        expert_file.write_text(yaml.dump(data, default_flow_style=False))
        return f"Expert '{name}' created and saved to experts/{safe_name}.yaml"
    return "Expert creation cancelled by user."


def propose_skill(name: str, description: str, steps: list[str] | None = None) -> str:
    """Propose a new skill. Requires user approval before saving."""
    safe_name = name.lower().replace(" ", "_")
    skill_file = SKILLS_DIR / f"{safe_name}.yaml"
    if skill_file.exists():
        return f"Skill '{safe_name}' already exists. Use a different name."
    console.print()
    body = f"[bold]Name:[/bold] {name}\n[bold]Description:[/bold] {description}"
    if steps:
        body += "\n[bold]Steps:[/bold]\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
    console.print(Panel(
        body,
        title="[swarm.accent]> New Skill Proposal[/swarm.accent]",
        border_style="bright_green",
        padding=(1, 2),
    ))
    if _auto_approve("[bold yellow]Approve this skill?[/bold yellow]"):
        data = {"name": name, "description": description, "version": "1.0"}
        if steps:
            data["steps"] = steps
        skill_file.write_text(yaml.dump(data, default_flow_style=False))
        # A3: dynamically register as a callable tool
        _register_skill_tool(safe_name, description)
        return f"Skill '{name}' created and saved to skills/{safe_name}.yaml (registered as tool: skill_{safe_name})"
    return "Skill creation cancelled by user."


def get_registry() -> str:
    """Return a formatted listing of all experts and skills."""
    lines = ["Experts:"]
    for f in sorted(EXPERTS_DIR.glob("*.yaml")):
        data = yaml.safe_load(f.read_text())
        lines.append(f"  * {data.get('name', f.stem)} -- {data.get('mindset', '(no mindset)')}")
    if len(lines) == 1:
        lines.append("  (none)")
    lines.append("\nSkills:")
    for f in sorted(SKILLS_DIR.glob("*.yaml")):
        data = yaml.safe_load(f.read_text())
        lines.append(f"  * {data.get('name', f.stem)} -- {data.get('description', '(no description)')}")
    if lines[-1] == "\nSkills:":
        lines.append("  (none)")
    return "\n".join(lines)


# -- Context-Aware Tab Completion --
class SwarmCompleter(Completer):
    """Smart completer: slash commands -> subcommands -> file paths / session names."""

    SLASH_COMMANDS = {
        # -- File operations --
        "/help": "Show this help",
        "/list": "List project directory",
        "/read": "Read file contents",
        "/write": "Write/create file (with approval)",
        "/run": "Run shell command (with approval)",
        "/search": "Search files by name in project",
        "/grep": "Search inside files for text",
        # -- Git & web --
        "/git": "Git status (log, diff, branch)",
        "/web": "Search the web (xAI live)",
        "/x": "Search X/Twitter posts (xAI live)",
        "/browse": "Fetch URL content (Playwright)",
        # -- Testing & editing --
        "/test": "Run project tests (auto-detect)",
        "/undo": "Undo last file edit (multi-level)",
        # -- Modes --
        "/trust": "Toggle trust mode (auto-approve)",
        "/readonly": "Toggle read-only mode (block writes)",
        "/project": "Switch project directory",
        "/doctor": "Check environment health",
        "/dashboard": "Open live TUI dashboard",
        "/metrics": "Show token usage and cost metrics",
        # -- AI & agents --
        "/self-improve": "Improve own source (shadow + test)",
        "/swarm": "Run multi-agent supervisor",
        "/abort": "Abort currently running swarm",
        "/experts": "List available experts",
        "/skills": "List available skills",
        # -- Session & meta --
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
                                             get_paths=lambda: [str(PROJECT_DIR)])
        self._dir_completer = PathCompleter(only_directories=True, expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        parts = text.split(maxsplit=1)
        cmd_part = parts[0]

        # Still typing the slash command itself
        if len(parts) == 1 and not text.endswith(" "):
            for cmd, desc in self.SLASH_COMMANDS.items():
                if cmd.startswith(cmd_part.lower()):
                    yield Completion(cmd, start_position=-len(cmd_part), display_meta=desc)
            return

        # Command is complete -- complete the argument
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
            # Complete "list" subcommand or directory paths for switching
            if not arg_text or not arg_text.endswith(" "):
                for sc in self.PROJECT_SUBCMDS:
                    if sc.startswith(arg_text.lower()):
                        yield Completion(sc, start_position=-len(arg_text))
            # Also offer recent projects as completions
            for rp in _load_recent_projects():
                name = Path(rp).name
                if name.lower().startswith(arg_text.lower()) or rp.lower().startswith(arg_text.lower()):
                    yield Completion(rp, start_position=-len(arg_text), display_meta="recent")
            # Fall through to directory path completion for arbitrary paths
            sub_doc = Document(arg_text, len(arg_text))
            yield from self._dir_completer.get_completions(sub_doc, complete_event)
        elif cmd in self.PATH_COMMANDS:
            sub_doc = Document(arg_text, len(arg_text))
            yield from self._path_completer.get_completions(sub_doc, complete_event)

    def _complete_session(self, arg_text: str):
        arg_parts = arg_text.split(maxsplit=1)
        subcmd = arg_parts[0] if arg_parts else ""

        # Completing the subcommand
        if len(arg_parts) <= 1 and not arg_text.endswith(" "):
            for sc in self.SESSION_SUBCMDS:
                if sc.startswith(subcmd.lower()):
                    yield Completion(sc, start_position=-len(subcmd))
            return

        # Completing session name for load/delete/save
        if subcmd.lower() in ("load", "delete", "save"):
            name_prefix = arg_parts[1] if len(arg_parts) > 1 else ""
            for s in list_sessions():
                sname = s["name"]
                if sname.lower().startswith(name_prefix.lower()):
                    yield Completion(sname, start_position=-len(name_prefix),
                                    display_meta=f"{s['messages']} msgs")


# -- Recent Projects (U1/G1) --
def _load_recent_projects() -> list[str]:
    """Load the recent projects list (most recent first)."""
    try:
        if _RECENT_PROJECTS_FILE.exists():
            return json.loads(_RECENT_PROJECTS_FILE.read_text(encoding="utf-8"))[:5]
    except Exception:
        pass
    return []


def _update_recent_projects(project_dir: Path):
    """Add/promote a project to the top of the recent list."""
    recents = _load_recent_projects()
    path_str = str(project_dir.resolve())
    recents = [p for p in recents if p != path_str]
    recents.insert(0, path_str)
    recents = recents[:5]
    try:
        _RECENT_PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _RECENT_PROJECTS_FILE.write_text(json.dumps(recents), encoding="utf-8")
    except Exception:
        pass


def _switch_project(new_dir: str):
    """Switch the active project directory and rescan context."""
    global PROJECT_DIR, PROJECT_CONTEXT, SYSTEM_PROMPT
    p = Path(new_dir).resolve()
    if not p.is_dir():
        console.print(f"[swarm.error]Not a directory: {new_dir}[/swarm.error]")
        return False
    PROJECT_DIR = p
    _update_recent_projects(PROJECT_DIR)
    # B3+B4: clear cross-project state to prevent contamination
    state.reset_project_state()
    PROJECT_CONTEXT = scan_project_context_cached(PROJECT_DIR)
    SYSTEM_PROMPT = build_system_prompt(PROJECT_CONTEXT)
    console.print(f"[swarm.accent]Switched to project:[/swarm.accent] [bold]{PROJECT_DIR}[/bold]")
    file_count = len(PROJECT_CONTEXT.get('key_files', {}))
    console.print(f"[swarm.dim]  context: {file_count} key file{'s' if file_count != 1 else ''} loaded[/swarm.dim]")
    return True


# -- Session Persistence --
def _session_path(name: str) -> Path:
    """Get the file path for a named session."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
    return SESSIONS_DIR / f"{safe_name}.json"

def save_session(name: str, conversation: list):
    """Save conversation to a named session file."""
    data = {
        "name": name,
        "project": str(PROJECT_DIR),
        "updated": datetime.now().isoformat(),
        "message_count": len([m for m in conversation if m["role"] != "system"]),
        "messages": [m for m in conversation if m["role"] != "system"],
    }
    _session_path(name).write_text(_redact_secrets(json.dumps(data, indent=2)))

def load_session(name: str) -> list | None:
    """Load a named session, returning messages (without system) or None."""
    path = _session_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("messages", [])
    except (json.JSONDecodeError, KeyError):
        return None

def list_sessions() -> list[dict]:
    """List all saved sessions with metadata."""
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
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
    """Delete a named session."""
    path = _session_path(name)
    if path.exists():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# A4 — SQLite coordination bus for multi-agent communication
# ---------------------------------------------------------------------------

class SwarmBus:
    """Lightweight SQLite message bus so swarm agents can see each other's work."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_dir = PROJECT_DIR / ".grokswarm"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "bus.db")
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  ts TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),"
            "  sender TEXT NOT NULL,"
            "  recipient TEXT NOT NULL DEFAULT '*',"
            "  kind TEXT NOT NULL DEFAULT 'result',"
            "  body TEXT NOT NULL"
            ")"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  ts TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),"
            "  model TEXT NOT NULL,"
            "  prompt_tokens INTEGER NOT NULL,"
            "  completion_tokens INTEGER NOT NULL,"
            "  total_tokens INTEGER NOT NULL"
            ")"
        )
        self.conn.commit()

    def clear(self):
        self.conn.execute("DELETE FROM messages")
        self.conn.commit()

    def post(self, sender: str, body: str, *, recipient: str = "*", kind: str = "result"):
        self.conn.execute(
            "INSERT INTO messages (sender, recipient, kind, body) VALUES (?, ?, ?, ?)",
            (sender, recipient, kind, body),
        )
        self.conn.commit()

    def read(self, recipient: str = "*", *, since_id: int = 0, limit: int = 100) -> list[dict]:
        cur = self.conn.execute(
            "SELECT id, ts, sender, recipient, kind, body FROM messages "
            "WHERE id > ? AND (recipient = ? OR recipient = '*') ORDER BY id DESC LIMIT ?",
            (since_id, recipient, limit),
        )
        # Return in chronological order
        return [
            {"id": r[0], "ts": r[1], "sender": r[2], "recipient": r[3],
             "kind": r[4], "body": r[5]}
            for r in reversed(cur.fetchall())
        ]

    def summary(self) -> str:
        msgs = self.read()
        if not msgs:
            return ""
        lines = [f"[{m['sender']}\u2192{m['recipient']}] {m['body'][:300]}" for m in msgs if m['kind'] == 'result']
        return "\n".join(lines)

    def log_usage(self, model: str, prompt_tokens: int, completion_tokens: int):
        self.conn.execute(
            "INSERT INTO metrics (model, prompt_tokens, completion_tokens, total_tokens) VALUES (?, ?, ?, ?)",
            (model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
        )
        self.conn.commit()

    def get_metrics(self) -> dict:
        cur = self.conn.execute(
            "SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens) FROM metrics"
        )
        row = cur.fetchone()
        return {
            "prompt_tokens": row[0] or 0,
            "completion_tokens": row[1] or 0,
            "total_tokens": row[2] or 0,
        }

    def check_abort(self) -> bool:
        """Check if an abort signal was sent to the bus."""
        cur = self.conn.execute("SELECT 1 FROM messages WHERE kind = 'abort' LIMIT 1")
        return cur.fetchone() is not None

    def close(self):
        self.conn.close()

_bus_instance = None
def get_bus() -> SwarmBus:
    global _bus_instance
    db_path = str(PROJECT_DIR / ".grokswarm" / "bus.db")
    if _bus_instance is None:
        _bus_instance = SwarmBus()
    else:
        try:
            cur_path = _bus_instance.conn.execute("PRAGMA database_list").fetchall()[0][2]
            if cur_path != db_path:
                _bus_instance.close()
                _bus_instance = SwarmBus()
        except Exception:
            _bus_instance = SwarmBus()
    return _bus_instance


def run_supervisor(task: str):
    console.print(f"[bold green]Supervisor analyzing task:[/bold green] {task}")
    existing = list_experts()
    system_prompt = f"""You are the Grok Swarm Supervisor. Break down the task and choose ONLY from these existing experts: {existing}.
Respond ONLY with valid JSON: {{"experts": ["assistant", "researcher"], "team_name": null or "string", "reason": "brief explanation"}}"""
    try:
        response = _api_call_with_retry(
            lambda: client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
            ),
            label="Supervisor"
        )
        if hasattr(response, 'usage') and response.usage:
            get_bus().log_usage(MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)
        plan = json.loads(response.choices[0].message.content.strip())
        console.print(f"[bold cyan]Plan:[/bold cyan] {plan}")
        return plan
    except (json.JSONDecodeError, ValueError, KeyError):
        console.print("[swarm.warning]Supervisor returned invalid JSON, falling back to assistant.[/swarm.warning]")
        return {"experts": ["assistant"], "team_name": None, "reason": "fallback"}
    except Exception as e:
        console.print(f"[swarm.error]Supervisor API error: {e}[/swarm.error]")
        return {"experts": ["assistant"], "team_name": None, "reason": f"API error fallback: {e}"}

def run_expert(name: str, task_desc: str, bus: SwarmBus | None = None):
    expert_file = EXPERTS_DIR / f"{name.lower()}.yaml"
    if not expert_file.exists():
        console.print(f"[red]Expert {name} not found.[/red]")
        return ""
    data = yaml.safe_load(expert_file.read_text())
    console.print(f"[bold cyan]-> Running Expert:[/bold cyan] {data['name']} -- {data['mindset']}")

    # Build context from prior bus messages so this expert sees earlier work
    prior_context = ""
    if bus:
        summary = bus.summary()
        if summary:
            prior_context = f"\n\n--- Prior agent outputs ---\n{summary}\n---\n"

    system_prompt = f"""You are {data['name']}, an expert with permanent mindset:
{data['mindset']}

Core objectives: {data.get('objectives', ['Execute efficiently'])}

Rules:
- Stay strictly in character.
- Respond concisely and directly.
- Focus on the task, not the AI brand.
- Focus only on helping the user.{prior_context}"""
    try:
        response = _api_call_with_retry(
            lambda: client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": task_desc}]
            ),
            label=f"Expert:{data['name']}"
        )
        if hasattr(response, 'usage') and response.usage:
            get_bus().log_usage(MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)
        content = response.choices[0].message.content
        console.print(content)
        save_memory(f"expert_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", content)
        if bus:
            bus.post(name, content)
        return content
    except Exception as e:
        console.print(f"[swarm.error]Expert {data['name']} API error: {e}[/swarm.error]")
        return f"Error: {e}"

# -- API Call Retry Logic --
MAX_API_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]  # seconds between retries


def _api_call_with_retry(call_fn, label: str = "API call"):
    """Retry an API call with exponential backoff on transient failures.
    call_fn should be a zero-arg callable that makes the API request."""
    last_error = None
    for attempt in range(MAX_API_RETRIES):
        try:
            return call_fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            # Don't retry on auth errors or invalid requests
            if any(k in err_str for k in ("401", "403", "invalid_api_key", "authentication")):
                raise
            if attempt < MAX_API_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                console.print(f"  [swarm.warning]\u26a0 {label} failed (attempt {attempt + 1}/{MAX_API_RETRIES}): {type(e).__name__}: {str(e)[:100]}[/swarm.warning]")
                console.print(f"  [swarm.dim]  retrying in {wait}s...[/swarm.dim]")
                time.sleep(wait)
            else:
                console.print(f"  [swarm.error]\u2718 {label} failed after {MAX_API_RETRIES} attempts: {e}[/swarm.error]")
                raise
    raise last_error  # unreachable, but satisfies type checker


MAX_CONVERSATION_MESSAGES = 40  # keep last N user+assistant messages (plus system)
COMPACTION_THRESHOLD = 50       # trigger compaction at this many non-system messages
COMPACTION_KEEP_RECENT = 20     # keep this many recent messages after compaction
COMPACTION_TOKEN_LIMIT = 100_000  # ~estimated token budget before forcing compaction


def _estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += len(content) // 4
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                total += len(tc.get("function", {}).get("arguments", "")) // 4
    return total


def _compact_conversation(conversation: list) -> list:
    """Summarize older messages into a compact summary, keeping recent ones intact.
    Uses the LLM to create a conversation summary, then replaces old messages with it."""
    system = [m for m in conversation if m["role"] == "system"]
    others = [m for m in conversation if m["role"] != "system"]

    if len(others) <= COMPACTION_KEEP_RECENT:
        return conversation

    # Split: old messages to summarize, recent messages to keep
    old_messages = others[:-COMPACTION_KEEP_RECENT]
    recent_messages = others[-COMPACTION_KEEP_RECENT:]

    # C4: Don't start recent in the middle of a tool-call exchange
    while recent_messages and recent_messages[0]["role"] == "tool":
        old_messages.append(recent_messages.pop(0))
    # C4: Also don't break after an assistant message that has tool_calls
    # (the tool responses must stay with their assistant message)
    while (recent_messages and recent_messages[0]["role"] == "assistant"
           and recent_messages[0].get("tool_calls")
           and len(recent_messages) > 1
           and recent_messages[1]["role"] == "tool"):
        # This assistant issued tool calls — keep pulling until we clear the tool responses
        old_messages.append(recent_messages.pop(0))
        while recent_messages and recent_messages[0]["role"] == "tool":
            old_messages.append(recent_messages.pop(0))

    # Build a text representation of old messages for summarization
    old_text_parts = []
    for m in old_messages:
        role = m["role"]
        content = m.get("content", "")
        if role == "tool":
            # Include a brief summary of tool results for context
            snippet = content[:150].replace("\n", " ") if content else "(empty)"
            old_text_parts.append(f"  tool result: {snippet}")
            continue
        if content:
            old_text_parts.append(f"{role}: {content[:200]}")
        if m.get("tool_calls"):
            tools_used = [tc["function"]["name"] for tc in m["tool_calls"]]
            old_text_parts.append(f"  (used tools: {', '.join(tools_used)})")

    if not old_text_parts:
        return conversation

    old_text = "\n".join(old_text_parts)
    if len(old_text) > 4000:
        old_text = old_text[:4000] + "\n... (truncated)"

    try:
        summary_response = _api_call_with_retry(
            lambda: client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Summarize this conversation history into a concise paragraph. Capture key decisions, files modified, tasks completed, and important context. Be factual and brief."},
                    {"role": "user", "content": old_text}
                ],
                max_tokens=500,
            ),
            label="Compaction"
        )
        if hasattr(summary_response, 'usage') and summary_response.usage:
            get_bus().log_usage(MODEL, summary_response.usage.prompt_tokens, summary_response.usage.completion_tokens)
        summary = summary_response.choices[0].message.content.strip()
    except Exception:
        # Fallback to hard trim if summarization fails
        return system + others[-MAX_CONVERSATION_MESSAGES:]

    # Build compacted conversation
    summary_msg = {"role": "user", "content": f"[CONVERSATION SUMMARY -- earlier messages compacted]\n{summary}"}
    ack_msg = {"role": "assistant", "content": "Understood. I have the context from our earlier conversation. Let's continue."}

    return system + [summary_msg, ack_msg] + recent_messages


def _trim_conversation(conversation: list) -> list:
    """Smart conversation management: compact old messages into a summary when needed."""
    system = [m for m in conversation if m["role"] == "system"]
    others = [m for m in conversation if m["role"] != "system"]

    # P3: token-aware compaction — trigger on estimated tokens OR message count
    est_tokens = _estimate_tokens(conversation)
    if len(others) > COMPACTION_THRESHOLD or est_tokens > COMPACTION_TOKEN_LIMIT:
        console.print(f"[swarm.dim]  ~ compacting conversation history (~{est_tokens:,} tokens, {len(others)} msgs)...[/swarm.dim]")
        return _compact_conversation(conversation)

    # Simple trim as fallback
    if len(others) > MAX_CONVERSATION_MESSAGES:
        others = others[-MAX_CONVERSATION_MESSAGES:]
        while others and others[0]["role"] == "tool":
            others.pop(0)

    return system + others


def _repair_json(raw: str) -> str:
    """Try to fix common LLM JSON issues: markdown fences, trailing commas."""
    s = raw.strip()
    # Strip markdown code fences
    if s.startswith("```"):
        lines = s.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        s = "\n".join(lines).strip()
    # Fix trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


MAX_TOOL_ROUNDS = 10
MAX_TOOL_RESULT_SIZE = 12000  # chars per tool result — prevents context overflow
MAX_STREAM_RETRIES = 2  # retries for mid-stream failures (network drops, etc.)

# Tools that only read state and can safely run in parallel
READ_ONLY_TOOLS = {
    "list_directory", "read_file", "search_files", "grep_files",
    "list_registry", "git_status", "git_diff", "git_log",
    "git_branch", "git_show_file", "git_blame",
    "fetch_page", "extract_links",
    "find_symbol", "find_references",
    "web_search", "x_search",
    "analyze_image",
}

# Tools that mutate files (tracked for auto-checkpoint reminders)
_FILE_MUTATION_TOOLS = {"edit_file", "write_file"}

# S5: Tools blocked in read-only mode (run_tests excluded — tests are non-mutating)
_READONLY_BLOCKED_TOOLS = {"write_file", "edit_file", "run_shell", "git_commit",
                           "git_checkout", "git_branch", "git_stash", "git_init",
                           "create_expert", "create_skill", "screenshot_page"}


def _execute_tool(name: str, args: dict, timed: bool = False):
    """Execute a single tool and return the result string (or (result, elapsed) if timed)."""
    t0 = time.perf_counter()

    # S5: Read-only session guard — block file-mutating tools
    if state.read_only and name in _READONLY_BLOCKED_TOOLS:
        result_str = f"BLOCKED: Session is in read-only mode. Use /readonly to toggle. Tool '{name}' is not allowed."
        if timed:
            return result_str, time.perf_counter() - t0
        return result_str

    # Mechanical guard: during /self-improve, block writes to main.py
    if state.self_improve_active and name in ("edit_file", "write_file"):
        target = args.get("path", "")
        if target and Path(target).name == "main.py" and not str(target).startswith(".grokswarm"):
            result_str = "BLOCKED: During /self-improve, you must only edit the shadow copy at .grokswarm/shadow/main.py, not main.py directly."
            if timed:
                return result_str, time.perf_counter() - t0
            return result_str

    # Mechanical guard: during /self-improve, block shell commands that touch main.py
    if state.self_improve_active and name == "run_shell":
        shell_cmd = args.get("command", "")
        # Match main.py references that aren't inside .grokswarm/shadow
        if re.search(r'(?<!shadow/)\bmain\.py\b', shell_cmd) and '.grokswarm' not in shell_cmd:
            result_str = "BLOCKED: During /self-improve, shell commands must not touch main.py directly. Edit .grokswarm/shadow/main.py instead."
            if timed:
                return result_str, time.perf_counter() - t0
            return result_str

    # B2: capture pre-edit state BEFORE running the tool (for undo)
    # B12: _pre_edit_content=None means file didn't exist (sentinel for delete-on-undo)
    _pre_edit_content: str | None = None
    _pre_edit_existed: bool = False
    if name in _FILE_MUTATION_TOOLS:
        edit_path = args.get("path", "")
        try:
            p = _safe_path(edit_path)
            if p and p.is_file():
                _pre_edit_content = p.read_text(encoding="utf-8", errors="ignore")
                _pre_edit_existed = True
        except Exception:
            pass

    handler = TOOL_DISPATCH.get(name)
    if handler:
        try:
            result = handler(args)
        except Exception as e:
            result = f"Error: {e}"
    else:
        result = f"Unknown tool: {name}"

    result_str = str(result)

    # Truncate large tool results to prevent context overflow
    if len(result_str) > MAX_TOOL_RESULT_SIZE:
        result_str = result_str[:MAX_TOOL_RESULT_SIZE] + f"\n... (truncated from {len(str(result)):,} chars to {MAX_TOOL_RESULT_SIZE:,})"

    # Auto-lint after file mutations
    lint_clean = False
    if name in ("edit_file", "write_file") and not result_str.startswith(("Error", "Cancelled", "Access denied", "Edit cancelled")):
        edited_path = _safe_path(args.get("path", ""))
        if edited_path and edited_path.exists():
            lint_err = _lint_file(edited_path)
            if lint_err:
                console.print(f"  [swarm.warning]\u26a0 lint error:[/swarm.warning] [dim]{lint_err[:120]}[/dim]")
                result_str += f"\n\n[AUTO-LINT ERROR] Syntax check failed after edit:\n{lint_err}\nYou MUST fix this immediately using edit_file before proceeding."
            else:
                console.print(f"  [swarm.accent]\u2714 lint clean[/swarm.accent]")
                lint_clean = True

    # Track file mutations for auto-checkpoint reminders
    if name in _FILE_MUTATION_TOOLS and not result_str.startswith(("Error", "Cancelled", "Access denied", "Edit cancelled")):
        edit_path = args.get("path", "")
        # G3: snapshot previous content for multi-level undo (pre-edit state captured above)
        # B12: for newly created files, store (path, None) sentinel for delete-on-undo
        if _pre_edit_existed:
            state.edit_history.append((edit_path, _pre_edit_content))
            if len(state.edit_history) > MAX_EDIT_HISTORY:
                state.edit_history.pop(0)
        elif not _pre_edit_existed:
            state.edit_history.append((edit_path, None))
            if len(state.edit_history) > MAX_EDIT_HISTORY:
                state.edit_history.pop(0)
        state.last_edited_file = edit_path
        # C9: incremental context refresh
        _incremental_context_refresh(edit_path)
        state.pending_write_count += 1
        if state.pending_write_count >= AUTO_CHECKPOINT_THRESHOLD:
            console.print(f"  [swarm.warning]\u26a0 {state.pending_write_count} file mutations without a commit \u2014 consider git_commit for a checkpoint[/swarm.warning]")
            result_str += f"\n\n[AUTO-CHECKPOINT] You have made {state.pending_write_count} file edits without committing. Consider using git_commit to create a checkpoint before continuing."
    elif name == "git_commit" and not result_str.startswith("Error"):
        state.pending_write_count = 0

    # Track test-fix cycle: when run_tests fails, remember the command
    if name == "run_tests" and result_str.startswith("[FAIL]"):
        # Store the test command for auto-retest after subsequent edits
        cmd = args.get("command") or ""
        if not cmd:
            fw = _detect_test_framework()
            if fw:
                cmd = TEST_COMMANDS[fw]["cmd"]
        pattern = args.get("pattern")
        if pattern and cmd:
            cmd += f" -k {pattern}" if "pytest" in cmd else f" {pattern}"
        if cmd:
            state.test_fix_state["cmd"] = cmd
            state.test_fix_state["attempts"] = 0
            result_str += f"\n\n[AUTO-TEST FAILURE] Tests failed. You MUST: 1) analyze the failure output above, 2) fix the code with edit_file, 3) the system will auto-rerun tests after your edit to verify the fix. Do NOT proceed to other tasks until tests pass."
    elif name == "run_tests" and result_str.startswith("[PASS]"):
        # Tests passed — clear any active test-fix cycle
        if state.test_fix_state["cmd"]:
            console.print(f"  [swarm.accent]\u2714 test-fix cycle complete[/swarm.accent]")
        state.test_fix_state["cmd"] = None
        state.test_fix_state["attempts"] = 0

    # Auto-retest after edits if we're in a test-fix cycle
    if (name in ("edit_file", "write_file")
            and lint_clean
            and state.test_fix_state["cmd"]
            and state.test_fix_state["attempts"] < MAX_TEST_FIX_ATTEMPTS):
        state.test_fix_state["attempts"] += 1
        test_cmd = state.test_fix_state["cmd"]
        attempt = state.test_fix_state["attempts"]
        console.print(f"  [swarm.accent]\u21bb auto-retest ({attempt}/{MAX_TEST_FIX_ATTEMPTS}):[/swarm.accent] [dim]{test_cmd}[/dim]")
        test_output = _run_tests_raw(test_cmd, timeout=60)
        if test_output.startswith("[PASS]"):
            console.print(f"  [swarm.accent]\u2714 tests pass — fix verified![/swarm.accent]")
            state.test_fix_state["cmd"] = None
            state.test_fix_state["attempts"] = 0
            result_str += f"\n\n[AUTO-RETEST PASSED] Tests now pass after your edit. Fix verified. You may proceed."
        else:
            # Extract just the key failure info to keep context lean
            fail_summary = test_output[:3000] if len(test_output) > 3000 else test_output
            result_str += f"\n\n[AUTO-RETEST FAILED] (attempt {attempt}/{MAX_TEST_FIX_ATTEMPTS}) Tests still failing after edit:\n{fail_summary}\nAnalyze the error and fix with another edit_file call."
            if attempt >= MAX_TEST_FIX_ATTEMPTS:
                console.print(f"  [swarm.warning]\u26a0 auto-retest limit reached — continuing without auto-retest[/swarm.warning]")
                result_str += "\n\nAuto-retest limit reached. Use run_tests manually to continue testing."

    elapsed = time.perf_counter() - t0
    if timed:
        return result_str, elapsed
    return result_str

def _stream_with_tools(conversation: list) -> str:
    """Stream a response, handling tool calls in a loop until a final text reply."""
    full_response = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for _round in range(MAX_TOOL_ROUNDS):
        full_response = ""
        tool_calls_data = {}
        finish_reason = None
        round_usage = None

        for _stream_attempt in range(1 + MAX_STREAM_RETRIES):
            try:
                with Live(Spinner("dots", text="[swarm.dim] thinking...[/swarm.dim]"), console=console, refresh_per_second=15, transient=True) as live:
                    stream = _api_call_with_retry(
                        lambda: client.chat.completions.create(
                            model=MODEL,
                            messages=conversation,
                            tools=TOOL_SCHEMAS,
                            stream=True,
                            stream_options={"include_usage": True},
                            max_tokens=MAX_TOKENS,
                        ),
                        label="Chat"
                    )
                    started = False
                    for chunk in stream:
                        # Usage info arrives in a final chunk with no choices
                        if hasattr(chunk, 'usage') and chunk.usage:
                            round_usage = chunk.usage
                            continue
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        delta = choice.delta

                        # Track finish reason
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason

                        if delta.content:
                            if not started:
                                live.update(Text(""))
                                started = True
                            full_response += delta.content
                            try:
                                live.update(Markdown(full_response))
                            except Exception:
                                live.update(Text(full_response))

                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_calls_data:
                                    tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                                if tc.id:
                                    tool_calls_data[idx]["id"] = tc.id
                                if tc.function and tc.function.name:
                                    tool_calls_data[idx]["name"] = tc.function.name
                                if tc.function and tc.function.arguments:
                                    tool_calls_data[idx]["arguments"] += tc.function.arguments
                break  # stream completed successfully
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if _stream_attempt < MAX_STREAM_RETRIES:
                    console.print(f"  [swarm.warning]\u26a0 Stream interrupted: {type(e).__name__}: {str(e)[:80]}[/swarm.warning]")
                    console.print(f"  [swarm.dim]  retrying stream (attempt {_stream_attempt + 2}/{1 + MAX_STREAM_RETRIES})...[/swarm.dim]")
                    full_response = ""
                    tool_calls_data = {}
                    finish_reason = None
                    round_usage = None
                    time.sleep(2)
                else:
                    raise

        # Accumulate usage stats
        if round_usage:
            pt = getattr(round_usage, 'prompt_tokens', 0) or 0
            ct = getattr(round_usage, 'completion_tokens', 0) or 0
            total_prompt_tokens += pt
            total_completion_tokens += ct
            get_bus().log_usage(MODEL, pt, ct)

        # Detect truncated response (model hit token limit)
        if finish_reason == "length":
            console.print("  [swarm.warning]\u26a0 Response was truncated (token limit). Output may be incomplete.[/swarm.warning]")

        # No tool calls -> done
        if not tool_calls_data:
            if full_response:
                # Re-print permanently since Live(transient=True) cleared it
                try:
                    console.print(Markdown(full_response))
                except Exception:
                    console.print(full_response)
                conversation.append({"role": "assistant", "content": full_response})
            # Show token usage
            if total_prompt_tokens or total_completion_tokens:
                total = total_prompt_tokens + total_completion_tokens
                console.print(f"  [dim]tokens: {total_prompt_tokens:,} in + {total_completion_tokens:,} out = {total:,} total[/dim]")
            return full_response

        # Print any text that accompanied the tool calls
        if full_response:
            try:
                console.print(Markdown(full_response))
            except Exception:
                console.print(full_response)

        # Append assistant message with tool_calls
        tool_calls_list = []
        for idx in sorted(tool_calls_data.keys()):
            tc = tool_calls_data[idx]
            tool_calls_list.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            })
        conversation.append({
            "role": "assistant",
            "content": full_response or None,
            "tool_calls": tool_calls_list,
        })

        # Execute each tool
        tool_count = len(tool_calls_data)
        console.print(f"  [swarm.dim]round {_round + 1}/{MAX_TOOL_ROUNDS} \u2014 {tool_count} tool{'s' if tool_count != 1 else ''}[/swarm.dim]")

        # Parse all tool arguments first
        parsed_tools: list[tuple[dict, str, dict]] = []  # (tc, name, args)
        for idx in sorted(tool_calls_data.keys()):
            tc = tool_calls_data[idx]
            name = tc["name"]
            try:
                args = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                # A7: try repairing common LLM JSON issues
                try:
                    args = json.loads(_repair_json(tc["arguments"]))
                except json.JSONDecodeError as e:
                    console.print(f"  [swarm.accent]\u26a1 {name}[/swarm.accent][dim] (invalid arguments)[/dim]")
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"Error: Failed to parse tool arguments as JSON: {e}. Raw arguments: {tc['arguments'][:200]}. Please retry with valid JSON arguments.",
                    })
                    continue
            parsed_tools.append((tc, name, args))

        # Build detail string for display
        def _tool_detail(name: str, args: dict) -> str:
            if name in ("write_file", "read_file", "edit_file"):
                return f" \u2192 {args.get('path', '?')}"
            elif name == "list_directory":
                return f" \u2192 {args.get('path', '.')}"
            elif name == "search_files":
                return f" \u2192 {args.get('query', '?')}"
            elif name == "grep_files":
                return f" \u2192 '{args.get('pattern', '?')}' in {args.get('path', '.')}"
            elif name == "run_shell":
                return f" \u2192 {args.get('command', '?')}"
            elif name in ("create_expert", "create_skill"):
                return f" \u2192 {args.get('name', '?')}"
            elif name == "list_registry":
                return ""
            elif name.startswith("git_"):
                if name == "git_commit":
                    return f" \u2192 {args.get('message', '?')[:50]}"
                elif name == "git_checkout":
                    return f" \u2192 {args.get('target', '?')}"
                elif name == "git_diff":
                    return f" \u2192 {args.get('path', 'all')}"
                elif name == "git_branch":
                    return f" \u2192 {args.get('name', 'list')}"
                elif name == "git_show_file":
                    return f" \u2192 {args.get('path', '?')}@{args.get('ref', 'HEAD')}"
                elif name == "git_blame":
                    return f" \u2192 {args.get('path', '?')}"
                elif name == "git_stash":
                    return f" \u2192 {args.get('action', 'list')}"
                elif name == "git_init":
                    return ""
                return ""
            elif name == "run_tests":
                return f" \u2192 {args.get('command', 'auto-detect')}"
            elif name in ("fetch_page", "extract_links"):
                return f" \u2192 {args.get('url', '?')[:60]}"
            elif name in ("web_search", "x_search"):
                return f" \u2192 '{args.get('query', '?')[:60]}'"
            elif name == "screenshot_page":
                return f" \u2192 {args.get('url', '?')[:40]} \u2192 {args.get('save_path', 'screenshot.png')}"
            elif name == "analyze_image":
                return f" \u2192 {args.get('path', '?')}"
            elif name.startswith("skill_"):
                ctx = args.get("context", "")
                return f" \u2192 {ctx[:60]}" if ctx else ""
            return ""

        try:
            # Determine if all tools in this round are read-only
            all_read_only = all(name in READ_ONLY_TOOLS for _, name, _ in parsed_tools)
            can_parallelize = all_read_only and len(parsed_tools) > 1

            if can_parallelize:
                # Execute read-only tools in parallel
                console.print(f"  [swarm.dim]\u21c4 executing {len(parsed_tools)} read-only tools in parallel[/swarm.dim]")
                futures = {}
                with ThreadPoolExecutor(max_workers=min(len(parsed_tools), 4)) as pool:
                    for tc, name, args in parsed_tools:
                        detail = _tool_detail(name, args)
                        console.print(f"  [swarm.accent]\u26a1 {name}[/swarm.accent][dim]{detail}[/dim]")
                        futures[pool.submit(_execute_tool, name, args, True)] = tc
                    for future in as_completed(futures):
                        tc = futures[future]
                        try:
                            result_str, elapsed = future.result()
                        except Exception as e:
                            result_str, elapsed = f"Error: {e}", 0.0
                        console.print(f"    [dim]\u2514 {tc['name']} done ({elapsed:.1f}s)[/dim]")
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result_str,
                        })
            else:
                # Execute sequentially (mutating tools or single tool)
                for tc, name, args in parsed_tools:
                    detail = _tool_detail(name, args)
                    console.print(f"  [swarm.accent]\u26a1 {name}[/swarm.accent][dim]{detail}[/dim]")
                    t0 = time.perf_counter()
                    result_str = _execute_tool(name, args)
                    elapsed = time.perf_counter() - t0
                    if elapsed >= 0.5:
                        console.print(f"    [dim]\u2514 done ({elapsed:.1f}s)[/dim]")
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })

        except KeyboardInterrupt:
            console.print("\n  [swarm.warning]\u26a0 Tool execution interrupted by user.[/swarm.warning]")
            # Ensure every tool call has a response (API requires it)
            responded_ids = {m["tool_call_id"] for m in conversation if m.get("role") == "tool"}
            for tc, name, args in parsed_tools:
                if tc["id"] not in responded_ids:
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "Cancelled: tool execution interrupted by user.",
                    })
            return full_response

    return full_response


def show_welcome(session_name: str | None = None):
    console.print()
    console.print(Panel(
        f"[bold white]Grok Swarm[/bold white]  [dim]v{VERSION}[/dim]\n"
        f"[dim]model: {MODEL}[/dim]",
        border_style="bright_green",
        padding=(1, 2),
        width=42,
    ))
    display_dir = os.environ.get("GROKSWARM_HOST_DIR") or str(PROJECT_DIR)
    console.print(f"[swarm.dim]  project:    [bold]{display_dir}[/bold][/swarm.dim]")
    file_count = len(PROJECT_CONTEXT.get('key_files', {}))
    console.print(f"[swarm.dim]  context:    {file_count} key file{'s' if file_count != 1 else ''} loaded[/swarm.dim]")
    if session_name:
        console.print(f"[swarm.dim]  session:    [bold]{session_name}[/bold] (auto-saving)[/swarm.dim]")
    else:
        console.print("[swarm.dim]  session:    (ephemeral -- use /session save <name> to persist)[/swarm.dim]")
    console.print("[swarm.dim]  /help for commands * /context to view * tab to complete[/swarm.dim]")
    console.print()

@app.callback(invoke_without_command=True)
def main(
    ctx: Context,
    session: str = typer.Option(None, "--session", "-s", help="Resume or start a named session"),
    model: str = typer.Option(None, "--model", "-m", help="Override model name (e.g. grok-3-latest)"),
    base_url: str = typer.Option(None, "--base-url", help="Override API base URL (e.g. https://api.openai.com/v1)"),
    api_key: str = typer.Option(None, "--api-key", help="Override API key (instead of XAI_API_KEY env var)"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="Override max output tokens per response (default: 16384)"),
    project_dir: str = typer.Option(None, "--project-dir", "-d", help="Set project directory (default: current working directory)"),
):
    global MODEL, BASE_URL, MAX_TOKENS, client, IGNORE_DIRS, _IGNORE_PATTERNS, _IGNORE_LITERALS
    global PROJECT_DIR, PROJECT_CONTEXT, SYSTEM_PROMPT
    # Resolve project directory: --project-dir flag > cwd
    if project_dir:
        p = Path(project_dir).resolve()
        if p.is_dir():
            PROJECT_DIR = p
        else:
            console.print(f"[swarm.warning]Warning: --project-dir '{project_dir}' not found, using cwd[/swarm.warning]")
    _update_recent_projects(PROJECT_DIR)
    PROJECT_CONTEXT = scan_project_context_cached(PROJECT_DIR)
    SYSTEM_PROMPT = build_system_prompt(PROJECT_CONTEXT)
    # U4: load .grokswarm.yml project config if present
    config_path = PROJECT_DIR / ".grokswarm.yml"
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not model and cfg.get("model"):
                MODEL = cfg["model"]
            if not max_tokens and cfg.get("max_tokens"):
                MAX_TOKENS = int(cfg["max_tokens"])
            if not base_url and cfg.get("base_url"):
                BASE_URL = cfg["base_url"]
                client = OpenAI(api_key=os.getenv(cfg.get("api_key_env", "XAI_API_KEY")) or XAI_API_KEY, base_url=BASE_URL)
            extra_ignore = cfg.get("ignore_dirs", [])
            if extra_ignore:
                IGNORE_DIRS = IGNORE_DIRS | set(extra_ignore)
                _IGNORE_PATTERNS = [p for p in IGNORE_DIRS if "*" in p]
                _IGNORE_LITERALS = IGNORE_DIRS - set(_IGNORE_PATTERNS)
            console.print(f"[swarm.dim]  loaded .grokswarm.yml[/swarm.dim]")
        except Exception as e:
            console.print(f"[swarm.warning]Warning: .grokswarm.yml: {e}[/swarm.warning]")
    if model:
        MODEL = model
    if max_tokens:
        MAX_TOKENS = max_tokens
    if base_url:
        BASE_URL = base_url
    if api_key or base_url:
        client = OpenAI(api_key=api_key or XAI_API_KEY, base_url=BASE_URL)
    if ctx.invoked_subcommand is None:
        show_welcome(session)
        chat(session_name=session)

@app.command()
def chat(session_name: str = typer.Argument(None, hidden=True)):
    """Interactive mode with tab completion + streaming."""
    history_file = Path("~/.grokswarm/history.txt").expanduser()
    history_file.parent.mkdir(exist_ok=True, parents=True)

    completer = SwarmCompleter()

    # Escape: first press closes completion menu, second press clears input
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
            # Perform the delete
            buf.delete_before_cursor(1)
            # Re-trigger completion if still typing a slash command
            if buf.text.startswith('/'):
                buf.start_completion()
            elif buf.complete_state:
                buf.cancel_completion()

    session = PromptSession(history=FileHistory(str(history_file)), completer=completer,
                            complete_while_typing=True, key_bindings=kb)
    # Make Escape instant by reducing the VT100 escape sequence timeout
    session.app.ttimeoutlen = 0.01
    session.app.timeoutlen = 0.01

    global PROJECT_CONTEXT, SYSTEM_PROMPT
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Load existing session if specified
    if session_name:
        saved_msgs = load_session(session_name)
        if saved_msgs:
            conversation.extend(saved_msgs)
            msg_count = len(saved_msgs)
            console.print(f"[swarm.accent]Resumed session '[bold]{session_name}[/bold]' ({msg_count} messages)[/swarm.accent]")
        else:
            console.print(f"[swarm.accent]Started new session '[bold]{session_name}[/bold]'[/swarm.accent]")
        console.print()

    while True:
        try:
            user_input = session.prompt(HTML("<b><ansibrightcyan>> </ansibrightcyan></b>")).strip()
            if not user_input:
                continue

            # -- Slash Commands --
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0][1:].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ["quit", "exit", "q"]:
                    if session_name:
                        save_session(session_name, conversation)
                        console.print(f"[swarm.dim]Session '{session_name}' saved.[/swarm.dim]")
                    console.print("[swarm.dim]Goodbye.[/swarm.dim]")
                    break
                elif cmd == "help":
                    _show_help()
                elif cmd == "clear":
                    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
                    os.system("cls" if os.name == "nt" else "clear")
                    show_welcome()
                elif cmd == "context":
                    _show_context(arg)
                    if arg == "refresh":
                        PROJECT_CONTEXT = scan_project_context(PROJECT_DIR)
                        _save_context_cache(PROJECT_DIR, PROJECT_CONTEXT)
                        SYSTEM_PROMPT = build_system_prompt(PROJECT_CONTEXT)
                        # Update system prompt in-place, preserving conversation history
                        conversation[0] = {"role": "system", "content": SYSTEM_PROMPT}
                elif cmd == "session":
                    result = _handle_session_command(arg, conversation, session_name)
                    if result:
                        session_name = result
                elif cmd == "list":
                    console.print(list_dir(arg or "."))
                elif cmd == "read":
                    if not arg:
                        console.print("[swarm.warning]Usage: /read <file>[/swarm.warning]")
                    else:
                        console.print(read_file(arg))
                elif cmd == "write":
                    if not arg:
                        console.print("[swarm.warning]Usage: /write <file>[/swarm.warning]")
                    else:
                        file_path = arg.split(maxsplit=1)[0]
                        console.print("[swarm.dim]Enter content (type END on a new line to finish):[/swarm.dim]")
                        lines = []
                        while True:
                            line = session.prompt("  ")
                            if line.strip() == "END":
                                break
                            lines.append(line)
                        console.print(write_file(file_path, "\n".join(lines)))
                elif cmd == "run":
                    if not arg:
                        console.print("[swarm.warning]Usage: /run <command>[/swarm.warning]")
                    else:
                        console.print(run_shell(arg))
                elif cmd == "search":
                    if not arg:
                        console.print("[swarm.warning]Usage: /search <query>[/swarm.warning]")
                    else:
                        console.print(search_files(arg))
                elif cmd == "grep":
                    if not arg:
                        console.print("[swarm.warning]Usage: /grep <pattern> [path]  (quote multi-word patterns)[/swarm.warning]")
                    else:
                        # Support quoted patterns: /grep "multi word" path
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
                        console.print(grep_files(grep_pattern, grep_path))
                elif cmd == "swarm":
                    if not arg:
                        console.print("[swarm.warning]Usage: /swarm <task>[/swarm.warning]")
                    else:
                        swarm(arg)
                elif cmd == "experts":
                    experts_list()
                elif cmd == "skills":
                    skills_list()
                elif cmd == "git":
                    if not arg:
                        console.print(git_status())
                    elif arg.startswith("log"):
                        count_str = arg.split(maxsplit=1)[1] if len(arg.split()) > 1 else "10"
                        console.print(git_log(int(count_str) if count_str.isdigit() else 10))
                    elif arg.startswith("diff"):
                        diff_path = arg.split(maxsplit=1)[1] if len(arg.split()) > 1 else None
                        console.print(git_diff(diff_path))
                    elif arg.startswith("branch"):
                        branch_name = arg.split(maxsplit=1)[1] if len(arg.split()) > 1 else None
                        console.print(git_branch(branch_name))
                    else:
                        console.print("[swarm.dim]Usage: /git [log|diff|branch] [args][/swarm.dim]")
                elif cmd == "web":
                    if not arg:
                        console.print("[swarm.warning]Usage: /web <query>[/swarm.warning]")
                    else:
                        console.print(f"[swarm.dim]Searching web: {arg}...[/swarm.dim]")
                        console.print(web_search(arg))
                elif cmd == "x":
                    if not arg:
                        console.print("[swarm.warning]Usage: /x <query>[/swarm.warning]")
                    else:
                        console.print(f"[swarm.dim]Searching X: {arg}...[/swarm.dim]")
                        console.print(x_search(arg))
                elif cmd == "browse":
                    if not arg:
                        console.print("[swarm.warning]Usage: /browse <url>[/swarm.warning]")
                    else:
                        console.print(f"[swarm.dim]Fetching {arg}...[/swarm.dim]")
                        console.print(fetch_page(arg))
                elif cmd == "test":
                    console.print(run_tests(arg if arg else None))
                elif cmd == "undo":
                    if not state.edit_history:
                        console.print("[swarm.warning]Nothing to undo. No edit history.[/swarm.warning]")
                    else:
                        undo_path, undo_content = state.edit_history[-1]
                        if undo_content is None:
                            # B12: file was newly created — undo means delete
                            console.print(f"[bold yellow]Undo file creation:[/bold yellow] {undo_path}")
                            console.print(f"[dim]Edit history depth: {len(state.edit_history)}[/dim]")
                            if Confirm.ask("Delete this newly created file?", default=False):
                                fp = _safe_path(undo_path)
                                if fp and fp.is_file():
                                    fp.unlink()
                                    state.edit_history.pop()
                                    console.print(f"[swarm.accent]Deleted {undo_path} (history: {len(state.edit_history)} remaining)[/swarm.accent]")
                                elif fp:
                                    console.print("[swarm.error]File no longer exists.[/swarm.error]")
                                    state.edit_history.pop()
                                else:
                                    console.print("[swarm.error]Cannot undo: path outside project.[/swarm.error]")
                            else:
                                console.print("[swarm.dim]Undo cancelled.[/swarm.dim]")
                        else:
                            console.print(f"[bold yellow]Undo last edit to:[/bold yellow] {undo_path}")
                            console.print(f"[dim]Edit history depth: {len(state.edit_history)}[/dim]")
                            if Confirm.ask("Restore previous content?", default=False):
                                fp = _safe_path(undo_path)
                                if fp:
                                    fp.write_text(undo_content, encoding="utf-8")
                                    state.edit_history.pop()
                                    console.print(f"[swarm.accent]Restored {undo_path} (history: {len(state.edit_history)} remaining)[/swarm.accent]")
                                else:
                                    console.print("[swarm.error]Cannot restore: path outside project.[/swarm.error]")
                            else:
                                console.print("[swarm.dim]Undo cancelled.[/swarm.dim]")
                elif cmd == "trust":
                    state.trust_mode = not state.trust_mode
                    trust_state = "ON" if state.trust_mode else "OFF"
                    color = "bold green" if state.trust_mode else "bold red"
                    console.print(f"[{color}]Trust mode: {trust_state}[/{color}]")
                    if state.trust_mode:
                        console.print("[swarm.dim]Non-dangerous ops will be auto-approved. Shell + destructive git still gated.[/swarm.dim]")
                elif cmd == "readonly":
                    state.read_only = not state.read_only
                    ro_state = "ON" if state.read_only else "OFF"
                    color = "bold yellow" if state.read_only else "bold green"
                    console.print(f"[{color}]Read-only mode: {ro_state}[/{color}]")
                    if state.read_only:
                        console.print("[swarm.dim]All file-mutating tools are blocked. Use /readonly again to unlock.[/swarm.dim]")
                elif cmd == "project":
                    if not arg or arg == "list":
                        recents = _load_recent_projects()
                        if recents:
                            console.print("[swarm.accent]Recent projects:[/swarm.accent]")
                            for i, rp in enumerate(recents, 1):
                                marker = " [bold](current)[/bold]" if rp == str(PROJECT_DIR.resolve()) else ""
                                console.print(f"  [bold]{i}.[/bold] {rp}{marker}")
                        else:
                            console.print("[swarm.dim]No recent projects.[/swarm.dim]")
                    else:
                        target = arg.strip()
                        # Support numeric shortcut from recent list
                        if target.isdigit():
                            recents = _load_recent_projects()
                            idx = int(target) - 1
                            if 0 <= idx < len(recents):
                                target = recents[idx]
                            else:
                                console.print("[swarm.error]Invalid project number.[/swarm.error]")
                                continue
                        if _switch_project(target):
                            conversation[0] = {"role": "system", "content": SYSTEM_PROMPT}
                elif cmd == "doctor":
                    _run_doctor()
                elif cmd == "dashboard":
                    dashboard()
                elif cmd == "metrics":
                    metrics = get_bus().get_metrics()
                    console.print()
                    console.print("[swarm.accent]Session Metrics[/swarm.accent]")
                    console.print(Rule(style="dim"))
                    console.print(f"  [bold]Prompt Tokens:[/bold]     {metrics['prompt_tokens']:,}")
                    console.print(f"  [bold]Completion Tokens:[/bold] {metrics['completion_tokens']:,}")
                    console.print(f"  [bold]Total Tokens:[/bold]      {metrics['total_tokens']:,}")
                    console.print()
                elif cmd == "abort":
                    abort()
                elif cmd == "self-improve":
                    if not arg:
                        console.print("[swarm.warning]Usage: /self-improve <description of improvement>[/swarm.warning]")
                        continue
                    shadow_dir = PROJECT_DIR / ".grokswarm" / "shadow"
                    shadow_dir.mkdir(parents=True, exist_ok=True)
                    shadow_file = shadow_dir / "main.py"
                    shutil.copy2(PROJECT_DIR / "main.py", shadow_file)
                    state.self_improve_active = True
                    console.print(f"[swarm.accent]Shadow copy created:[/swarm.accent] [dim]{shadow_file.relative_to(PROJECT_DIR)}[/dim]")
                    console.print("[swarm.dim]The swarm will edit and test the shadow copy safely...[/swarm.dim]\n")
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
                    conversation = _trim_conversation(conversation)
                    full_response = _stream_with_tools(conversation)
                    if session_name:
                        save_session(session_name, conversation)
                    state.self_improve_active = False
                    console.print("\n[bold yellow]Self-Improvement Complete.[/bold yellow]")
                    console.print(f"[dim]Review changes: run_shell 'python -c \"import difflib,pathlib; a=pathlib.Path(\\\"main.py\\\").read_text().splitlines(); b=pathlib.Path(\\\".grokswarm/shadow/main.py\\\").read_text().splitlines(); print(chr(10).join(difflib.unified_diff(a,b,lineterm=\\\"\\\",n=3)))'[/dim]")
                    if Confirm.ask("Promote shadow copy to main.py?", default=False):
                        # Verify shadow compiles before promotion
                        check = subprocess.run(["python", "-m", "py_compile", str(shadow_file)], capture_output=True, text=True)
                        if check.returncode != 0:
                            console.print(f"[bold red]Shadow copy has syntax errors — promotion blocked.[/bold red]")
                            console.print(f"[dim]{check.stderr[:300]}[/dim]")
                        else:
                            # A6+B10: run tests in isolated temp dir with shadow copy
                            test_file = PROJECT_DIR / "test_grokswarm.py"
                            if test_file.exists():
                                console.print("[swarm.dim]Running test suite against shadow copy (isolated)...[/swarm.dim]")
                                import tempfile as _tf
                                with _tf.TemporaryDirectory(prefix="grokswarm_test_") as iso_dir:
                                    iso = Path(iso_dir)
                                    shutil.copy2(shadow_file, iso / "main.py")
                                    # Copy any other files created in shadow dir
                                    for f in shadow_dir.glob("*.py"):
                                        if f.name != "main.py":
                                            shutil.copy2(f, iso / f.name)
                                    shutil.copy2(test_file, iso / "test_grokswarm.py")
                                    # Copy any supporting test fixtures
                                    for f in PROJECT_DIR.glob("conftest*.py"):
                                        shutil.copy2(f, iso / f.name)
                                    test_check = subprocess.run(
                                        [sys.executable, "-m", "pytest", "test_grokswarm.py", "-x", "-q"],
                                        capture_output=True, text=True, cwd=str(iso), timeout=120
                                    )
                                if test_check.returncode != 0:
                                    console.print(f"[bold red]Tests failed — promotion blocked.[/bold red]")
                                    console.print(f"[dim]{test_check.stdout[-500:] if test_check.stdout else test_check.stderr[:300]}[/dim]")
                                    console.print("[swarm.dim]Shadow copy preserved. Fix tests before promoting.[/swarm.dim]")
                                else:
                                    console.print("[swarm.accent]Tests passed (isolated)![/swarm.accent]")
                                    shutil.copy2(shadow_file, PROJECT_DIR / "main.py")
                                    # Also copy any other new files created in shadow dir
                                    for f in shadow_dir.glob("*.py"):
                                        if f.name != "main.py":
                                            shutil.copy2(f, PROJECT_DIR / f.name)
                                    console.print("[swarm.accent]main.py updated! Restart to load new features.[/swarm.accent]")
                            else:
                                shutil.copy2(shadow_file, PROJECT_DIR / "main.py")
                                for f in shadow_dir.glob("*.py"):
                                    if f.name != "main.py":
                                        shutil.copy2(f, PROJECT_DIR / f.name)
                                console.print("[swarm.accent]main.py updated! Restart to load new features.[/swarm.accent]")
                    else:
                        console.print("[swarm.dim]Shadow copy preserved at .grokswarm/shadow/main.py[/swarm.dim]")
                else:
                    console.print(f"[swarm.dim]Unknown command: /{cmd} -- type /help[/swarm.dim]")
                continue

            # -- Exit without slash --
            if user_input.lower() in ["exit", "quit", "q"]:
                if session_name:
                    save_session(session_name, conversation)
                    console.print(f"[swarm.dim]Session '{session_name}' saved.[/swarm.dim]")
                console.print("[swarm.dim]Goodbye.[/swarm.dim]")
                break

            # -- Conversational AI with streaming + tool calling --
            conversation.append({"role": "user", "content": user_input})
            conversation = _trim_conversation(conversation)
            console.print()

            full_response = _stream_with_tools(conversation)

            if session_name:
                save_session(session_name, conversation)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[swarm.dim]Interrupted. Type /quit to exit.[/swarm.dim]")
        except EOFError:
            if session_name:
                save_session(session_name, conversation)
                console.print(f"[swarm.dim]Session '{session_name}' saved.[/swarm.dim]")
            console.print("[swarm.dim]Goodbye.[/swarm.dim]")
            break
        except Exception as e:
            console.print(f"[swarm.error]Error: {e}[/swarm.error]")


def _run_doctor():
    """U2: Check environment health."""
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
        # L2: check if chromium browser binary is actually installed
        _pw_dir = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH",
                       os.environ.get("LOCALAPPDATA", "")))
        if _pw_dir.name != "ms-playwright":
            _pw_dir = _pw_dir / "ms-playwright"
        if not _pw_dir.exists():
            _pw_dir = Path.home() / ".cache" / "ms-playwright"  # Linux/Mac fallback
        if _pw_dir.exists() and any(_pw_dir.glob("chromium-*")):
            checks.append(("playwright", "installed + chromium ready", True))
        else:
            checks.append(("playwright", "installed (run: playwright install chromium)", False))
    except ImportError:
        checks.append(("playwright", "not installed (optional)", False))
    checks.append(("project dir", str(PROJECT_DIR), PROJECT_DIR.is_dir()))
    cfg_exists = (PROJECT_DIR / ".grokswarm.yml").exists()
    checks.append((".grokswarm.yml", "found" if cfg_exists else "not found (optional)", cfg_exists))
    console.print()
    console.print("[swarm.accent]Doctor — Environment Check[/swarm.accent]")
    console.print(Rule(style="dim"))
    for label, value, ok in checks:
        icon = "[bold green]\u2713[/bold green]" if ok else "[bold yellow]\u25cb[/bold yellow]"
        console.print(f"  {icon} [bold]{label:<20}[/bold] {value}")
    console.print()


def _show_help():
    console.print()
    console.print("[swarm.accent]Commands[/swarm.accent]")
    console.print(Rule(style="dim"))
    help_items = [
        # -- File operations --
        ("/help", "Show this help"),
        ("/list [path]", "List project directory"),
        ("/read <file>", "Read file contents"),
        ("/write <file>", "Write/create file (with approval)"),
        ("/run <cmd>", "Run shell command (with approval)"),
        ("/search <query>", "Search files by name in project"),
        ("/grep <pattern> [path]", "Search inside files for text"),
        # -- Git & web --
        ("/git", "Git status (log, diff, branch)"),
        ("/web <query>", "Search the web (xAI live)"),
        ("/x <query>", "Search X/Twitter posts (xAI live)"),
        ("/browse <url>", "Fetch URL content (Playwright)"),
        # -- Testing & editing --
        ("/test [cmd]", "Run project tests (auto-detect framework)"),
        ("/undo", "Undo last file edit (multi-level)"),
        # -- Modes --
        ("/trust", "Toggle trust mode (auto-approve)"),
        ("/readonly", "Toggle read-only mode (block writes)"),
        ("/project <path>", "Switch project directory"),
        ("/doctor", "Check environment health"),
        ("/dashboard", "Open live TUI dashboard"),
        ("/metrics", "Show token usage and cost metrics"),
        # -- AI & agents --
        ("/self-improve <desc>", "Improve own source code (shadow + auto-test)"),
        ("/swarm <task>", "Run multi-agent supervisor"),
        ("/abort", "Abort currently running swarm"),
        ("/experts", "List available experts"),
        ("/skills", "List available skills"),
        # -- Session & meta --
        ("/context", "Show project context (refresh to rescan)"),
        ("/session", "Manage sessions (list/save/load/delete)"),
        ("/clear", "Clear conversation & screen"),
        ("/quit", "Exit"),
    ]
    for cmd, desc in help_items:
        console.print(f"  [bold]{cmd:<24}[/bold] [dim]{desc}[/dim]")
    console.print()


def _show_context(arg: str = ""):
    """Display current project context."""
    console.print()
    console.print("[swarm.accent]Project Context[/swarm.accent]")
    console.print(Rule(style="dim"))
    console.print(f"  [bold]Project:[/bold]   {PROJECT_CONTEXT['project_name']}")
    console.print(f"  [bold]Directory:[/bold] {PROJECT_CONTEXT['project_dir']}")
    console.print()
    console.print("[swarm.accent]File Tree[/swarm.accent]")
    console.print(Rule(style="dim"))
    console.print(f"[dim]{PROJECT_CONTEXT['tree']}[/dim]")
    console.print()
    key_files = PROJECT_CONTEXT.get("key_files", {})
    if key_files:
        console.print(f"[swarm.accent]Key Files Loaded ({len(key_files)})[/swarm.accent]")
        console.print(Rule(style="dim"))
        for fname in key_files:
            size = len(key_files[fname])
            console.print(f"  [bold]{fname:<25}[/bold] [dim]{size:,} chars[/dim]")
    else:
        console.print("[swarm.dim]  No key files found.[/swarm.dim]")

    # Language stats
    lang_stats = PROJECT_CONTEXT.get("language_stats", {})
    if lang_stats:
        console.print()
        console.print(f"[swarm.accent]Languages[/swarm.accent]")
        console.print(Rule(style="dim"))
        stats_str = "  ".join(f"[bold]{ext}[/bold]:{count}" for ext, count in list(lang_stats.items())[:10])
        console.print(f"  {stats_str}")

    # Code structure
    code_struct = PROJECT_CONTEXT.get("code_structure", {})
    if code_struct:
        console.print()
        console.print(f"[swarm.accent]Code Structure ({len(code_struct)} files)[/swarm.accent]")
        console.print(Rule(style="dim"))
        for filepath, defs in list(code_struct.items())[:10]:
            console.print(f"  [bold]{filepath}[/bold]")
            for d in defs[:5]:
                console.print(f"    [dim]{d}[/dim]")
            if len(defs) > 5:
                console.print(f"    [dim]... ({len(defs) - 5} more)[/dim]")

    console.print()
    if arg == "refresh":
        console.print("[swarm.accent]Context refreshed from disk.[/swarm.accent]")
        console.print()


def _handle_session_command(arg: str, conversation: list, current_session: str | None) -> str | None:
    """Handle /session subcommands. Returns new session name if changed, else None."""
    parts = arg.split(maxsplit=1) if arg else []
    subcmd = parts[0].lower() if parts else ""
    subarg = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "list" or not subcmd:
        sessions = list_sessions()
        if not sessions:
            console.print("[swarm.dim]No saved sessions.[/swarm.dim]")
            return None
        console.print()
        console.print("[swarm.accent]Saved Sessions[/swarm.accent]")
        console.print(Rule(style="dim"))
        for s in sessions:
            marker = " [bold green]< active[/bold green]" if current_session and s["name"] == current_session else ""
            ts = s["updated"][:16].replace("T", " ") if s["updated"] != "unknown" else "?"
            console.print(f"  [bold]{s['name']:<20}[/bold] [dim]{s['messages']} msgs  *  {ts}[/dim]{marker}")
        console.print()
        return None
    elif subcmd == "save":
        if not subarg:
            if current_session:
                subarg = current_session
            else:
                console.print("[swarm.warning]Usage: /session save <name>[/swarm.warning]")
                return None
        save_session(subarg, conversation)
        console.print(f"[swarm.accent]Session '[bold]{subarg}[/bold]' saved ({len([m for m in conversation if m['role'] != 'system'])} messages).[/swarm.accent]")
        return subarg
    elif subcmd == "load":
        if not subarg:
            console.print("[swarm.warning]Usage: /session load <name>[/swarm.warning]")
            return None
        msgs = load_session(subarg)
        if msgs:
            conversation.clear()
            conversation.append({"role": "system", "content": SYSTEM_PROMPT})
            conversation.extend(msgs)
            console.print(f"[swarm.accent]Loaded session '[bold]{subarg}[/bold]' ({len(msgs)} messages). Auto-saving enabled.[/swarm.accent]")
            return subarg
        else:
            console.print(f"[swarm.warning]Session '{subarg}' not found.[/swarm.warning]")
            return None
    elif subcmd == "delete":
        if not subarg:
            console.print("[swarm.warning]Usage: /session delete <name>[/swarm.warning]")
            return None
        if delete_session(subarg):
            console.print(f"[swarm.accent]Session '{subarg}' deleted.[/swarm.accent]")
        else:
            console.print(f"[swarm.warning]Session '{subarg}' not found.[/swarm.warning]")
        return None
    else:
        console.print("[swarm.dim]Usage: /session [list|save|load|delete] <name>[/swarm.dim]")
        return None


# -- All original rich commands (fully restored) --
@app.command()
def swarm(description: str):
    plan = run_supervisor(description)
    bus = get_bus()
    bus.clear()  # clear old messages for new run
    bus.post("supervisor", json.dumps(plan), kind="plan")
    for expert_name in plan.get("experts", ["assistant"]):
        if bus.check_abort():
            console.print("[swarm.warning]Swarm aborted by dashboard.[/swarm.warning]")
            break
        run_expert(expert_name, description, bus=bus)
    console.print("[bold green]Swarm task complete.[/bold green]")

@app.command("team-save")
def team_save(name: str):
    team = {"name": name, "experts": list_experts(), "created": datetime.now().isoformat()}
    (TEAMS_DIR / f"{name.lower()}.yaml").write_text(yaml.dump(team))
    console.print(f"[green]Team '{name}' saved.[/green]")

@app.command("team-list")
def team_list():
    table = Table(title="Saved Teams")
    table.add_column("Team Name")
    for t in TEAMS_DIR.glob("*.yaml"):
        table.add_row(t.stem)
    console.print(table)

@app.command("team-run")
def team_run(name: str, task_desc: str):
    team_file = TEAMS_DIR / f"{name.lower()}.yaml"
    if not team_file.exists():
        console.print(f"[red]Team {name} not found.[/red]")
        return
    data = yaml.safe_load(team_file.read_text())
    console.print(f"[bold cyan]Running Team:[/bold cyan] {data['name']}")
    bus = get_bus()
    bus.clear()
    for expert in data.get("experts", []):
        if bus.check_abort():
            console.print("[swarm.warning]Team run aborted by dashboard.[/swarm.warning]")
            break
        run_expert(expert, task_desc, bus=bus)

@app.command()
def task(description: str):
    console.print(f"[bold green]Executing task:[/bold green] {description}")
    run_expert("assistant", description)

@app.command()
def expert(name: str, task_desc: str):
    run_expert(name, task_desc)

@app.command("skills-list")
def skills_list():
    table = Table(title="Skill Registry")
    table.add_column("Skill")
    for s in list_skills():
        table.add_row(s)
    console.print(table)

@app.command("experts-list")
def experts_list():
    table = Table(title="Expert Registry")
    table.add_column("Expert")
    for e in list_experts():
        table.add_row(e)
    console.print(table)

@app.command()
def create_skill(name: str, description: str):
    skill = {"name": name, "description": description, "version": "1.0"}
    (SKILLS_DIR / f"{name.lower()}.yaml").write_text(yaml.dump(skill))
    console.print(f"[green]Skill '{name}' created.[/green]")

@app.command()
def create_expert(name: str, mindset: str):
    expert = {"name": name, "mindset": mindset, "objectives": []}
    (EXPERTS_DIR / f"{name.lower()}.yaml").write_text(yaml.dump(expert))
    console.print(f"[green]Expert '{name}' created.[/green]")


# ---------------------------------------------------------------------------
# A5 — Live TUI dashboard
# ---------------------------------------------------------------------------

def _build_dashboard() -> Layout:
    """Build a Rich Layout snapshot of the current GrokSwarm state."""
    # -- Project info panel --
    bus = get_bus()
    metrics = bus.get_metrics()
    total_tokens = metrics["total_tokens"]
    # Rough cost estimate for grok-4-1-fast-reasoning (placeholder: $0.002/1k prompt, $0.004/1k completion)
    # We'll just show tokens for now, or a generic cost if we want.
    # Let's just show tokens.
    proj_lines = [
        f"[bold]Directory:[/bold]  {PROJECT_DIR}",
        f"[bold]Model:[/bold]      {MODEL}",
        f"[bold]Version:[/bold]    {VERSION}",
        f"[bold]Trust:[/bold]      {'[green]ON[/green]' if state.trust_mode else '[red]OFF[/red]'}",
        f"[bold]Read-only:[/bold]  {'[yellow]ON[/yellow]' if state.read_only else 'OFF'}",
    ]
    project_panel = Panel("\n".join(proj_lines), title="Project", border_style="cyan")

    metrics_lines = [
        f"[bold]Prompt Tokens:[/bold]     {metrics['prompt_tokens']:,}",
        f"[bold]Completion Tokens:[/bold] {metrics['completion_tokens']:,}",
        f"[bold]Total Tokens:[/bold]      {total_tokens:,}",
    ]
    metrics_panel = Panel("\n".join(metrics_lines), title="Session Metrics", border_style="magenta")

    # -- Experts panel --
    expert_files = sorted(EXPERTS_DIR.glob("*.yaml"))
    expert_names = [f.stem for f in expert_files] or ["(none)"]
    experts_panel = Panel("\n".join(expert_names), title=f"Experts ({len(expert_files)})", border_style="green")

    # -- Skills panel --
    skill_files = sorted(SKILLS_DIR.glob("*.yaml"))
    skill_names = [f.stem for f in skill_files] or ["(none)"]
    skills_panel = Panel("\n".join(skill_names), title=f"Skills ({len(skill_files)})", border_style="green")

    # -- Sessions panel --
    sess_files = sorted(SESSIONS_DIR.glob("*.json"))[-10:]  # last 10
    sess_names = [f.stem for f in sess_files] or ["(none)"]
    sessions_panel = Panel("\n".join(sess_names), title=f"Sessions ({len(list(SESSIONS_DIR.glob('*.json')))})", border_style="yellow")

    # -- Recent edits panel --
    recent = state.edit_history[-5:] if state.edit_history else []
    edit_lines = []
    for path, content in reversed(recent):
        label = "new" if content is None else "edit"
        edit_lines.append(f"[dim]{label}[/dim]  {path}")
    edits_panel = Panel("\n".join(edit_lines) or "(no edits yet)", title="Recent Edits", border_style="magenta")

    # -- Teams panel --
    team_files = sorted(TEAMS_DIR.glob("*.yaml"))
    team_names = [f.stem for f in team_files] or ["(none)"]
    teams_panel = Panel("\n".join(team_names), title=f"Teams ({len(team_files)})", border_style="blue")

    # -- SwarmBus Feed panel --
    msgs = bus.read(limit=15)
    feed_lines = []
    for m in msgs:
        ts = m['ts'].split()[-1]  # just the time part
        body = m['body'].replace('\n', ' ')
        if len(body) > 100:
            body = body[:97] + "..."
        if m['kind'] == 'plan':
            feed_lines.append(f"[dim]{ts}[/dim] [bold cyan]Plan:[/bold cyan] {body}")
        elif m['kind'] == 'abort':
            feed_lines.append(f"[dim]{ts}[/dim] [bold red]ABORT SIGNAL[/bold red]")
        else:
            feed_lines.append(f"[dim]{ts}[/dim] [[bold green]{m['sender']}[/bold green]\u2192{m['recipient']}] {body}")
    feed_panel = Panel("\n".join(feed_lines) or "(no active swarm messages)", title="Live Swarm Feed", border_style="cyan")

    # -- Compose layout --
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=7),
        Layout(name="middle", size=10),
        Layout(feed_panel, name="feed"),
        Layout(name="bottom", size=8),
    )
    layout["header"].split_row(
        Layout(project_panel, name="project"),
        Layout(metrics_panel, name="metrics"),
    )
    layout["middle"].split_row(
        Layout(experts_panel, name="experts"),
        Layout(skills_panel, name="skills"),
        Layout(teams_panel, name="teams"),
    )
    layout["bottom"].split_row(
        Layout(sessions_panel, name="sessions"),
        Layout(edits_panel, name="edits"),
    )
    return layout


@app.command()
def abort():
    """Send an abort signal to stop any currently running swarm."""
    bus = get_bus()
    bus.post("user", "Abort requested", kind="abort")
    console.print("[bold red]Abort signal sent to SwarmBus.[/bold red]")


@app.command()
def dashboard():
    """Live TUI dashboard — shows project state, experts, skills, sessions."""
    console.print("[swarm.accent]GrokSwarm Dashboard[/swarm.accent]  (press Ctrl+C to exit, run 'grokswarm abort' to stop swarms)\n")
    try:
        with Live(_build_dashboard(), console=console, refresh_per_second=2, screen=True) as live:
            while True:
                time.sleep(0.5)
                live.update(_build_dashboard())
    except KeyboardInterrupt:
        console.print("\n[swarm.dim]Dashboard closed.[/swarm.dim]")


if __name__ == "__main__":
    app()