"""Project scanning, symbol index, AST, context cache, safe_path."""

import os
import re
import ast
import json
import fnmatch
from pathlib import Path

import grokswarm.shared as shared

# -- Project Context Constants --
CONTEXT_FILES = [
    "README.md", "readme.md", "README.rst",
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "Cargo.toml", "go.mod", "Makefile",
    ".env.example", "GOALS.md",
]
MAX_CONTEXT_FILE_SIZE = 8000
MAX_TREE_DEPTH = 4
MAX_TREE_FILES = 150
MAX_SCAN_FILES = 500
MAX_INDEX_FILE_SIZE = 500_000
IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info", "chroma_db", ".next", "target",
}
_IGNORE_PATTERNS = [p for p in IGNORE_DIRS if "*" in p]
_IGNORE_LITERALS = IGNORE_DIRS - set(_IGNORE_PATTERNS)


def _should_ignore(name: str) -> bool:
    if name in _IGNORE_LITERALS:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in _IGNORE_PATTERNS)


CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb"}

BASE_SYSTEM_PROMPT = """You are Grok Swarm, an expert AI assistant for Aaron. You are concise, direct, and helpful.

COMMUNICATION RULES:
- Communicate technically and precisely. No hype, no enthusiasm, no filler phrases.
- NEVER use emojis in responses. Not one. No checkmarks, no rockets, no stars, no weather icons, nothing.
- Do not say "Fixed!", "Done!", "Enjoy!", "Perfect!", "Runs perfect!", "polished!", or similar.
- State what you changed and what the result was. That's it.
- Do not claim you tested or verified something unless you actually ran it and saw the output.
- If you only ran pytest, say "unit tests pass" — do NOT say "verified" or "tested" or "confirmed working".
- If the user asks you to test something, actually run it (run_shell) and report the real output.
- Avoid web_search for standard programming knowledge. You already know Python, APIs, frameworks. Only search when you genuinely need current/unknown information.

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

END-TO-END TESTING:
- Unit tests verify logic. E2E tests verify what the user actually sees. BOTH are required.
- After fixing a visual or UI bug, do NOT just run unit tests and report success. You must verify the rendered output.
- Use run_app_capture to launch CLI or TUI apps with a timeout and capture their stdout/stderr. This lets you see what the app actually outputs.
- Use capture_tui_screenshot for Textual apps — it runs headless, captures a screenshot, and you can analyze it with analyze_image.
- For web apps, use screenshot_page + analyze_image to verify rendered pages visually.
- When writing tests for TUI apps (Textual, Rich, curses), write async integration tests using the framework's test harness:
  - Textual: `async with app.run_test() as pilot:` — renders headlessly, query widgets, simulate input, assert content.
  - Rich: capture Console output with `console = Console(file=StringIO())` and assert on the string.
- Key principle: if a bug is visual (wrong layout, missing content, garbled text), the test that proves it fixed must inspect the RENDERED output, not just data structures.
- After claiming a fix, verify it: run the app with run_app_capture, take a screenshot with capture_tui_screenshot, or write an integration test that exercises the actual UI.

FIX-VERIFY-RETRY LOOP (MANDATORY):
- When capture_tui_screenshot or run_app_capture reveals an error:
  1. READ the full error traceback. Identify the root cause (import error? CSS error? runtime error?).
  2. Make ONE targeted fix based on the error message.
  3. VERIFY the fix by re-running capture_tui_screenshot or run_app_capture IMMEDIATELY.
  4. If it still fails, go back to step 1. Do NOT make another guess without verifying.
  5. NEVER commit code that you have not verified runs successfully.
- If you've tried 3 fixes and the app still crashes, STOP and report the failure to the user with the traceback.

COMMON TEXTUAL APP BUGS (learn these so you fix correctly on the first try):
- @on(MessageType, "css_selector") requires the message to have a 'control' attribute. Key events do NOT have one. Instead of @on(Key, "c"), use a method named key_c(self, event: Key) or handle in on_key(self, event: Key) with if event.key == "c".
- Textual CSS is FLAT — no SCSS-style nesting. Wrong: `DataTable { table { border: ... } }`. Right: `DataTable { border: ... }` on separate lines.
- Textual CSS does NOT support: opacity values in background (background: #1a1a1a 40%), percentage in text-style (text-style: bold 200%), or most shorthand CSS properties.
- Input() does NOT have a 'variants' parameter. The Button widget has 'variant' (singular). Check widget constructors carefully.
- Header(markup=True) is NOT a valid parameter in modern Textual. Remove it.
- When @work decorators fail silently, the UI shows stale/empty data. Add try/except inside @work methods and call self.notify() on error.

ERROR RECOVERY:
- After edit_file or write_file, the system automatically checks for syntax errors. If lint errors are reported, you MUST fix them immediately using edit_file before doing anything else.
- If a tool call fails (e.g. edit_file can't find old_text), read the error message carefully, use read_file to inspect the current file state, and retry with corrected arguments.
- Never leave a file in a broken state. If your edit introduced errors, fix them before responding to the user.

Format responses in markdown when appropriate. Be specific and actionable. Never use emojis."""


def _iter_project_files(project_dir: Path, extensions: set[str] | None = None, max_files: int = MAX_SCAN_FILES):
    count = 0
    for dirpath, dirnames, filenames in os.walk(project_dir):
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


def _scan_tree(root: Path, prefix: str = "", depth: int = 0) -> list[str]:
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
    symbols = []
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        shared.console.print(f"  [swarm.dim]AST: syntax error in {file_path.name}: {e.msg} line {e.lineno}[/swarm.dim]")
        return symbols
    except Exception:
        return symbols

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
                name = next((g for g in m.groups() if g), None)
                if name == symbol_name:
                    results.append({"file": rel, "line": i, "type": "definition"})
    return results


def find_symbol(name: str) -> str:
    results = []
    for p, rel in _iter_project_files(shared.PROJECT_DIR, extensions=CODE_EXTENSIONS):
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
    results = []
    name_lower = name.lower()
    for p, rel in _iter_project_files(shared.PROJECT_DIR, extensions=CODE_EXTENSIONS):
        ext = p.suffix.lower()
        if ext == ".py":
            imports = _extract_python_imports(p)
            for imp in imports:
                if name_lower in imp.lower().split("."):
                    results.append(f"  {rel} (import {imp})")
                    break
            else:
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(text.splitlines(), 1):
                        if re.search(r'\b' + re.escape(name) + r'\b', line):
                            results.append(f"  {rel}:{i}: {line.rstrip()[:100]}")
                            break
                except Exception:
                    pass
        else:
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


def _file_size_ok(p: Path) -> bool:
    try:
        return p.stat().st_size <= MAX_INDEX_FILE_SIZE
    except OSError:
        return False


def _build_import_graph(project_dir: Path) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}
    py_modules: set[str] = set()
    py_files: list[tuple[Path, Path]] = [
        (p, rel) for p, rel in _iter_project_files(project_dir, extensions={".py"})
        if _file_size_ok(p)
    ]
    for p, rel in py_files:
        module_name = str(rel.with_suffix("")).replace(os.sep, ".")
        py_modules.add(module_name)
        for part in rel.parent.parts:
            py_modules.add(part)
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


def _detect_language_stats(project_dir: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    for p, rel in _iter_project_files(project_dir, max_files=MAX_SCAN_FILES):
        ext = p.suffix.lower()
        if ext:
            stats[ext] = stats.get(ext, 0) + 1
    return dict(sorted(stats.items(), key=lambda x: -x[1])[:15])


def _build_deep_symbol_index(project_dir: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for p, rel in _iter_project_files(project_dir, extensions=CODE_EXTENSIONS):
        try:
            if p.stat().st_size > MAX_INDEX_FILE_SIZE:
                continue
        except OSError:
            continue
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
                    sym_name = next((g for g in m.groups() if g), None)
                    if sym_name:
                        defs.append(f"  L{i}: {sym_name}")
                if len(defs) >= 15:
                    defs.append("  ... (truncated)")
                    break
        if defs:
            index[str(rel)] = defs
    return index


def scan_project_context(project_dir: Path) -> dict:
    context = {
        "project_dir": str(project_dir),
        "project_name": project_dir.name,
        "tree": "",
        "key_files": {},
        "code_structure": {},
        "language_stats": {},
    }
    tree_lines = _scan_tree(project_dir)
    context["tree"] = f"{project_dir.name}/\n" + "\n".join(tree_lines) if tree_lines else "(empty)"
    for fname in CONTEXT_FILES:
        fpath = project_dir / fname
        content = _read_context_file(fpath)
        if content:
            context["key_files"][fname] = content
    context["code_structure"] = _build_deep_symbol_index(project_dir)
    context["language_stats"] = _detect_language_stats(project_dir)
    context["import_graph"] = _build_import_graph(project_dir)
    return context


def _context_cache_key(project_dir: Path) -> Path:
    import hashlib
    h = hashlib.sha256(str(project_dir.resolve()).encode()).hexdigest()[:16]
    return shared.CONTEXT_CACHE_DIR / f"{project_dir.name}_{h}.json"


def _project_mtime(project_dir: Path) -> float:
    latest = 0.0
    try:
        for p, _ in _iter_project_files(project_dir, max_files=2000):
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
    try:
        data = dict(context)
        data["_mtime"] = _project_mtime(project_dir)
        _context_cache_key(project_dir).write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def scan_project_context_cached(project_dir: Path) -> dict:
    cached = _load_cached_context(project_dir)
    if cached:
        shared.console.print("[swarm.dim]  context loaded from cache[/swarm.dim]")
        return cached
    context = scan_project_context(project_dir)
    _save_context_cache(project_dir, context)
    return context


def format_context_for_prompt(ctx: dict) -> str:
    parts = [f"\n--- PROJECT CONTEXT ---"]
    parts.append(f"Project: {ctx['project_name']}")
    parts.append(f"Directory: {ctx['project_dir']}")
    lang_stats = ctx.get("language_stats", {})
    if lang_stats:
        stats_str = ", ".join(f"{ext}: {count}" for ext, count in list(lang_stats.items())[:10])
        parts.append(f"\nLanguages: {stats_str}")
    parts.append(f"\nFile tree:\n```\n{ctx['tree']}\n```")
    for fname, content in ctx["key_files"].items():
        parts.append(f"\n{fname}:\n```\n{content}\n```")
    code_struct = ctx.get("code_structure", {})
    if code_struct:
        parts.append("\nCode structure (symbols):")
        for filepath, defs in list(code_struct.items())[:20]:
            parts.append(f"\n{filepath}:")
            parts.extend(defs)
    import_graph = ctx.get("import_graph", {})
    if import_graph:
        parts.append("\nImport graph (local dependencies):")
        for filepath, deps in list(import_graph.items())[:30]:
            parts.append(f"  {filepath} -> {', '.join(deps)}")
    parts.append("--- END PROJECT CONTEXT ---")
    return "\n".join(parts)


def build_system_prompt(project_context: dict | None = None) -> str:
    prompt = BASE_SYSTEM_PROMPT
    if project_context:
        prompt += "\n" + shared._sanitize_surrogates(format_context_for_prompt(project_context))
    if shared.PROJECT_DIR.resolve() != shared.GROKSWARM_HOME:
        prompt += ("\n\n--- SELF-KNOWLEDGE ---\n"
                   "You are GrokSwarm. Your own source code is accessible read-only via the @grokswarm/ prefix.\n"
                   "  - read_file with path '@grokswarm/main.py' to read your own source\n"
                   "  - list_directory with path '@grokswarm/' to see your own files\n"
                   "Use this for self-reference — understanding your own capabilities, reviewing your implementation, "
                   "or when asked about how you work. All @grokswarm/ access is read-only.\n"
                   "--- END SELF-KNOWLEDGE ---")
    return prompt


def _incremental_context_refresh(file_path_str: str):
    try:
        fp = shared.PROJECT_DIR / file_path_str
        if not fp.is_file():
            return
        rel = str(fp.relative_to(shared.PROJECT_DIR))
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
            cs = shared.PROJECT_CONTEXT.setdefault("code_structure", {})
            if defs:
                cs[rel] = defs
            elif rel in cs:
                del cs[rel]
        shared.SYSTEM_PROMPT = build_system_prompt(shared.PROJECT_CONTEXT)
    except Exception:
        pass


def _safe_path(path: str) -> Path | None:
    candidate = shared.PROJECT_DIR / path
    try:
        cur = shared.PROJECT_DIR
        for part in candidate.relative_to(shared.PROJECT_DIR).parts:
            cur = cur / part
            if cur.is_symlink():
                real_target = cur.resolve()
                if not real_target.is_relative_to(shared.PROJECT_DIR.resolve()):
                    return None
        full = candidate.resolve()
    except (ValueError, OSError):
        return None
    if not full.is_relative_to(shared.PROJECT_DIR.resolve()):
        return None
    return full


def _grokswarm_read_path(path: str) -> Path | None:
    rel = path[len("@grokswarm/"):]
    candidate = (shared.GROKSWARM_HOME / rel).resolve()
    if not candidate.is_relative_to(shared.GROKSWARM_HOME):
        return None
    return candidate
