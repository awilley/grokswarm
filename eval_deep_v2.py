"""
eval_deep_v2.py — Real-World & Adversarial Evaluation Tasks

Phase 3: Real-world complexity (R1-R3, J1-J2)
Phase 4: Failure mode / resilience testing (K1-K4)

Builds on eval_deep.py framework. Same data structures, scoring, and runners.

Two modes:
  1. Unit eval (mocked API):  python -m pytest eval_deep_v2.py -v
  2. Live eval (real API):    python eval_deep_v2.py --live [--task R1]
"""

import ast
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import pytest

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

os.environ.setdefault("XAI_API_KEY", "test-key-for-eval")
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import grokswarm.shared as shared
from grokswarm.agents import run_expert, SwarmBus
from grokswarm.guardrails import Orchestrator, reset_model_tiers

from eval_deep import (
    WeightedCheck, DeepEvalTask, RunMetrics, ComparativeResult,
    wcheck, _run_weighted_checks, _compute_verdict,
    _compute_efficiency_scores, _save_eval_scores,
    _setup_deep_workspace, _run_single_agent, _run_swarm,
    run_comparative, run_learning_eval, format_deep_report,
    check_file_exists, check_file_contains, check_file_not_contains,
    check_python_compiles, check_pytest_passes, check_function_exists,
    check_output_matches, check_cli_args, check_cli_exitcode,
    check_bug_fixed, check_class_exists, check_import_works,
    CATEGORY_MULTIPLIERS,
)

# ---------------------------------------------------------------------------
# Load corpus files
# ---------------------------------------------------------------------------

_CORPUS_DIR = Path(__file__).parent / "eval_corpus"

MONOLITH_CODE = (_CORPUS_DIR / "monolith.py").read_text(encoding="utf-8")
TEST_MONOLITH_CODE = (_CORPUS_DIR / "test_monolith.py").read_text(encoding="utf-8")
TASK_QUEUE_CODE = (_CORPUS_DIR / "task_queue.py").read_text(encoding="utf-8")
TEST_MARKDOWN_CODE = (_CORPUS_DIR / "test_markdown.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Additional check functions for V2 tasks
# ---------------------------------------------------------------------------

def check_no_god_functions(path: str, max_lines: int = 50):
    """Check that no function in the file exceeds max_lines."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        try:
            tree = ast.parse(f.read_text(errors="replace"))
        except SyntaxError as e:
            return False, f"Syntax error in {path}: {e}"
        god_fns = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                if length > max_lines:
                    god_fns.append(f"{node.name}({length} lines)")
        if god_fns:
            return False, f"God functions in {path}: {', '.join(god_fns)}"
        return True, f"No function > {max_lines} lines in {path}"
    _check.__name__ = f"no_god_functions({path}, {max_lines})"
    return _check


def check_has_docstrings(path: str, min_coverage: float = 0.5):
    """Check that at least min_coverage fraction of public functions have docstrings."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        try:
            tree = ast.parse(f.read_text(errors="replace"))
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        total = 0
        documented = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    total += 1
                    if (node.body and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)):
                        documented += 1
        if total == 0:
            return True, f"No public functions in {path}"
        ratio = documented / total
        if ratio >= min_coverage:
            return True, f"Docstring coverage: {ratio:.0%} ({documented}/{total}) in {path}"
        return False, f"Low docstring coverage: {ratio:.0%} ({documented}/{total}) in {path}"
    _check.__name__ = f"has_docstrings({path}, {min_coverage:.0%})"
    return _check


def check_max_file_lines(path: str, max_lines: int = 100):
    """Check that a file is under max_lines."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        lines = len(f.read_text(errors="replace").splitlines())
        if lines <= max_lines:
            return True, f"{path}: {lines} lines (limit {max_lines})"
        return False, f"{path}: {lines} lines exceeds limit {max_lines}"
    _check.__name__ = f"max_file_lines({path}, {max_lines})"
    return _check


def check_no_circular_imports(package_dir: str, import_name: str):
    """Check that importing the package doesn't cause circular import errors."""
    def _check(workspace: Path) -> tuple[bool, str]:
        pkg = workspace / package_dir
        if not pkg.exists():
            return False, f"Package directory missing: {package_dir}"
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, r'{workspace}'); import {import_name}"],
            capture_output=True, text=True, timeout=10, cwd=str(workspace)
        )
        if result.returncode == 0:
            return True, f"Import {import_name} succeeded (no circular imports)"
        if "circular" in result.stderr.lower() or "ImportError" in result.stderr:
            return False, f"Circular import in {import_name}: {result.stderr[:150]}"
        return False, f"Import failed: {result.stderr[:150]}"
    _check.__name__ = f"no_circular_imports({package_dir})"
    return _check


def check_file_hash_unchanged(path: str, expected_hash: str):
    """Check that a file's content hash matches (file not modified)."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        content = f.read_text(errors="replace")
        actual_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if actual_hash == expected_hash:
            return True, f"{path} unchanged (hash matches)"
        return False, f"{path} was modified (hash mismatch)"
    _check.__name__ = f"file_hash_unchanged({path})"
    return _check


def check_has_lock(path: str, lock_type: str = "asyncio.Lock"):
    """Check that a file uses the specified lock type."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        content = f.read_text(errors="replace")
        if lock_type in content:
            return True, f"Found {lock_type} in {path}"
        return False, f"Missing {lock_type} in {path}"
    _check.__name__ = f"has_lock({path}, {lock_type})"
    return _check


def check_shutdown_guard(path: str):
    """Check that the task queue prevents adding tasks after shutdown."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        content = f.read_text(errors="replace")
        # Look for shutdown check in add_task
        if ("shutdown" in content and
                ("RuntimeError" in content or "raise" in content) and
                "add_task" in content):
            return True, f"Shutdown guard found in {path}"
        return False, f"No shutdown guard found in add_task in {path}"
    _check.__name__ = f"shutdown_guard({path})"
    return _check


def check_stress_test_exists(test_path: str, min_tasks: int = 20):
    """Check that a stress test exists that creates enough concurrent tasks."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / test_path
        if not f.exists():
            return False, f"File missing: {test_path}"
        content = f.read_text(errors="replace")
        if ("stress" in content.lower() or "concurrent" in content.lower()
                or "race" in content.lower()):
            return True, f"Stress/concurrency test found in {test_path}"
        return False, f"No stress/concurrency test found in {test_path}"
    _check.__name__ = f"stress_test_exists({test_path})"
    return _check


def check_partial_success(paths_easy: list[str], paths_hard: list[str]):
    """Check that easy fixes were done but hard ones didn't break things."""
    def _check(workspace: Path) -> tuple[bool, str]:
        easy_ok = 0
        for p in paths_easy:
            f = workspace / p
            if f.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(f)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    easy_ok += 1
        if easy_ok >= len(paths_easy):
            return True, f"All {len(paths_easy)} easy files compile"
        return False, f"Only {easy_ok}/{len(paths_easy)} easy files compile"
    _check.__name__ = f"partial_success({len(paths_easy)} easy)"
    return _check


# ---------------------------------------------------------------------------
# Compute test_markdown hash for tamper detection
# ---------------------------------------------------------------------------

_TEST_MARKDOWN_HASH = hashlib.sha256(TEST_MARKDOWN_CODE.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Task R1: Refactor 400-Line Module
# ---------------------------------------------------------------------------

TASK_R1 = DeepEvalTask(
    id="R1",
    category="R",
    description="Refactor tangled monolith into clean package structure",
    task_prompt=textwrap.dedent("""\
        The file `monolith.py` is a tangled HTTP API client (~300 lines) with
        auth, caching, retry, parsing, and logging all mixed into one class.

        Refactor it into a clean `client/` package with separate modules:
        - `client/__init__.py` — re-exports the public API
        - `client/models.py` — Widget, WidgetPage, APIError, AuthToken dataclasses
        - `client/auth.py` — Auth token management (ensure_auth, refresh)
        - `client/cache.py` — Response caching (TTL cache, get/set/clear)
        - `client/retry.py` — Retry logic with backoff
        - `client/http.py` — Core HTTP client class (WidgetAPIClient)

        IMPORTANT:
        - Keep `monolith.py` as a thin re-export facade so existing imports work:
          `from monolith import WidgetAPIClient, Widget, WidgetPage, APIError`
        - The existing test suite `test_monolith.py` must pass WITHOUT modification.
        - Each new module should be under 100 lines.
        - No circular imports between modules.

        Run the tests after refactoring to verify everything works.
    """),
    setup_files={
        "monolith.py": MONOLITH_CODE,
        "test_monolith.py": TEST_MONOLITH_CODE,
    },
    expert="coder",
    max_rounds=25,
    timeout=240,
    checks=[
        # Core correctness — tests must still pass
        wcheck("tests_pass", check_pytest_passes("test_monolith.py"), 5.0, "correctness"),
        wcheck("monolith_still_importable",
               check_file_contains("monolith.py", "WidgetAPIClient"), 2.0, "correctness"),
        # Package structure exists
        wcheck("init_exists", check_file_exists("client/__init__.py"), 1.5, "correctness"),
        wcheck("models_exists", check_file_exists("client/models.py"), 1.5, "correctness"),
        wcheck("auth_exists", check_file_exists("client/auth.py"), 1.5, "correctness"),
        wcheck("cache_exists", check_file_exists("client/cache.py"), 1.5, "correctness"),
        wcheck("http_exists", check_file_exists("client/http.py"), 1.5, "correctness"),
        # Each module compiles
        wcheck("models_compiles", check_python_compiles("client/models.py"), 1.0, "correctness"),
        wcheck("auth_compiles", check_python_compiles("client/auth.py"), 1.0, "correctness"),
        wcheck("cache_compiles", check_python_compiles("client/cache.py"), 1.0, "correctness"),
        wcheck("http_compiles", check_python_compiles("client/http.py"), 1.0, "correctness"),
        # No circular imports
        wcheck("no_circular", check_no_circular_imports("client", "client"), 2.0, "correctness"),
        # Quality — modules are appropriately sized
        wcheck("models_small", check_max_file_lines("client/models.py", 100), 1.0, "quality"),
        wcheck("auth_small", check_max_file_lines("client/auth.py", 100), 1.0, "quality"),
        wcheck("cache_small", check_max_file_lines("client/cache.py", 100), 1.0, "quality"),
        wcheck("no_god_fns_http", check_no_god_functions("client/http.py", 60), 1.5, "quality"),
        # Completeness — facade works
        wcheck("monolith_is_facade", check_max_file_lines("monolith.py", 20), 1.0, "completeness"),
        wcheck("test_unchanged", check_file_hash_unchanged(
            "test_monolith.py", _TEST_MARKDOWN_HASH  # Will be recomputed at module load
        ), 1.0, "completeness"),
    ],
)

# Fix the hash — compute from actual test_monolith content
_TEST_MONOLITH_HASH = hashlib.sha256(TEST_MONOLITH_CODE.encode()).hexdigest()[:16]
# Patch the check that used the wrong hash
TASK_R1.checks[-1] = wcheck("test_unchanged",
    check_file_hash_unchanged("test_monolith.py", _TEST_MONOLITH_HASH),
    1.0, "completeness")


# ---------------------------------------------------------------------------
# Task R2: Debug Race Condition
# ---------------------------------------------------------------------------

TASK_R2 = DeepEvalTask(
    id="R2",
    category="R",
    description="Find and fix race conditions in async task queue",
    task_prompt=textwrap.dedent("""\
        The file `task_queue.py` implements an async task queue with a worker pool.
        It has reliability issues — sometimes results go missing and the
        completed count is wrong when running multiple workers concurrently.

        Your task:
        1. Read and understand the code
        2. Identify the concurrency bugs (there are at least 3)
        3. Fix them properly using asyncio synchronization primitives
           (NOT threading.Lock — this is async code, use asyncio.Lock)
        4. Also: tasks can be added after shutdown() — add a guard
        5. Write `test_task_queue.py` with:
           - A stress test that runs 50+ tasks across multiple workers
             to verify the race conditions are fixed
           - Tests for shutdown behavior (can't add after shutdown)
           - Tests for basic functionality (add, complete, results)
        6. Run the tests

        Hints:
        - Look at how `completed_count` and `results` are accessed
        - Multiple workers run concurrently in the event loop
        - Even though Python has the GIL, += on shared state in async code
          can interleave at await points
    """),
    setup_files={"task_queue.py": TASK_QUEUE_CODE},
    expert="coder",
    max_rounds=25,
    timeout=240,
    checks=[
        # Bug fixes (correctness)
        wcheck("has_asyncio_lock", check_has_lock("task_queue.py", "asyncio.Lock"), 3.0, "correctness"),
        wcheck("no_threading_lock",
               check_file_not_contains("task_queue.py", "threading.Lock"), 1.5, "correctness"),
        wcheck("shutdown_guard", check_shutdown_guard("task_queue.py"), 2.5, "correctness"),
        wcheck("still_compiles", check_python_compiles("task_queue.py"), 2.0, "correctness"),
        # Tests (quality)
        wcheck("test_exists", check_file_exists("test_task_queue.py"), 1.0, "quality"),
        wcheck("test_compiles", check_python_compiles("test_task_queue.py"), 1.0, "quality"),
        wcheck("tests_pass", check_pytest_passes("test_task_queue.py"), 3.0, "quality"),
        wcheck("has_stress_test", check_stress_test_exists("test_task_queue.py"), 2.0, "quality"),
        # Edge cases
        wcheck("tests_shutdown",
               check_file_contains("test_task_queue.py", "shutdown"), 1.0, "edge_cases"),
        wcheck("tests_concurrent",
               check_file_contains("test_task_queue.py", "worker"), 1.0, "edge_cases"),
        wcheck("lock_used_for_counter",
               check_file_contains("task_queue.py", "async with"), 1.5, "edge_cases"),
        # Completeness
        wcheck("taskstatus_intact",
               check_file_contains("task_queue.py", "TaskStatus"), 0.5, "completeness"),
        wcheck("priority_intact",
               check_file_contains("task_queue.py", "priority"), 0.5, "completeness"),
    ],
)


# ---------------------------------------------------------------------------
# Task R3: Implement Against Failing Test Suite
# ---------------------------------------------------------------------------

TASK_R3 = DeepEvalTask(
    id="R3",
    category="R",
    description="Implement markdown converter to pass 15 predefined tests",
    task_prompt=textwrap.dedent("""\
        The file `test_markdown.py` contains 15 tests for a Markdown-to-HTML
        converter. No implementation exists yet.

        Your task:
        1. Read ALL the tests carefully to understand the expected behavior
        2. Create `markdown_converter.py` with a `to_html(text: str) -> str` function
        3. Make ALL 15 tests pass
        4. Do NOT modify `test_markdown.py`

        The converter should handle:
        - Headings: #, ##, ### → h1, h2, h3
        - Bold: **text** → <strong>text</strong>
        - Italic: *text* → <em>text</em>
        - Inline code: `code` → <code>code</code>
        - Code blocks: ```\\ncode\\n``` → <pre><code>code</code></pre>
        - Links: [text](url) → <a href="url">text</a>
        - Unordered lists: - item → <ul><li>item</li></ul>
        - Ordered lists: 1. item → <ol><li>item</li></ol>
        - Paragraphs: text\\n\\ntext → <p>text</p>\\n<p>text</p>
        - Escaped asterisks: \\*text\\* → *text*
        - Nested: **bold *italic*** → <strong>bold <em>italic</em></strong>
        - Empty input → ""
        - Mixed inline formatting within paragraphs

        Run the tests to verify.
    """),
    setup_files={"test_markdown.py": TEST_MARKDOWN_CODE},
    expert="coder",
    max_rounds=25,
    timeout=300,
    checks=[
        # THE core check — all 15 tests pass (correctness, weight 5.0, * 3x = 15.0)
        wcheck("all_tests_pass", check_pytest_passes("test_markdown.py"), 5.0, "correctness"),
        wcheck("converter_exists", check_file_exists("markdown_converter.py"), 1.0, "correctness"),
        wcheck("converter_compiles", check_python_compiles("markdown_converter.py"), 2.0, "correctness"),
        wcheck("has_to_html", check_function_exists("markdown_converter.py", "to_html"), 1.5, "correctness"),
        # Test not modified
        wcheck("test_not_modified", check_file_hash_unchanged(
            "test_markdown.py", _TEST_MARKDOWN_HASH), 2.0, "correctness"),
        # Quality
        wcheck("no_god_functions",
               check_no_god_functions("markdown_converter.py", 80), 1.0, "quality"),
        wcheck("has_docstrings",
               check_has_docstrings("markdown_converter.py", 0.3), 0.5, "quality"),
        # Edge cases
        wcheck("handles_empty", check_file_contains("markdown_converter.py", '""'), 0.5, "edge_cases"),
        # Completeness — no external deps
        wcheck("no_external_deps",
               check_file_not_contains("markdown_converter.py", "pip install"), 0.5, "completeness"),
    ],
)


# ---------------------------------------------------------------------------
# Task J1: Code Quality Eval (URL Shortener)
# ---------------------------------------------------------------------------

TASK_J1 = DeepEvalTask(
    id="J1",
    category="J",
    description="Build URL shortener with quality heuristics",
    task_prompt=textwrap.dedent("""\
        Build a URL shortener library in `shortener.py` with:

        1. A `URLShortener` class that:
           - `shorten(url: str) -> str` — generates a short code for a URL
           - `resolve(code: str) -> str | None` — returns original URL or None
           - `stats(code: str) -> dict` — returns {url, created_at, access_count}
           - Uses in-memory storage (dict)
           - Validates URLs (must start with http:// or https://)
           - Handles collisions (if short code already exists, generate new one)
           - Short codes should be 6-8 alphanumeric characters

        2. Create `test_shortener.py` with tests for:
           - Basic shorten → resolve round-trip
           - Invalid URL raises ValueError
           - Stats tracking (access_count increments on resolve)
           - Collision handling
           - Empty/None inputs
           - Multiple URLs produce different codes

        Run the tests.
    """),
    setup_files={},
    expert="coder",
    max_rounds=20,
    timeout=180,
    checks=[
        # Correctness
        wcheck("file_exists", check_file_exists("shortener.py"), 1.0, "correctness"),
        wcheck("compiles", check_python_compiles("shortener.py"), 1.5, "correctness"),
        wcheck("has_class", check_class_exists("shortener.py", "URLShortener"), 1.5, "correctness"),
        wcheck("has_shorten", check_function_exists("shortener.py", "shorten"), 1.0, "correctness"),
        wcheck("has_resolve", check_function_exists("shortener.py", "resolve"), 1.0, "correctness"),
        wcheck("has_stats", check_function_exists("shortener.py", "stats"), 1.0, "correctness"),
        wcheck("tests_pass", check_pytest_passes("test_shortener.py"), 3.0, "correctness"),
        # Quality — code quality heuristics
        wcheck("no_god_functions", check_no_god_functions("shortener.py", 40), 1.5, "quality"),
        wcheck("has_docstrings", check_has_docstrings("shortener.py", 0.5), 1.0, "quality"),
        wcheck("reasonable_size", check_max_file_lines("shortener.py", 150), 0.5, "quality"),
        # Edge cases
        wcheck("validates_urls",
               check_file_contains("shortener.py", "http"), 1.0, "edge_cases"),
        wcheck("handles_collisions",
               check_file_contains("shortener.py", "collision"), 0.5, "edge_cases"),
        wcheck("tests_invalid_url",
               check_file_contains("test_shortener.py", "ValueError"), 1.0, "edge_cases"),
        # Completeness
        wcheck("test_exists", check_file_exists("test_shortener.py"), 0.5, "completeness"),
        wcheck("test_compiles", check_python_compiles("test_shortener.py"), 0.5, "completeness"),
    ],
)


# ---------------------------------------------------------------------------
# Task J2: Architecture Decision Eval (Plugin System)
# ---------------------------------------------------------------------------

TASK_J2 = DeepEvalTask(
    id="J2",
    category="J",
    description="Design and implement a plugin system with quality focus",
    task_prompt=textwrap.dedent("""\
        Build a plugin system for a CLI tool.

        Requirements:
        1. Create `plugin_manager.py` with a `PluginManager` class:
           - `discover(directory: str)` — scan directory for Python plugin files
           - `load(name: str)` — load a plugin by name
           - `unload(name: str)` — unload a plugin
           - `list_plugins() -> list[dict]` — list discovered plugins with status
           - `execute(plugin_name: str, command: str, *args)` — run a plugin command

        2. Plugin contract (document this in the code):
           - Each plugin is a .py file in the plugins directory
           - Must define `PLUGIN_NAME: str` and `PLUGIN_VERSION: str`
           - Must define a `commands() -> dict[str, Callable]` function
           - Optional: `on_load()` and `on_unload()` lifecycle hooks

        3. Handle edge cases:
           - Invalid plugins (missing required attributes) → skip with warning
           - Duplicate plugin names → keep first loaded
           - Plugin errors → catch and report, don't crash the manager

        4. Create `test_plugin_manager.py` with tests:
           - Discover valid plugins
           - Skip invalid plugins
           - Load/unload lifecycle
           - Command execution
           - Error handling

        5. Create a `plugins/` directory with 2 example plugins:
           - `plugins/hello.py` — a "hello" command that returns a greeting
           - `plugins/math_plugin.py` — "add" and "multiply" commands

        Run the tests.
    """),
    setup_files={},
    expert="coder",
    max_rounds=25,
    timeout=240,
    checks=[
        # Correctness
        wcheck("manager_exists", check_file_exists("plugin_manager.py"), 1.5, "correctness"),
        wcheck("manager_compiles", check_python_compiles("plugin_manager.py"), 1.5, "correctness"),
        wcheck("has_discover", check_function_exists("plugin_manager.py", "discover"), 1.0, "correctness"),
        wcheck("has_load", check_function_exists("plugin_manager.py", "load"), 1.0, "correctness"),
        wcheck("has_execute", check_function_exists("plugin_manager.py", "execute"), 1.0, "correctness"),
        wcheck("tests_pass", check_pytest_passes("test_plugin_manager.py"), 3.0, "correctness"),
        wcheck("hello_plugin", check_file_exists("plugins/hello.py"), 1.0, "correctness"),
        wcheck("math_plugin", check_file_exists("plugins/math_plugin.py"), 1.0, "correctness"),
        # Quality
        wcheck("no_god_functions",
               check_no_god_functions("plugin_manager.py", 50), 1.5, "quality"),
        wcheck("has_docstrings",
               check_has_docstrings("plugin_manager.py", 0.5), 1.0, "quality"),
        wcheck("documents_contract",
               check_file_contains("plugin_manager.py", "PLUGIN_NAME"), 1.0, "quality"),
        # Edge cases
        wcheck("handles_invalid",
               check_file_contains("plugin_manager.py", "invalid"), 0.5, "edge_cases"),
        wcheck("handles_errors",
               check_file_contains("plugin_manager.py", "except"), 0.5, "edge_cases"),
        wcheck("tests_error_handling",
               check_file_contains("test_plugin_manager.py", "invalid"), 1.0, "edge_cases"),
        # Completeness
        wcheck("has_list_plugins",
               check_function_exists("plugin_manager.py", "list_plugins"), 0.5, "completeness"),
        wcheck("has_unload",
               check_function_exists("plugin_manager.py", "unload"), 0.5, "completeness"),
    ],
)


# ---------------------------------------------------------------------------
# Task K1: Partial Failure Recovery
# ---------------------------------------------------------------------------

K1_EASY_BUG = textwrap.dedent("""\
    def calculate_average(numbers: list[float]) -> float:
        \"\"\"Calculate the average of a list of numbers.\"\"\"
        total = sum(numbers)
        # BUG: missing return statement
        total / len(numbers)

    def calculate_median(numbers: list[float]) -> float:
        \"\"\"Calculate the median of a list of numbers.\"\"\"
        sorted_nums = sorted(numbers)
        n = len(sorted_nums)
        if n % 2 == 0:
            return (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        return sorted_nums[n // 2]

    if __name__ == "__main__":
        data = [1, 2, 3, 4, 5]
        avg = calculate_average(data)
        print(f"Average: {avg}")
        print(f"Median: {calculate_median(data)}")
""")

K1_MEDIUM_BUG = textwrap.dedent("""\
    def merge_sorted(list_a: list[int], list_b: list[int]) -> list[int]:
        \"\"\"Merge two sorted lists into one sorted list.\"\"\"
        result = []
        i, j = 0, 0
        while i < len(list_a) and j < len(list_b):
            if list_a[i] <= list_b[j]:
                result.append(list_a[i])
                i += 1
            else:
                result.append(list_b[j])
                j += 1
        # BUG: only appends remaining from list_a, forgets the other list
        result.extend(list_a[i:])
        # Missing: should also extend with remaining elements of second list
        return result

    def binary_search(arr: list[int], target: int) -> int:
        \"\"\"Return index of target in sorted arr, or -1 if not found.\"\"\"
        lo, hi = 0, len(arr) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    if __name__ == "__main__":
        a = [1, 3, 5, 7]
        b = [2, 4, 6, 8]
        print(merge_sorted(a, b))  # Should be [1,2,3,4,5,6,7,8]
""")

# The "hard" bug — obfuscated code that's nearly impossible to fix correctly
K1_HARD_BUG = textwrap.dedent("""\
    # Legacy code — DO NOT TOUCH unless you fully understand the algorithm
    # This implements a custom compression scheme (LZ-variant)
    import struct

    def _build_table(data, max_bits=12):
        t = {}
        for i in range(256):
            t[bytes([i])] = i
        nc = 256
        mb = (1 << max_bits) - 1
        w = bytes([data[0]]) if data else b""
        out = []
        for c in data[1:]:
            cb = bytes([c])
            wc = w + cb
            if wc in t:
                w = wc
            else:
                out.append(t[w])
                if nc <= mb:
                    t[wc] = nc
                    nc += 1
                w = cb
        if w:
            out.append(t[w])
        return out

    def compress(data: bytes) -> bytes:
        \"\"\"Compress bytes using LZ-variant algorithm.\"\"\"
        if not data:
            return b""
        codes = _build_table(data)
        # BUG: struct.pack format is wrong for codes > 255
        # Should use '>H' (unsigned short) but uses '>B' (unsigned byte)
        # This silently truncates codes, corrupting the output
        result = b""
        for code in codes:
            result += struct.pack('>B', code & 0xFF)
        return result

    def decompress(data: bytes) -> bytes:
        \"\"\"Decompress bytes. WARNING: broken if compress() is broken.\"\"\"
        if not data:
            return b""
        codes = [struct.unpack('>B', data[i:i+1])[0] for i in range(len(data))]
        t = {i: bytes([i]) for i in range(256)}
        nc = 256
        w = t[codes[0]]
        out = [w]
        for code in codes[1:]:
            if code in t:
                entry = t[code]
            elif code == nc:
                entry = w + w[:1]
            else:
                raise ValueError(f"Bad code: {code}")
            out.append(entry)
            t[nc] = w + entry[:1]
            nc += 1
            w = entry
        return b"".join(out)
""")

TASK_K1 = DeepEvalTask(
    id="K1",
    category="K",
    description="Partial failure — fix easy and medium bugs, handle unfixable one",
    task_prompt=textwrap.dedent("""\
        This project has 3 files with bugs:

        1. `stats.py` — has a simple bug. Fix it and verify with a test.
        2. `sorting.py` — has a logic bug in merge_sorted. Fix it and verify.
        3. `compression.py` — has a bug in a compression algorithm.
           This one is complex legacy code. Fix it if you can, but if you
           can't fully fix it, document what's wrong and move on.

        For each file you fix, write tests in `test_fixes.py`.
        Run the tests to verify your fixes work.

        Focus on getting the easy and medium fixes right. Don't let the
        hard one block progress on the others.
    """),
    setup_files={
        "stats.py": K1_EASY_BUG,
        "sorting.py": K1_MEDIUM_BUG,
        "compression.py": K1_HARD_BUG,
    },
    expert="coder",
    max_rounds=20,
    timeout=180,
    checks=[
        # Easy fix (correctness)
        wcheck("stats_has_return",
               check_file_contains("stats.py", "return"), 2.0, "correctness"),
        wcheck("stats_compiles", check_python_compiles("stats.py"), 1.0, "correctness"),
        # Medium fix (correctness)
        wcheck("sorting_has_extend_b",
               check_file_contains("sorting.py", "list_b[j:]"), 2.5, "correctness"),
        wcheck("sorting_compiles", check_python_compiles("sorting.py"), 1.0, "correctness"),
        # Tests (quality)
        wcheck("test_exists", check_file_exists("test_fixes.py"), 1.0, "quality"),
        wcheck("test_compiles", check_python_compiles("test_fixes.py"), 1.0, "quality"),
        wcheck("tests_pass", check_pytest_passes("test_fixes.py"), 3.0, "quality"),
        wcheck("tests_stats", check_file_contains("test_fixes.py", "average"), 1.0, "quality"),
        wcheck("tests_sorting", check_file_contains("test_fixes.py", "merge"), 1.0, "quality"),
        # Edge cases — didn't break the hard file
        wcheck("compression_still_compiles",
               check_python_compiles("compression.py"), 1.5, "edge_cases"),
        wcheck("median_still_works",
               check_function_exists("stats.py", "calculate_median"), 0.5, "edge_cases"),
        wcheck("binary_search_intact",
               check_function_exists("sorting.py", "binary_search"), 0.5, "edge_cases"),
        # Completeness — attempted or documented the hard bug
        wcheck("addressed_compression",
               check_file_contains("compression.py", "struct.pack"), 0.5, "completeness"),
    ],
)


# ---------------------------------------------------------------------------
# Task K2: Bad Orchestrator Decomposition Recovery
# ---------------------------------------------------------------------------

K2_QUERY_BUILDER = textwrap.dedent("""\
    class QueryBuilder:
        \"\"\"Simple SQL query builder for the user service.\"\"\"

        def __init__(self, table: str):
            self.table = table
            self._conditions: list[str] = []
            self._order_by: str | None = None

        def where(self, condition: str) -> "QueryBuilder":
            self._conditions.append(condition)
            return self

        def order_by(self, column: str, direction: str = "ASC") -> "QueryBuilder":
            self._order_by = f"{column} {direction}"
            return self

        def build(self) -> str:
            sql = f"SELECT * FROM {self.table}"
            if self._conditions:
                sql += " WHERE " + " AND ".join(self._conditions)
            if self._order_by:
                sql += f" ORDER BY {self._order_by}"
            return sql

        # NOTE: No limit() or offset() method exists!
        # This is the hidden dependency that pagination needs.
""")

K2_USER_SERVICE = textwrap.dedent("""\
    from query_builder import QueryBuilder

    class UserService:
        \"\"\"Service for managing users.\"\"\"

        def __init__(self, db):
            self.db = db

        def get_users(self, active_only: bool = False) -> list[dict]:
            qb = QueryBuilder("users")
            if active_only:
                qb.where("active = 1")
            qb.order_by("created_at", "DESC")
            query = qb.build()
            return self.db.execute(query)

        def get_user(self, user_id: int) -> dict | None:
            qb = QueryBuilder("users").where(f"id = {user_id}")
            results = self.db.execute(qb.build())
            return results[0] if results else None

        def create_user(self, name: str, email: str) -> dict:
            self.db.execute(
                f"INSERT INTO users (name, email, active) VALUES ('{name}', '{email}', 1)"
            )
            return {"name": name, "email": email, "active": True}
""")

K2_MOCK_DB = textwrap.dedent("""\
    class MockDB:
        \"\"\"In-memory mock database for testing.\"\"\"

        def __init__(self):
            self._data: dict[str, list[dict]] = {
                "users": [
                    {"id": i, "name": f"User{i}", "email": f"user{i}@test.com",
                     "active": i % 3 != 0, "created_at": f"2024-01-{i:02d}"}
                    for i in range(1, 51)  # 50 users
                ]
            }

        def execute(self, query: str) -> list[dict]:
            \"\"\"Simulate SQL execution against in-memory data.\"\"\"
            # Parse table name
            if "FROM" in query:
                parts = query.split("FROM")[1].strip().split()
                table = parts[0]
            elif "INTO" in query:
                return []  # INSERT
            else:
                return []

            data = list(self._data.get(table, []))

            # Apply WHERE conditions (simple simulation)
            if "WHERE" in query:
                where_part = query.split("WHERE")[1].split("ORDER")[0].strip()
                conditions = [c.strip() for c in where_part.split("AND")]
                for cond in conditions:
                    if "active = 1" in cond:
                        data = [r for r in data if r.get("active")]
                    elif "id = " in cond:
                        uid = int(cond.split("=")[1].strip())
                        data = [r for r in data if r["id"] == uid]

            # Apply ORDER BY
            if "ORDER BY" in query:
                order_part = query.split("ORDER BY")[1].strip()
                col = order_part.split()[0]
                desc = "DESC" in order_part.upper()
                data.sort(key=lambda r: r.get(col, ""), reverse=desc)

            # Apply LIMIT/OFFSET if present
            if "LIMIT" in query:
                limit_part = query.split("LIMIT")[1].strip()
                parts = limit_part.split("OFFSET")
                limit = int(parts[0].strip())
                offset = int(parts[1].strip()) if len(parts) > 1 else 0
                data = data[offset:offset + limit]

            return data
""")

TASK_K2 = DeepEvalTask(
    id="K2",
    category="K",
    description="Hidden dependency — QueryBuilder lacks pagination support",
    task_prompt=textwrap.dedent("""\
        Add pagination to the user list endpoint.

        The project has:
        - `query_builder.py` — a SQL query builder
        - `user_service.py` — a user service that uses the query builder
        - `mock_db.py` — an in-memory mock database (supports LIMIT/OFFSET in SQL)

        Your task:
        1. Add a `get_users_paginated(page: int, per_page: int = 10)` method
           to UserService that returns paginated results
        2. It should return: {"users": [...], "page": page, "per_page": per_page,
           "total": total_count}
        3. Write `test_pagination.py` with tests:
           - Page 1 returns first per_page users
           - Page 2 returns next batch
           - Total count is correct
           - Empty page returns empty list
        4. Run the tests

        NOTE: You may need to extend the QueryBuilder to support LIMIT and OFFSET.
        The mock database already supports LIMIT/OFFSET in SQL queries.
    """),
    setup_files={
        "query_builder.py": K2_QUERY_BUILDER,
        "user_service.py": K2_USER_SERVICE,
        "mock_db.py": K2_MOCK_DB,
    },
    expert="coder",
    max_rounds=20,
    timeout=180,
    checks=[
        # Core: pagination works (correctness)
        wcheck("has_paginated_method",
               check_function_exists("user_service.py", "get_users_paginated"), 2.5, "correctness"),
        wcheck("qb_has_limit",
               check_file_contains("query_builder.py", "limit"), 2.0, "correctness"),
        wcheck("qb_has_offset",
               check_file_contains("query_builder.py", "offset"), 2.0, "correctness"),
        wcheck("tests_pass", check_pytest_passes("test_pagination.py"), 3.0, "correctness"),
        wcheck("user_service_compiles",
               check_python_compiles("user_service.py"), 1.0, "correctness"),
        wcheck("qb_compiles",
               check_python_compiles("query_builder.py"), 1.0, "correctness"),
        # Quality
        wcheck("test_exists", check_file_exists("test_pagination.py"), 1.0, "quality"),
        wcheck("test_compiles", check_python_compiles("test_pagination.py"), 1.0, "quality"),
        wcheck("qb_returns_self",
               check_file_contains("query_builder.py", "return self"), 0.5, "quality"),
        # Edge cases
        wcheck("tests_empty_page",
               check_file_contains("test_pagination.py", "empty"), 1.0, "edge_cases"),
        wcheck("tests_total",
               check_file_contains("test_pagination.py", "total"), 1.0, "edge_cases"),
        # Completeness — didn't break existing functionality
        wcheck("existing_get_users_intact",
               check_function_exists("user_service.py", "get_users"), 0.5, "completeness"),
        wcheck("existing_get_user_intact",
               check_function_exists("user_service.py", "get_user"), 0.5, "completeness"),
    ],
)


# ---------------------------------------------------------------------------
# Task K3: Graceful Degradation Under Tight Budget
# ---------------------------------------------------------------------------

# Reuses H1's setup files from eval_deep.py
from eval_deep import H1_SETUP_FILES

TASK_K3 = DeepEvalTask(
    id="K3",
    category="K",
    description="Tight budget — accomplish as much as possible in 8 rounds",
    task_prompt=textwrap.dedent("""\
        This project has a mini web framework in `framework/`. Add authentication:

        1. Create `framework/auth.py` with a User dataclass and authenticate function
        2. Add a POST /login endpoint to main_app.py
        3. Write basic tests in `test_auth.py`

        IMPORTANT: You have a very limited budget. Focus on getting the most
        important parts working first. If you can't finish everything,
        make sure what you DID finish compiles and works.

        Priorities (in order):
        1. framework/auth.py with User + authenticate
        2. /login endpoint in main_app.py
        3. Tests (even minimal ones)
    """),
    setup_files=H1_SETUP_FILES,
    expert="coder",
    max_rounds=8,   # Very tight!
    timeout=120,
    checks=[
        # Did SOMETHING useful (correctness)
        wcheck("auth_module_exists", check_file_exists("framework/auth.py"), 2.0, "correctness"),
        wcheck("auth_compiles", check_python_compiles("framework/auth.py"), 2.0, "correctness"),
        wcheck("has_user", check_class_exists("framework/auth.py", "User"), 1.5, "correctness"),
        wcheck("has_authenticate",
               check_function_exists("framework/auth.py", "authenticate"), 1.5, "correctness"),
        # Prioritized correctly (quality)
        wcheck("login_endpoint",
               check_file_contains("main_app.py", "/login"), 1.5, "quality"),
        wcheck("main_still_compiles",
               check_python_compiles("main_app.py"), 1.0, "quality"),
        # Nice to have but not critical under tight budget
        wcheck("test_exists", check_file_exists("test_auth.py"), 0.5, "completeness"),
        wcheck("tests_pass", check_pytest_passes("test_auth.py"), 1.0, "completeness"),
        # Didn't break existing code (edge_cases)
        wcheck("framework_intact",
               check_python_compiles("framework/app.py"), 1.5, "edge_cases"),
        wcheck("index_still_works",
               check_file_contains("main_app.py", "@app.route"), 0.5, "edge_cases"),
    ],
)


# ---------------------------------------------------------------------------
# Task K4: Conflicting Instructions
# ---------------------------------------------------------------------------

K4_REQUIREMENTS = textwrap.dedent("""\
    # Requirements for config_parser.py

    Implement a `ConfigParser` class that reads key=value config files.

    ## Behavior:
    - `load(filepath)` — reads the config file
    - `get(key)` — returns the value for a key
    - `get(key, default)` — returns default if key not found
    - On missing key WITHOUT default, return None  <<< CONTRADICTS TESTS

    ## Format:
    - Lines with `key = value` (spaces around = are optional)
    - Lines starting with # are comments
    - Empty lines are ignored
    - Values are always strings
""")

K4_TESTS = textwrap.dedent("""\
    import pytest
    from config_parser import ConfigParser

    def test_load_and_get():
        parser = ConfigParser()
        parser.load("test_config.ini")
        assert parser.get("name") == "TestApp"
        assert parser.get("version") == "1.0"

    def test_get_with_default():
        parser = ConfigParser()
        parser.load("test_config.ini")
        assert parser.get("missing", "fallback") == "fallback"

    def test_get_missing_raises():
        \"\"\"Missing key WITHOUT default should raise KeyError, not return None.\"\"\"
        parser = ConfigParser()
        parser.load("test_config.ini")
        with pytest.raises(KeyError):
            parser.get("nonexistent")

    def test_comments_ignored():
        parser = ConfigParser()
        parser.load("test_config.ini")
        # The comment key should not be present
        with pytest.raises(KeyError):
            parser.get("# comment")

    def test_empty_lines_ignored():
        parser = ConfigParser()
        parser.load("test_config.ini")
        assert parser.get("debug") == "true"

    def test_spaces_around_equals():
        parser = ConfigParser()
        parser.load("test_config.ini")
        assert parser.get("host") == "localhost"
""")

K4_CONFIG_FILE = textwrap.dedent("""\
    # Application config
    name = TestApp
    version=1.0

    # Server settings
    host = localhost
    port = 8080

    debug = true
""")

TASK_K4 = DeepEvalTask(
    id="K4",
    category="K",
    description="Conflicting requirements — tests say KeyError, docs say None",
    task_prompt=textwrap.dedent("""\
        Implement the config parser described in `requirements.txt`.
        The test file `test_config.py` defines the expected behavior.
        A sample config file `test_config.ini` is provided.

        Create `config_parser.py` with a `ConfigParser` class.
        Make all tests pass. Do NOT modify the test file.

        Read both the requirements AND the tests carefully — if they
        disagree, the tests are the ground truth.
    """),
    setup_files={
        "requirements.txt": K4_REQUIREMENTS,
        "test_config.py": K4_TESTS,
        "test_config.ini": K4_CONFIG_FILE,
    },
    expert="coder",
    max_rounds=15,
    timeout=120,
    checks=[
        # Core — tests pass (the ground truth)
        wcheck("all_tests_pass", check_pytest_passes("test_config.py"), 5.0, "correctness"),
        wcheck("parser_exists", check_file_exists("config_parser.py"), 1.0, "correctness"),
        wcheck("parser_compiles", check_python_compiles("config_parser.py"), 1.5, "correctness"),
        wcheck("has_class", check_class_exists("config_parser.py", "ConfigParser"), 1.0, "correctness"),
        wcheck("has_load", check_function_exists("config_parser.py", "load"), 1.0, "correctness"),
        wcheck("has_get", check_function_exists("config_parser.py", "get"), 1.0, "correctness"),
        # Followed tests over docs (quality — did it raise KeyError?)
        wcheck("raises_keyerror",
               check_file_contains("config_parser.py", "KeyError"), 2.0, "quality"),
        wcheck("test_not_modified", check_file_hash_unchanged(
            "test_config.py",
            hashlib.sha256(K4_TESTS.encode()).hexdigest()[:16]
        ), 1.5, "quality"),
        # Edge cases
        wcheck("handles_comments",
               check_file_contains("config_parser.py", "#"), 0.5, "edge_cases"),
        wcheck("handles_spaces",
               check_file_contains("config_parser.py", "strip"), 0.5, "edge_cases"),
        # Noted the contradiction (bonus quality)
        wcheck("noticed_contradiction",
               check_file_contains("config_parser.py", "contradict"), 0.5, "quality"),
    ],
)


# ---------------------------------------------------------------------------
# All V2 tasks
# ---------------------------------------------------------------------------

V2_EVAL_TASKS: list[DeepEvalTask] = [
    TASK_R1, TASK_R2, TASK_R3,
    TASK_J1, TASK_J2,
    TASK_K1, TASK_K2, TASK_K3, TASK_K4,
]


# ---------------------------------------------------------------------------
# Pytest unit tests
# ---------------------------------------------------------------------------

class TestV2DataStructures:
    """Verify all V2 task definitions are well-formed."""

    def test_all_tasks_have_checks(self):
        for task in V2_EVAL_TASKS:
            assert len(task.checks) > 0, f"Task {task.id} has no checks"

    def test_all_tasks_have_unique_ids(self):
        ids = [t.id for t in V2_EVAL_TASKS]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs: {ids}"

    def test_all_tasks_have_valid_category(self):
        valid = ("R", "J", "K")
        for task in V2_EVAL_TASKS:
            assert task.category in valid, f"Task {task.id}: bad category '{task.category}'"

    def test_all_checks_have_valid_category(self):
        for task in V2_EVAL_TASKS:
            for wc in task.checks:
                assert wc.category in CATEGORY_MULTIPLIERS, \
                    f"Task {task.id}, check {wc.name}: bad category '{wc.category}'"

    def test_task_count(self):
        assert len(V2_EVAL_TASKS) == 9, f"Expected 9 tasks, got {len(V2_EVAL_TASKS)}"


class TestV2CheckFunctions:
    """Verify the new V2 check functions."""

    def test_no_god_functions_pass(self, tmp_path):
        (tmp_path / "small.py").write_text("def f():\n    return 1\n")
        ok, msg = check_no_god_functions("small.py", 10)(tmp_path)
        assert ok is True

    def test_no_god_functions_fail(self, tmp_path):
        code = "def f():\n" + "    x = 1\n" * 60
        (tmp_path / "big.py").write_text(code)
        ok, msg = check_no_god_functions("big.py", 50)(tmp_path)
        assert ok is False
        assert "God functions" in msg

    def test_has_docstrings_pass(self, tmp_path):
        code = 'def foo():\n    """Docstring."""\n    pass\ndef bar():\n    """Doc."""\n    pass\n'
        (tmp_path / "doc.py").write_text(code)
        ok, msg = check_has_docstrings("doc.py", 0.5)(tmp_path)
        assert ok is True

    def test_has_docstrings_fail(self, tmp_path):
        code = "def foo():\n    pass\ndef bar():\n    pass\n"
        (tmp_path / "nodoc.py").write_text(code)
        ok, msg = check_has_docstrings("nodoc.py", 0.5)(tmp_path)
        assert ok is False

    def test_max_file_lines_pass(self, tmp_path):
        (tmp_path / "short.py").write_text("x = 1\n" * 5)
        ok, msg = check_max_file_lines("short.py", 10)(tmp_path)
        assert ok is True

    def test_max_file_lines_fail(self, tmp_path):
        (tmp_path / "long.py").write_text("x = 1\n" * 200)
        ok, msg = check_max_file_lines("long.py", 100)(tmp_path)
        assert ok is False

    def test_no_circular_imports_pass(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("from mypkg.a import hello\n")
        (pkg / "a.py").write_text("def hello(): return 'hi'\n")
        ok, msg = check_no_circular_imports("mypkg", "mypkg")(tmp_path)
        assert ok is True

    def test_file_hash_unchanged_pass(self, tmp_path):
        content = "hello world"
        h = hashlib.sha256(content.encode()).hexdigest()[:16]
        (tmp_path / "f.txt").write_text(content)
        ok, msg = check_file_hash_unchanged("f.txt", h)(tmp_path)
        assert ok is True

    def test_file_hash_unchanged_fail(self, tmp_path):
        (tmp_path / "f.txt").write_text("modified content")
        ok, msg = check_file_hash_unchanged("f.txt", "0000000000000000")(tmp_path)
        assert ok is False

    def test_has_lock_pass(self, tmp_path):
        (tmp_path / "q.py").write_text("import asyncio\nlock = asyncio.Lock()\n")
        ok, msg = check_has_lock("q.py", "asyncio.Lock")(tmp_path)
        assert ok is True

    def test_has_lock_fail(self, tmp_path):
        (tmp_path / "q.py").write_text("x = 1\n")
        ok, msg = check_has_lock("q.py", "asyncio.Lock")(tmp_path)
        assert ok is False

    def test_shutdown_guard_pass(self, tmp_path):
        code = "class Q:\n    async def add_task(self):\n        if self._shutdown:\n            raise RuntimeError('shutdown')\n"
        (tmp_path / "q.py").write_text(code)
        ok, msg = check_shutdown_guard("q.py")(tmp_path)
        assert ok is True

    def test_shutdown_guard_fail(self, tmp_path):
        code = "class Q:\n    async def add_task(self):\n        pass\n"
        (tmp_path / "q.py").write_text(code)
        ok, msg = check_shutdown_guard("q.py")(tmp_path)
        assert ok is False


class TestV2SetupFiles:
    """Verify task setup files create correctly and bugs are real."""

    def test_r1_monolith_setup(self, tmp_path):
        _setup_deep_workspace(TASK_R1, tmp_path)
        assert (tmp_path / "monolith.py").exists()
        assert (tmp_path / "test_monolith.py").exists()
        # Monolith should compile
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(tmp_path / "monolith.py")],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0

    def test_r1_tests_pass_before_refactor(self, tmp_path):
        """Tests should pass against the un-refactored monolith."""
        _setup_deep_workspace(TASK_R1, tmp_path)
        ok, msg = check_pytest_passes("test_monolith.py")(tmp_path)
        assert ok is True, f"Monolith tests should pass before refactoring: {msg}"

    def test_r2_task_queue_setup(self, tmp_path):
        _setup_deep_workspace(TASK_R2, tmp_path)
        assert (tmp_path / "task_queue.py").exists()
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(tmp_path / "task_queue.py")],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0

    def test_r2_has_bugs(self, tmp_path):
        """Verify the race condition bugs are present."""
        _setup_deep_workspace(TASK_R2, tmp_path)
        content = (tmp_path / "task_queue.py").read_text()
        assert "asyncio.Lock" not in content, "Bug should be present: no Lock"
        assert "self.completed_count += 1" in content, "Bug: unprotected counter"

    def test_r3_test_markdown_setup(self, tmp_path):
        _setup_deep_workspace(TASK_R3, tmp_path)
        assert (tmp_path / "test_markdown.py").exists()
        # No converter exists yet — tests should fail
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_markdown.py", "-x", "-q"],
            capture_output=True, text=True, timeout=10, cwd=str(tmp_path)
        )
        assert result.returncode != 0, "Tests should fail with no implementation"

    def test_k1_bugs_present(self, tmp_path):
        _setup_deep_workspace(TASK_K1, tmp_path)
        # Easy: missing return
        stats_content = (tmp_path / "stats.py").read_text()
        # Check the calculate_average function doesn't return
        assert "return total / len" not in stats_content or "return" not in stats_content.split("calculate_average")[1].split("calculate_median")[0]
        # Medium: missing extend for list_b
        sorting_content = (tmp_path / "sorting.py").read_text()
        assert "list_b[j:]" not in sorting_content

    def test_k2_missing_pagination(self, tmp_path):
        _setup_deep_workspace(TASK_K2, tmp_path)
        qb_content = (tmp_path / "query_builder.py").read_text()
        assert "limit" not in qb_content.lower() or "LIMIT" not in qb_content
        us_content = (tmp_path / "user_service.py").read_text()
        assert "paginated" not in us_content

    def test_k4_contradiction_present(self, tmp_path):
        """Verify requirements say None but tests expect KeyError."""
        _setup_deep_workspace(TASK_K4, tmp_path)
        reqs = (tmp_path / "requirements.txt").read_text()
        assert "return None" in reqs
        tests = (tmp_path / "test_config.py").read_text()
        assert "KeyError" in tests


class TestV2Report:
    """Verify V2 tasks integrate with the report formatter."""

    def test_format_report_v2(self):
        results = [
            ComparativeResult(
                task_id="R1", category="R", description="Refactor test",
                single=RunMetrics(quality_score=0.7, time_seconds=30, cost_usd=0.05,
                                  checks_passed=10, checks_total=15, check_details=[]),
                swarm=RunMetrics(quality_score=0.85, time_seconds=25, cost_usd=0.12,
                                checks_passed=13, checks_total=15, check_details=[]),
                verdict="swarm_better", quality_delta=0.15, speedup=1.2, cost_ratio=2.4,
            ),
        ]
        report = format_deep_report(results)
        assert "R1" in report
        assert "Swarm Better" in report


class TestV2CorpusIntegrity:
    """Verify corpus files load correctly."""

    def test_monolith_code_loaded(self):
        assert len(MONOLITH_CODE) > 5000, "Monolith code should be substantial"
        assert "class WidgetAPIClient" in MONOLITH_CODE

    def test_test_monolith_code_loaded(self):
        assert "class TestWidgetClient" in TEST_MONOLITH_CODE

    def test_task_queue_code_loaded(self):
        assert len(TASK_QUEUE_CODE) > 3000
        assert "class AsyncTaskQueue" in TASK_QUEUE_CODE

    def test_test_markdown_code_loaded(self):
        assert "def test_h1" in TEST_MARKDOWN_CODE
        assert "def test_empty_input" in TEST_MARKDOWN_CODE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _run_live_v2_eval(task_ids: list[str] | None = None):
    """Run V2 eval tasks with real API calls."""
    tasks = V2_EVAL_TASKS
    if task_ids:
        tasks = [t for t in V2_EVAL_TASKS if t.id in task_ids]
        if not tasks:
            print(f"No tasks found matching: {task_ids}")
            return

    results = []
    for task in tasks:
        print(f"\n{'='*70}")
        print(f"V2 Eval: {task.id} ({task.category}) — {task.description}")
        print(f"{'='*70}")

        if task.use_swarm:
            result = await run_comparative(task)
        else:
            with tempfile.TemporaryDirectory(prefix=f"v2_{task.id}_") as tmp:
                ws = Path(tmp)
                single_metrics = await _run_single_agent(task, ws)
                result = ComparativeResult(
                    task_id=task.id, category=task.category,
                    description=task.description, single=single_metrics,
                )
                _compute_efficiency_scores(result)
        results.append(result)

    report = format_deep_report(results)
    print(f"\n{report}")

    report_path = Path(".grokswarm") / "v2_eval_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    json_path = Path(".grokswarm") / "v2_eval_results.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"Results saved to: {json_path}")

    # Save efficiency scores
    scores_path = _save_eval_scores(results)
    print(f"Scores saved to: {scores_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GrokSwarm V2 Real-World Eval")
    parser.add_argument("--live", action="store_true", help="Run with real API calls")
    parser.add_argument("--task", nargs="*", help="Specific task IDs (e.g., R1 K2)")
    parser.add_argument("--list", action="store_true", help="List available V2 eval tasks")
    parser.add_argument("--claude", action="store_true",
                        help="Route agent execution through Claude Code CLI")
    args = parser.parse_args()

    if args.claude:
        import shutil
        if not shutil.which("claude"):
            print("Error: Claude Code CLI not found in PATH")
            sys.exit(1)
        shared.state.claude_mode = True
        print("Claude Code mode: ON — agents will execute via claude -p")

    if args.list:
        print(f"{'ID':<8} {'Cat':>4} {'Checks':>7} {'Rounds':>7} {'Expert':<10} Description")
        print("-" * 95)
        for t in V2_EVAL_TASKS:
            print(f"{t.id:<8} {t.category:>4} {len(t.checks):>7} {t.max_rounds:>7} "
                  f"{t.expert:<10} {t.description}")
        return

    if args.live:
        task_ids = args.task if args.task else None
        asyncio.run(_run_live_v2_eval(task_ids))
    else:
        print("Running pytest V2 eval suite...")
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


if __name__ == "__main__":
    main()
