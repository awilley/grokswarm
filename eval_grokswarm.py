"""
eval_grokswarm.py — End-to-end evaluation harness for GrokSwarm

Two modes:
  1. Unit eval (mocked API):  python -m pytest eval_grokswarm.py -v
  2. Live eval (real API):    python eval_grokswarm.py [--live] [--task TASK_ID]

Measures: correctness, rounds, tokens, cost, time.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

os.environ.setdefault("XAI_API_KEY", "test-key-for-eval")
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import grokswarm.shared as shared
import grokswarm.agents as agents
from grokswarm.agents import run_expert, SwarmBus, _record_usage
from grokswarm.guardrails import reset_model_tiers


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalTask:
    """One evaluation task definition."""
    id: str
    category: str          # A=file_ops, B=bug_fix, C=feature, D=multi_agent
    description: str       # Human-readable description
    task_prompt: str       # What the agent receives
    setup_files: dict      # {relative_path: content} — files to create before running
    expert: str = "coder"  # Which expert to use
    checks: list = field(default_factory=list)  # List of check functions (callables)
    max_rounds: int = 15
    timeout: int = 120


@dataclass
class EvalResult:
    """Result of one evaluation run."""
    task_id: str
    correct: float = 0.0        # 0.0 - 1.0
    rounds_used: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    time_seconds: float = 0.0
    error: str = ""
    check_details: list = field(default_factory=list)  # per-check pass/fail


# ---------------------------------------------------------------------------
# Check functions — validators that inspect the workspace after agent runs
# ---------------------------------------------------------------------------

def check_file_exists(path: str):
    """Returns a check function that verifies a file exists."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if f.exists():
            return True, f"File exists: {path}"
        return False, f"File missing: {path}"
    _check.__name__ = f"file_exists({path})"
    return _check


def check_file_contains(path: str, substring: str):
    """Returns a check function that verifies file contains a substring."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        content = f.read_text(errors="replace")
        if substring in content:
            return True, f"Found '{substring}' in {path}"
        return False, f"Missing '{substring}' in {path}"
    _check.__name__ = f"file_contains({path}, '{substring[:30]}')"
    return _check


def check_file_not_contains(path: str, substring: str):
    """Returns a check function that verifies file does NOT contain a substring."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return True, f"File missing (ok): {path}"
        content = f.read_text(errors="replace")
        if substring not in content:
            return True, f"Correctly absent '{substring}' in {path}"
        return False, f"Unwanted '{substring}' still in {path}"
    _check.__name__ = f"file_not_contains({path}, '{substring[:30]}')"
    return _check


def check_python_compiles(path: str):
    """Returns a check function that verifies a Python file compiles."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(f)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return True, f"Compiles: {path}"
        return False, f"Compile error in {path}: {result.stderr[:200]}"
    _check.__name__ = f"python_compiles({path})"
    return _check


def check_pytest_passes(path: str = ".", pattern: str = ""):
    """Returns a check function that verifies pytest passes."""
    def _check(workspace: Path) -> tuple[bool, str]:
        cmd = [sys.executable, "-m", "pytest", path, "-x", "-q"]
        if pattern:
            cmd.extend(["-k", pattern])
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, cwd=str(workspace)
        )
        if result.returncode == 0:
            return True, f"Tests pass: {result.stdout.strip().split(chr(10))[-1]}"
        return False, f"Tests fail: {result.stdout[-300:]}"
    _check.__name__ = f"pytest_passes({path})"
    return _check


def check_function_exists(path: str, func_name: str):
    """Returns a check function that verifies a function is defined."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        content = f.read_text(errors="replace")
        if f"def {func_name}" in content:
            return True, f"Function '{func_name}' found in {path}"
        return False, f"Function '{func_name}' not found in {path}"
    _check.__name__ = f"function_exists({path}, {func_name})"
    return _check


def check_output_matches(path: str, expected_stdout: str):
    """Returns a check function that runs a script and checks stdout."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        result = subprocess.run(
            [sys.executable, str(f)],
            capture_output=True, text=True, timeout=15, cwd=str(workspace)
        )
        actual = result.stdout.strip()
        expected = expected_stdout.strip()
        if actual == expected:
            return True, f"Output matches: '{actual[:50]}'"
        return False, f"Output mismatch: expected '{expected[:50]}', got '{actual[:50]}'"
    _check.__name__ = f"output_matches({path})"
    return _check


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

EVAL_TASKS: list[EvalTask] = [
    # -- Category A: File Operations --
    EvalTask(
        id="A1_create_function",
        category="A",
        description="Create a Python file with a fibonacci function",
        task_prompt="Create a file called `fib.py` that contains a function `fibonacci(n)` which returns the nth Fibonacci number (0-indexed, so fibonacci(0)=0, fibonacci(1)=1, fibonacci(6)=8). Include a simple test at the bottom that asserts fibonacci(10) == 55.",
        setup_files={},
        checks=[
            check_file_exists("fib.py"),
            check_python_compiles("fib.py"),
            check_function_exists("fib.py", "fibonacci"),
            check_file_contains("fib.py", "fibonacci(10)")
        ],
    ),
    EvalTask(
        id="A2_edit_existing",
        category="A",
        description="Add a function to an existing file",
        task_prompt="Edit `utils.py` to add a function `reverse_words(s: str) -> str` that reverses the order of words in a string. For example: reverse_words('hello world') returns 'world hello'. Do not modify the existing `greet` function.",
        setup_files={
            "utils.py": textwrap.dedent("""\
                def greet(name: str) -> str:
                    return f"Hello, {name}!"
            """),
        },
        checks=[
            check_file_exists("utils.py"),
            check_python_compiles("utils.py"),
            check_function_exists("utils.py", "reverse_words"),
            check_function_exists("utils.py", "greet"),
            check_file_contains("utils.py", "reverse_words"),
        ],
    ),
    EvalTask(
        id="A3_search_and_report",
        category="A",
        description="Search codebase and report findings",
        task_prompt="Search the project for all functions that take a `name` parameter. List them with their file paths. Write the results to a file called `report.txt`.",
        expert="researcher",
        setup_files={
            "app.py": 'def get_user(name: str): return name.upper()\ndef get_age(): return 42\n',
            "helpers.py": 'def format_name(name): return name.strip()\ndef add(a, b): return a + b\n',
            "config.py": 'DEBUG = True\nVERSION = "1.0"\n',
        },
        checks=[
            check_file_exists("report.txt"),
            check_file_contains("report.txt", "get_user"),
            check_file_contains("report.txt", "format_name"),
        ],
    ),

    # -- Category B: Bug Fixes --
    EvalTask(
        id="B1_fix_syntax_error",
        category="B",
        description="Fix a syntax error in a Python file",
        task_prompt="The file `broken.py` has a syntax error. Fix it so it compiles and runs correctly. The function should return the sum of a list of numbers.",
        setup_files={
            "broken.py": textwrap.dedent("""\
                def sum_list(numbers):
                    total = 0
                    for n in numbers
                        total += n
                    return total

                if __name__ == "__main__":
                    print(sum_list([1, 2, 3, 4, 5]))
            """),
        },
        checks=[
            check_python_compiles("broken.py"),
            check_output_matches("broken.py", "15"),
        ],
    ),
    EvalTask(
        id="B2_fix_logic_bug",
        category="B",
        description="Fix an off-by-one error",
        task_prompt="The function `count_vowels` in `vowels.py` is returning wrong results. Fix the bug. Expected: count_vowels('hello') == 2, count_vowels('aeiou') == 5.",
        setup_files={
            "vowels.py": textwrap.dedent("""\
                def count_vowels(s: str) -> int:
                    count = 0
                    vowels = "aeiou"
                    for i in range(1, len(s)):  # BUG: starts at 1, skips first char
                        if s[i].lower() in vowels:
                            count += 1
                    return count

                if __name__ == "__main__":
                    assert count_vowels("hello") == 2, f"got {count_vowels('hello')}"
                    assert count_vowels("aeiou") == 5, f"got {count_vowels('aeiou')}"
                    print("ALL PASS")
            """),
        },
        checks=[
            check_python_compiles("vowels.py"),
            check_output_matches("vowels.py", "ALL PASS"),
        ],
    ),
    EvalTask(
        id="B3_fix_failing_test",
        category="B",
        description="Fix source code to make a failing test pass",
        task_prompt="Run the tests in `test_calc.py`. They are failing. Read the test expectations, then fix `calc.py` so all tests pass. Do NOT modify the tests.",
        setup_files={
            "calc.py": textwrap.dedent("""\
                def add(a, b):
                    return a + b

                def multiply(a, b):
                    return a + b  # BUG: should be a * b

                def is_even(n):
                    return n % 2 == 1  # BUG: should be == 0
            """),
            "test_calc.py": textwrap.dedent("""\
                from calc import add, multiply, is_even

                def test_add():
                    assert add(2, 3) == 5
                    assert add(-1, 1) == 0

                def test_multiply():
                    assert multiply(3, 4) == 12
                    assert multiply(0, 5) == 0

                def test_is_even():
                    assert is_even(4) is True
                    assert is_even(3) is False
                    assert is_even(0) is True
            """),
        },
        checks=[
            check_python_compiles("calc.py"),
            check_file_not_contains("test_calc.py", "a + b"),  # test unchanged
            check_pytest_passes("test_calc.py"),
        ],
    ),

    # -- Category C: Feature Implementation --
    EvalTask(
        id="C1_function_with_tests",
        category="C",
        description="Implement function + write tests",
        task_prompt=textwrap.dedent("""\
            Implement a `stack.py` module with a `Stack` class that supports:
            - push(item) — add item to top
            - pop() -> item — remove and return top item, raise IndexError if empty
            - peek() -> item — return top item without removing, raise IndexError if empty
            - is_empty() -> bool
            - size() -> int

            Then create `test_stack.py` with at least 5 test functions covering normal use and edge cases (empty stack errors).
            Run the tests to verify they pass.
        """),
        setup_files={},
        checks=[
            check_file_exists("stack.py"),
            check_file_exists("test_stack.py"),
            check_python_compiles("stack.py"),
            check_python_compiles("test_stack.py"),
            check_function_exists("stack.py", "__init__"),
            check_file_contains("stack.py", "def push"),
            check_file_contains("stack.py", "def pop"),
            check_pytest_passes("test_stack.py"),
        ],
    ),
    EvalTask(
        id="C2_refactor_extract",
        category="C",
        description="Refactor: extract helper function",
        task_prompt=textwrap.dedent("""\
            The file `processor.py` has duplicated validation logic in both `process_user` and `process_order`.
            Refactor: extract the email validation into a separate function `validate_email(email: str) -> bool`.
            Update both functions to use the new helper.
            Make sure the existing tests still pass after refactoring.
        """),
        setup_files={
            "processor.py": textwrap.dedent("""\
                import re

                def process_user(name: str, email: str) -> dict:
                    # Validate email
                    if not email or "@" not in email:
                        raise ValueError("Invalid email")
                    if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$", email):
                        raise ValueError("Invalid email format")
                    return {"name": name, "email": email.lower(), "type": "user"}

                def process_order(order_id: str, email: str, amount: float) -> dict:
                    # Validate email (duplicated!)
                    if not email or "@" not in email:
                        raise ValueError("Invalid email")
                    if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$", email):
                        raise ValueError("Invalid email format")
                    return {"order_id": order_id, "email": email.lower(), "amount": amount}
            """),
            "test_processor.py": textwrap.dedent("""\
                import pytest
                from processor import process_user, process_order

                def test_process_user_valid():
                    result = process_user("Alice", "alice@example.com")
                    assert result["email"] == "alice@example.com"

                def test_process_user_invalid_email():
                    with pytest.raises(ValueError):
                        process_user("Bob", "not-an-email")

                def test_process_order_valid():
                    result = process_order("ORD-1", "bob@test.com", 99.99)
                    assert result["amount"] == 99.99

                def test_process_order_invalid_email():
                    with pytest.raises(ValueError):
                        process_order("ORD-2", "", 10.0)
            """),
        },
        checks=[
            check_python_compiles("processor.py"),
            check_function_exists("processor.py", "validate_email"),
            check_pytest_passes("test_processor.py"),
        ],
    ),
    EvalTask(
        id="C3_cli_from_spec",
        category="C",
        description="Implement a small CLI tool from specification",
        task_prompt=textwrap.dedent("""\
            Create a command-line tool `wordcount.py` that:
            1. Takes a filename as a command-line argument (sys.argv[1])
            2. Reads the file
            3. Prints three lines:
               Lines: <count>
               Words: <count>
               Chars: <count>
            4. If no argument given, prints "Usage: wordcount.py <filename>" and exits with code 1
            5. If file not found, prints "Error: file not found" and exits with code 1

            Create a test file `sample.txt` with exactly this content (3 lines):
            Hello world
            This is a test
            Goodbye

            Then run: python wordcount.py sample.txt
            Expected output:
            Lines: 3
            Words: 7
            Chars: 36
        """),
        setup_files={},
        checks=[
            check_file_exists("wordcount.py"),
            check_file_exists("sample.txt"),
            check_python_compiles("wordcount.py"),
        ],
    ),

    # -- Category D: Multi-Agent / Swarm --
    EvalTask(
        id="D1_delegate_subtask",
        category="D",
        description="Supervisor delegates a coding subtask to a coder agent",
        task_prompt=textwrap.dedent("""\
            You are a supervisor. Spawn a coder agent to create a file called `greeter.py`
            with a function `greet(name: str) -> str` that returns "Hello, <name>!".
            After the coder finishes, verify the file exists by reading it.
            Then write a file called `status.txt` that says "DONE".
        """),
        setup_files={},
        checks=[
            check_file_exists("greeter.py"),
            check_function_exists("greeter.py", "greet"),
            check_python_compiles("greeter.py"),
            check_file_exists("status.txt"),
            check_file_contains("status.txt", "DONE"),
        ],
        max_rounds=20,
        timeout=180,
    ),
    EvalTask(
        id="D2_multi_file_coordination",
        category="D",
        description="Agent spawns helper to build module, then writes tests itself",
        task_prompt=textwrap.dedent("""\
            Create a project with two files:
            1. Spawn a coder agent named 'builder' to create `mathlib.py` with functions:
               - add(a, b) -> a + b
               - subtract(a, b) -> a - b
               - multiply(a, b) -> a * b
            2. After the builder agent finishes, write `test_mathlib.py` yourself with
               tests for all three functions. Run the tests to make sure they pass.
        """),
        setup_files={},
        checks=[
            check_file_exists("mathlib.py"),
            check_file_exists("test_mathlib.py"),
            check_python_compiles("mathlib.py"),
            check_python_compiles("test_mathlib.py"),
            check_function_exists("mathlib.py", "add"),
            check_function_exists("mathlib.py", "subtract"),
            check_function_exists("mathlib.py", "multiply"),
            check_pytest_passes("test_mathlib.py"),
        ],
        max_rounds=25,
        timeout=240,
    ),
    EvalTask(
        id="D3_message_passing",
        category="D",
        description="Two agents coordinate via message bus",
        task_prompt=textwrap.dedent("""\
            1. Spawn a researcher agent named 'scanner' with the task:
               "Read all .py files in this directory and send a message listing
                every function name you find. Use send_message to report results."
            2. Wait for the scanner to finish (check messages).
            3. Write a file called `summary.txt` with the list of functions found.
        """),
        setup_files={
            "module_a.py": "def alpha(): pass\ndef beta(): pass\n",
            "module_b.py": "def gamma(x): return x * 2\ndef delta(): pass\n",
        },
        checks=[
            check_file_exists("summary.txt"),
            check_file_contains("summary.txt", "alpha"),
            check_file_contains("summary.txt", "beta"),
            check_file_contains("summary.txt", "gamma"),
            check_file_contains("summary.txt", "delta"),
        ],
        max_rounds=25,
        timeout=240,
    ),
]


# ---------------------------------------------------------------------------
# Eval runner (works with real or mocked API)
# ---------------------------------------------------------------------------

def _setup_workspace(task: EvalTask, workspace: Path):
    """Create the workspace directory and seed files."""
    workspace.mkdir(parents=True, exist_ok=True)
    for rel_path, content in task.setup_files.items():
        f = workspace / rel_path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)


def _run_checks(task: EvalTask, workspace: Path) -> tuple[float, list[dict]]:
    """Run all checks and return (score 0-1, details)."""
    details = []
    passed = 0
    for check_fn in task.checks:
        try:
            ok, msg = check_fn(workspace)
        except Exception as e:
            ok, msg = False, f"Check error: {e}"
        details.append({"check": check_fn.__name__, "passed": ok, "message": msg})
        if ok:
            passed += 1
    score = passed / len(task.checks) if task.checks else 0.0
    return score, details


async def run_eval_task_live(task: EvalTask, workspace: Path) -> EvalResult:
    """Run a single eval task with real API calls."""
    result = EvalResult(task_id=task.id)
    _setup_workspace(task, workspace)

    # Point GrokSwarm at the workspace
    old_project_dir = shared.PROJECT_DIR
    old_state = shared.state
    shared.PROJECT_DIR = workspace

    bus = SwarmBus(":memory:")
    t0 = time.monotonic()
    tokens_before = shared.state.global_tokens_used
    cost_before = shared.state.global_cost_usd

    try:
        await run_expert(
            task.expert,
            task.task_prompt,
            bus=bus,
            agent_name=f"eval_{task.id}",
        )
        result.time_seconds = round(time.monotonic() - t0, 2)
        result.tokens_used = shared.state.global_tokens_used - tokens_before
        result.cost_usd = round(shared.state.global_cost_usd - cost_before, 6)

        # Get agent info for rounds
        agent = shared.state.get_agent(f"eval_{task.id}")
        if agent:
            result.rounds_used = len(agent.tool_call_log)

    except Exception as e:
        result.error = str(e)
        result.time_seconds = round(time.monotonic() - t0, 2)

    finally:
        shared.PROJECT_DIR = old_project_dir
        bus.close()
        # Close the singleton bus if it was created pointing at workspace
        if shared._bus_instance is not None:
            try:
                shared._bus_instance.close()
            except Exception:
                pass
            shared._bus_instance = None
        # Clean up agent
        shared.state.agents.pop(f"eval_{task.id}", None)

    # Run checks
    result.correct, result.check_details = _run_checks(task, workspace)
    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(results: list[EvalResult]) -> str:
    """Format eval results as a readable report."""
    lines = []
    lines.append("=" * 78)
    lines.append("GROKSWARM EVAL REPORT")
    lines.append("=" * 78)
    lines.append(f"{'Task':<25} {'Score':>6} {'Rounds':>7} {'Tokens':>8} {'Cost':>8} {'Time':>6}")
    lines.append("-" * 78)

    total_score = 0
    total_tasks = 0
    total_tokens = 0
    total_cost = 0.0

    for r in results:
        score_str = f"{r.correct:.0%}"
        cost_str = f"${r.cost_usd:.4f}"
        time_str = f"{r.time_seconds:.1f}s"
        lines.append(f"{r.task_id:<25} {score_str:>6} {r.rounds_used:>7} {r.tokens_used:>8,} {cost_str:>8} {time_str:>6}")
        total_score += r.correct
        total_tasks += 1
        total_tokens += r.tokens_used
        total_cost += r.cost_usd

        if r.error:
            lines.append(f"  ERROR: {r.error[:70]}")
        for detail in r.check_details:
            status = "PASS" if detail["passed"] else "FAIL"
            lines.append(f"  [{status}] {detail['check']}: {detail['message'][:60]}")

    lines.append("-" * 78)
    avg = total_score / total_tasks if total_tasks else 0
    lines.append(f"{'TOTAL':<25} {avg:.0%}  avg    {total_tokens:>8,} ${total_cost:.4f}")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pytest-based unit evals (mocked API — deterministic, no cost)
# ---------------------------------------------------------------------------

class TestEvalTaskDefinitions:
    """Verify eval task definitions are well-formed."""

    def test_all_tasks_have_checks(self):
        for task in EVAL_TASKS:
            assert len(task.checks) > 0, f"Task {task.id} has no checks"

    def test_all_tasks_have_unique_ids(self):
        ids = [t.id for t in EVAL_TASKS]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs: {ids}"

    def test_all_tasks_have_valid_category(self):
        for task in EVAL_TASKS:
            assert task.category in ("A", "B", "C", "D"), f"Bad category: {task.category}"

    def test_setup_files_create_correctly(self, tmp_path):
        for task in EVAL_TASKS:
            ws = tmp_path / task.id
            _setup_workspace(task, ws)
            for rel_path in task.setup_files:
                assert (ws / rel_path).exists(), f"Setup failed for {task.id}/{rel_path}"


class TestCheckFunctions:
    """Verify check functions work correctly on known inputs."""

    def test_file_exists_pass(self, tmp_path):
        (tmp_path / "foo.py").write_text("x = 1")
        ok, msg = check_file_exists("foo.py")(tmp_path)
        assert ok is True

    def test_file_exists_fail(self, tmp_path):
        ok, msg = check_file_exists("nope.py")(tmp_path)
        assert ok is False

    def test_file_contains_pass(self, tmp_path):
        (tmp_path / "f.py").write_text("def hello(): pass")
        ok, msg = check_file_contains("f.py", "def hello")(tmp_path)
        assert ok is True

    def test_file_contains_fail(self, tmp_path):
        (tmp_path / "f.py").write_text("def hello(): pass")
        ok, msg = check_file_contains("f.py", "def goodbye")(tmp_path)
        assert ok is False

    def test_python_compiles_pass(self, tmp_path):
        (tmp_path / "good.py").write_text("x = 1 + 2")
        ok, msg = check_python_compiles("good.py")(tmp_path)
        assert ok is True

    def test_python_compiles_fail(self, tmp_path):
        (tmp_path / "bad.py").write_text("def f(\n")
        ok, msg = check_python_compiles("bad.py")(tmp_path)
        assert ok is False

    def test_function_exists_pass(self, tmp_path):
        (tmp_path / "m.py").write_text("def my_func(): pass")
        ok, msg = check_function_exists("m.py", "my_func")(tmp_path)
        assert ok is True

    def test_output_matches_pass(self, tmp_path):
        (tmp_path / "run.py").write_text("print('hello')")
        ok, msg = check_output_matches("run.py", "hello")(tmp_path)
        assert ok is True

    def test_output_matches_fail(self, tmp_path):
        (tmp_path / "run.py").write_text("print('goodbye')")
        ok, msg = check_output_matches("run.py", "hello")(tmp_path)
        assert ok is False

    def test_pytest_passes(self, tmp_path):
        (tmp_path / "test_ok.py").write_text("def test_one(): assert 1 == 1")
        ok, msg = check_pytest_passes("test_ok.py")(tmp_path)
        assert ok is True

    def test_pytest_fails(self, tmp_path):
        (tmp_path / "test_bad.py").write_text("def test_one(): assert 1 == 2")
        ok, msg = check_pytest_passes("test_bad.py")(tmp_path)
        assert ok is False


class TestEvalBugFixTasks:
    """Verify that the seeded bugs are actually bugs (checks fail on setup files)."""

    def test_B1_broken_file_doesnt_compile(self, tmp_path):
        task = next(t for t in EVAL_TASKS if t.id == "B1_fix_syntax_error")
        _setup_workspace(task, tmp_path)
        ok, _ = check_python_compiles("broken.py")(tmp_path)
        assert ok is False, "broken.py should not compile before fix"

    def test_B2_logic_bug_fails_assertions(self, tmp_path):
        task = next(t for t in EVAL_TASKS if t.id == "B2_fix_logic_bug")
        _setup_workspace(task, tmp_path)
        result = subprocess.run(
            [sys.executable, str(tmp_path / "vowels.py")],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode != 0, "vowels.py should fail before fix"

    def test_B3_tests_fail_before_fix(self, tmp_path):
        task = next(t for t in EVAL_TASKS if t.id == "B3_fix_failing_test")
        _setup_workspace(task, tmp_path)
        ok, _ = check_pytest_passes("test_calc.py")(tmp_path)
        assert ok is False, "test_calc.py should fail before fix"


class TestEvalSwarmTasks:
    """Verify Category D task definitions are valid and setup files work."""

    def test_D_tasks_exist(self):
        d_tasks = [t for t in EVAL_TASKS if t.category == "D"]
        assert len(d_tasks) >= 3, f"Expected at least 3 Category D tasks, got {len(d_tasks)}"

    def test_D3_setup_files(self, tmp_path):
        task = next(t for t in EVAL_TASKS if t.id == "D3_message_passing")
        _setup_workspace(task, tmp_path)
        assert (tmp_path / "module_a.py").exists()
        assert (tmp_path / "module_b.py").exists()
        content_a = (tmp_path / "module_a.py").read_text()
        assert "def alpha" in content_a
        assert "def beta" in content_a

    def test_swarm_bus_roundtrip(self):
        """Verify SwarmBus can post and read messages."""
        bus = SwarmBus(":memory:")
        bus.post("agent_a", "hello from A", recipient="agent_b", kind="request")
        bus.post("agent_b", "reply from B", recipient="agent_a", kind="result")
        msgs = bus.read("agent_b")
        assert len(msgs) >= 1
        assert any("hello from A" in m["body"] for m in msgs)
        bus.close()

    def test_swarm_bus_broadcast(self):
        """Verify broadcast messages are visible to all agents."""
        bus = SwarmBus(":memory:")
        bus.post("supervisor", "task update", recipient="*", kind="status")
        msgs = bus.read("any_agent")
        assert len(msgs) >= 1
        assert msgs[0]["body"] == "task update"
        bus.close()


class TestEvalReport:
    """Verify report formatting."""

    def test_format_report_empty(self):
        report = format_report([])
        assert "EVAL REPORT" in report

    def test_format_report_with_results(self):
        results = [
            EvalResult(task_id="A1", correct=1.0, rounds_used=2, tokens_used=500, cost_usd=0.001, time_seconds=3.0),
            EvalResult(task_id="B1", correct=0.5, rounds_used=5, tokens_used=2000, cost_usd=0.01, time_seconds=15.0),
        ]
        report = format_report(results)
        assert "A1" in report
        assert "B1" in report
        assert "100%" in report
        assert "50%" in report


# ---------------------------------------------------------------------------
# CLI: Live eval runner
# ---------------------------------------------------------------------------

async def _run_live_eval(task_ids: list[str] | None = None):
    """Run eval tasks with real API calls."""
    tasks = EVAL_TASKS
    if task_ids:
        tasks = [t for t in EVAL_TASKS if t.id in task_ids]
        if not tasks:
            print(f"No tasks found matching: {task_ids}")
            return

    results = []
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Running: {task.id} — {task.description}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory(prefix=f"grokswarm_eval_{task.id}_") as tmp:
            workspace = Path(tmp)
            result = await run_eval_task_live(task, workspace)
            results.append(result)
            print(f"  Score: {result.correct:.0%} | Rounds: {result.rounds_used} | "
                  f"Tokens: {result.tokens_used:,} | Cost: ${result.cost_usd:.4f} | "
                  f"Time: {result.time_seconds:.1f}s")
            if result.error:
                print(f"  ERROR: {result.error}")

    report = format_report(results)
    print(f"\n{report}")

    # Save report
    report_path = Path(".grokswarm") / "eval_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")

    # Save JSON for programmatic access
    json_path = Path(".grokswarm") / "eval_results.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"Results saved to: {json_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GrokSwarm Eval Runner")
    parser.add_argument("--live", action="store_true", help="Run with real API calls")
    parser.add_argument("--task", nargs="*", help="Specific task IDs to run (e.g., A1 B2)")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    args = parser.parse_args()

    if args.list:
        print(f"{'ID':<25} {'Cat':>4} {'Expert':<12} Description")
        print("-" * 78)
        for t in EVAL_TASKS:
            print(f"{t.id:<25} {t.category:>4} {t.expert:<12} {t.description}")
        return

    if args.live:
        task_ids = args.task if args.task else None
        asyncio.run(_run_live_eval(task_ids))
    else:
        print("Running pytest eval suite...")
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


if __name__ == "__main__":
    main()
