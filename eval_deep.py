"""
eval_deep.py — Deep Comparative Evaluation Framework for GrokSwarm

Measures whether swarm architecture adds genuine value over a single agent
by running identical tasks through both single-agent and multi-agent paths.

Two modes:
  1. Unit eval (mocked API):  python -m pytest eval_deep.py -v
  2. Live eval (real API):    python eval_deep.py --live [--task E1]

Categories:
  E — Comparative: Same task, single vs swarm
  F — Parallel Speedup: Independent subtasks, parallel vs sequential
  G — Adversarial Quality: Seeded bugs + ambiguous specs
  H — Scale Beyond Context: Task too large for one agent's context
  I — Learning Over Time: Repeat after LessonsDB has data
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
from typing import Callable
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Only set a fake key if none exists AND we're not running --live
# (shared.py loads the real key from .env via dotenv)
if "--live" not in sys.argv and not os.environ.get("XAI_API_KEY"):
    os.environ["XAI_API_KEY"] = "test-key-for-eval"

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import grokswarm.shared as shared
import grokswarm.agents as agents
from grokswarm.agents import run_expert, SwarmBus, _record_usage
from grokswarm.guardrails import Orchestrator, LessonsDB, reset_model_tiers

# Reuse check helpers from the existing eval framework
from eval_grokswarm import (
    check_file_exists,
    check_file_contains,
    check_file_not_contains,
    check_python_compiles,
    check_pytest_passes,
    check_function_exists,
    check_output_matches,
    _setup_workspace as _setup_workspace_basic,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CATEGORY_MULTIPLIERS = {
    "correctness": 3.0,
    "quality": 2.0,
    "edge_cases": 2.0,
    "completeness": 1.0,
}


@dataclass
class WeightedCheck:
    """A single evaluation check with weight and category."""
    name: str
    check_fn: Callable  # (workspace: Path) -> (bool, str)
    weight: float = 1.0
    category: str = "correctness"  # correctness | quality | edge_cases | completeness


@dataclass
class DeepEvalTask:
    """One deep evaluation task definition."""
    id: str
    category: str              # E, F, G, H, I
    description: str
    task_prompt: str
    setup_files: dict          # {relative_path: content}
    checks: list[WeightedCheck] = field(default_factory=list)
    expert: str = "coder"
    max_rounds: int = 20
    timeout: int = 180
    use_swarm: bool = True     # Whether to also run through Orchestrator
    learning_seed: dict | None = None  # Pre-seed LessonsDB for Category I
    baseline_scores: dict | None = None  # {"claude_code": 0.85} for comparison


@dataclass
class RunMetrics:
    """Metrics from a single evaluation run."""
    quality_score: float = 0.0   # Weighted check score 0.0-1.0
    time_seconds: float = 0.0
    cost_usd: float = 0.0
    tokens_used: int = 0
    rounds_used: int = 0
    checks_passed: int = 0
    checks_total: int = 0
    check_details: list[dict] = field(default_factory=list)
    planning_time: float = 0.0   # Orchestrator decomposition time (swarm only)
    deliberation_time: float = 0.0  # Dualhead deliberation time (excluded from exec time)
    error: str = ""


@dataclass
class ComparativeResult:
    """Result comparing single-agent vs swarm on the same task."""
    task_id: str
    category: str = ""
    description: str = ""
    single: RunMetrics = field(default_factory=RunMetrics)
    swarm: RunMetrics = field(default_factory=RunMetrics)
    verdict: str = "tie"        # "swarm_better" | "single_better" | "tie"
    quality_delta: float = 0.0  # swarm.quality - single.quality
    speedup: float = 1.0        # single.time / swarm.time
    cost_ratio: float = 1.0     # swarm.cost / single.cost
    # Efficiency scores (quality adjusted for cost and time)
    single_cost_score: float = 0.0
    single_time_score: float = 0.0
    single_overall: float = 0.0
    swarm_cost_score: float = 0.0
    swarm_time_score: float = 0.0
    swarm_overall: float = 0.0


@dataclass
class StatisticalResult:
    """Aggregated result from N runs of the same task."""
    task_id: str
    n_runs: int = 0
    single_scores: list[float] = field(default_factory=list)
    swarm_scores: list[float] = field(default_factory=list)
    single_mean: float = 0.0
    single_stddev: float = 0.0
    swarm_mean: float = 0.0
    swarm_stddev: float = 0.0
    single_times: list[float] = field(default_factory=list)
    swarm_times: list[float] = field(default_factory=list)
    single_costs: list[float] = field(default_factory=list)
    swarm_costs: list[float] = field(default_factory=list)
    significant: bool = False  # delta > max(single_stddev, swarm_stddev)
    verdict: str = "inconclusive"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _compute_efficiency(quality: float, cost: float) -> float:
    """Quality per dollar. Higher is better."""
    return quality / max(cost, 0.001)


def _compute_verdict_v2(single: RunMetrics, swarm: RunMetrics) -> tuple[str, float, float, float]:
    """Cost-adjusted verdict. Same as _compute_verdict but with efficiency awareness."""
    quality_delta = swarm.quality_score - single.quality_score

    # Use exec time (excluding deliberation) for speedup
    s_exec = single.time_seconds - single.deliberation_time
    w_exec = swarm.time_seconds - swarm.deliberation_time
    if s_exec > 0 and w_exec > 0:
        speedup = s_exec / w_exec
    else:
        speedup = 1.0

    if single.cost_usd > 0 and swarm.cost_usd > 0:
        cost_ratio = swarm.cost_usd / single.cost_usd
    else:
        cost_ratio = 1.0

    single_eff = _compute_efficiency(single.quality_score, single.cost_usd)
    swarm_eff = _compute_efficiency(swarm.quality_score, swarm.cost_usd)

    if quality_delta > 0.05:
        if cost_ratio <= 2.0 or swarm_eff >= single_eff:
            verdict = "swarm_efficient"
        else:
            verdict = "swarm_better_but_costly"
    elif quality_delta < -0.05:
        verdict = "single_efficient"
    else:
        verdict = "tie"

    return verdict, quality_delta, round(speedup, 2), round(cost_ratio, 2)


# ---------------------------------------------------------------------------
# Weighted scoring
# ---------------------------------------------------------------------------

def _run_weighted_checks(checks: list[WeightedCheck], workspace: Path) -> RunMetrics:
    """Run all weighted checks and compute a weighted quality score."""
    details = []
    total_weight = 0.0
    earned_weight = 0.0
    passed_count = 0

    for wc in checks:
        multiplier = CATEGORY_MULTIPLIERS.get(wc.category, 1.0)
        effective_weight = wc.weight * multiplier
        total_weight += effective_weight

        try:
            ok, msg = wc.check_fn(workspace)
        except Exception as e:
            ok, msg = False, f"Check error: {e}"

        if ok:
            earned_weight += effective_weight
            passed_count += 1

        details.append({
            "check": wc.name,
            "category": wc.category,
            "weight": wc.weight,
            "multiplier": multiplier,
            "passed": ok,
            "message": msg,
        })

    score = earned_weight / total_weight if total_weight > 0 else 0.0
    return RunMetrics(
        quality_score=round(score, 4),
        checks_passed=passed_count,
        checks_total=len(checks),
        check_details=details,
    )


def _compute_verdict(single: RunMetrics, swarm: RunMetrics) -> tuple[str, float, float, float]:
    """Compute verdict, quality_delta, speedup, cost_ratio."""
    quality_delta = swarm.quality_score - single.quality_score

    # Use exec time (excluding deliberation) for speedup
    s_exec = single.time_seconds - single.deliberation_time
    w_exec = swarm.time_seconds - swarm.deliberation_time
    if s_exec > 0 and w_exec > 0:
        speedup = s_exec / w_exec
    else:
        speedup = 1.0

    if single.cost_usd > 0 and swarm.cost_usd > 0:
        cost_ratio = swarm.cost_usd / single.cost_usd
    else:
        cost_ratio = 1.0

    if quality_delta > 0.05:
        verdict = "swarm_better"
    elif quality_delta < -0.05:
        verdict = "single_better"
    else:
        verdict = "tie"

    return verdict, quality_delta, round(speedup, 2), round(cost_ratio, 2)


def _compute_efficiency_scores(r: ComparativeResult) -> None:
    """Compute cost/time/overall efficiency scores for a ComparativeResult.

    Mutates *r* in-place. When only one path ran (e.g. learning eval),
    cost and time scores equal raw quality (no comparison penalty).
    """
    sq = r.single.quality_score
    wq = r.swarm.quality_score
    has_swarm = r.swarm.checks_total > 0

    if has_swarm:
        # Cost scores: quality * (min_cost / this_cost)
        min_cost = min(r.single.cost_usd, r.swarm.cost_usd) if r.single.cost_usd > 0 and r.swarm.cost_usd > 0 else 0
        if min_cost > 0:
            r.single_cost_score = round(sq * (min_cost / r.single.cost_usd), 4)
            r.swarm_cost_score = round(wq * (min_cost / r.swarm.cost_usd), 4)
        else:
            r.single_cost_score = round(sq, 4)
            r.swarm_cost_score = round(wq, 4)

        # Time scores: quality * (min_exec_time / this_exec_time) — excludes deliberation
        s_exec = r.single.time_seconds - r.single.deliberation_time
        w_exec = r.swarm.time_seconds - r.swarm.deliberation_time
        min_time = min(s_exec, w_exec) if s_exec > 0 and w_exec > 0 else 0
        if min_time > 0:
            r.single_time_score = round(sq * (min_time / s_exec), 4)
            r.swarm_time_score = round(wq * (min_time / w_exec), 4)
        else:
            r.single_time_score = round(sq, 4)
            r.swarm_time_score = round(wq, 4)
    else:
        # Only one path ran — scores equal raw quality
        r.single_cost_score = round(sq, 4)
        r.single_time_score = round(sq, 4)
        r.swarm_cost_score = 0.0
        r.swarm_time_score = 0.0

    # Overall = mean(quality, cost_score, time_score)
    r.single_overall = round((sq + r.single_cost_score + r.single_time_score) / 3, 4)
    if has_swarm:
        r.swarm_overall = round((wq + r.swarm_cost_score + r.swarm_time_score) / 3, 4)
    else:
        r.swarm_overall = 0.0


# ---------------------------------------------------------------------------
# Improvement notes generation
# ---------------------------------------------------------------------------

def _generate_notes(checks_single: list[dict], checks_swarm: list[dict]) -> dict:
    """Analyze check details to produce strengths, weaknesses, and suggestions.

    Returns {"strengths": [...], "weaknesses": [...], "suggestions": [...]}.
    """
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []

    # Group checks by category and count pass/fail per path
    for label, checks in [("Single", checks_single), ("Swarm", checks_swarm)]:
        if not checks:
            continue
        by_cat: dict[str, list[dict]] = {}
        for c in checks:
            cat = c.get("category", "other")
            by_cat.setdefault(cat, []).append(c)

        for cat, cat_checks in by_cat.items():
            passed = [c for c in cat_checks if c.get("passed")]
            failed = [c for c in cat_checks if not c.get("passed")]

            if len(passed) == len(cat_checks) and len(cat_checks) > 0:
                strengths.append(f"All {len(passed)} {cat} checks passed ({label})")

            for c in failed:
                w = c.get("weight", 1.0)
                msg = str(c.get("message", ""))[:80]
                weaknesses.append(
                    f"[{label}] {c.get('check', '?')} ({cat} w={w:.1f}): {msg}"
                )
                # Derive actionable suggestion from check name + message
                hint = _suggest_fix(c.get("check", ""), msg)
                if hint:
                    suggestions.append(hint)

    # Note differences between paths when both ran
    if checks_single and checks_swarm:
        single_names = {c.get("check") for c in checks_single if c.get("passed")}
        swarm_names = {c.get("check") for c in checks_swarm if c.get("passed")}
        single_only = single_names - swarm_names
        swarm_only = swarm_names - single_names
        if single_only:
            strengths.append(f"Single passed {len(single_only)} check(s) that Swarm missed")
        if swarm_only:
            strengths.append(f"Swarm passed {len(swarm_only)} check(s) that Single missed")

    # Deduplicate suggestions
    seen: set[str] = set()
    unique_suggestions: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique_suggestions.append(s)

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": unique_suggestions,
    }


def _suggest_fix(check_name: str, message: str) -> str:
    """Derive an actionable suggestion from a failing check's name and message."""
    name = check_name.lower()
    msg = message.lower()

    if "error" in name or "error" in msg:
        return f"Improve error handling for: {check_name}"
    if "exit" in name or "exit code" in msg:
        return f"Fix exit code behavior for: {check_name}"
    if "missing" in msg or "not found" in msg:
        return f"Ensure required output/file exists: {check_name}"
    if "import" in name or "import" in msg:
        return f"Fix import or dependency issue: {check_name}"
    if "test" in name or "pytest" in msg:
        return f"Fix failing test: {check_name}"
    if "syntax" in msg or "compile" in name:
        return f"Fix syntax/compilation error: {check_name}"
    if message.strip():
        return f"Address: {check_name} — {message[:60]}"
    return ""


# ---------------------------------------------------------------------------
# Persistent score storage
# ---------------------------------------------------------------------------

def _save_eval_scores(results: list[ComparativeResult]) -> Path:
    """Save efficiency scores to .grokswarm/eval_scores.json (atomic write).

    Keyed by task_id — re-running a task overwrites its entry.
    Returns the path of the written file.
    """
    scores_path = Path(".grokswarm") / "eval_scores.json"
    scores_path.parent.mkdir(exist_ok=True)

    # Load existing
    existing: dict = {}
    if scores_path.exists():
        try:
            existing = json.loads(scores_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Upsert
    import datetime as _dt
    for r in results:
        existing[r.task_id] = {
            "task_id": r.task_id,
            "category": r.category,
            "description": r.description,
            "verdict": r.verdict,
            "single_quality": r.single.quality_score,
            "single_cost_usd": r.single.cost_usd,
            "single_time_s": r.single.time_seconds,
            "single_exec_time_s": r.single.time_seconds - r.single.deliberation_time,
            "single_deliberation_time_s": r.single.deliberation_time,
            "single_cost_score": r.single_cost_score,
            "single_time_score": r.single_time_score,
            "single_overall": r.single_overall,
            "swarm_quality": r.swarm.quality_score,
            "swarm_cost_usd": r.swarm.cost_usd,
            "swarm_time_s": r.swarm.time_seconds,
            "swarm_exec_time_s": r.swarm.time_seconds - r.swarm.deliberation_time,
            "swarm_deliberation_time_s": r.swarm.deliberation_time,
            "swarm_cost_score": r.swarm_cost_score,
            "swarm_time_score": r.swarm_time_score,
            "swarm_overall": r.swarm_overall,
            "single_checks": r.single.check_details,
            "swarm_checks": r.swarm.check_details,
            "notes": _generate_notes(r.single.check_details, r.swarm.check_details),
            "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        }

    # Atomic write (tmp + replace)
    tmp_path = scores_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    tmp_path.replace(scores_path)
    return scores_path


def _load_eval_scores() -> dict:
    """Load saved eval scores from .grokswarm/eval_scores.json."""
    scores_path = Path(".grokswarm") / "eval_scores.json"
    if not scores_path.exists():
        return {}
    try:
        return json.loads(scores_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Check function builders (extending eval_grokswarm's set)
# ---------------------------------------------------------------------------

def wcheck(name: str, check_fn: Callable, weight: float = 1.0,
           category: str = "correctness") -> WeightedCheck:
    """Shorthand to create a WeightedCheck."""
    return WeightedCheck(name=name, check_fn=check_fn, weight=weight, category=category)


def check_cli_args(path: str, args: list[str], expected_stdout: str):
    """Check that running a script with given args produces expected stdout."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        result = subprocess.run(
            [sys.executable, str(f)] + args,
            capture_output=True, text=True, timeout=15, cwd=str(workspace)
        )
        actual = result.stdout.strip()
        expected = expected_stdout.strip()
        if expected in actual:
            return True, f"Output contains expected: '{expected[:50]}'"
        return False, f"Expected '{expected[:50]}' in output, got '{actual[:60]}'"
    _check.__name__ = f"cli_args({path}, {args})"
    return _check


def check_cli_exitcode(path: str, args: list[str], expected_code: int):
    """Check that running a script with given args returns expected exit code."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        result = subprocess.run(
            [sys.executable, str(f)] + args,
            capture_output=True, text=True, timeout=15, cwd=str(workspace)
        )
        if result.returncode == expected_code:
            return True, f"Exit code {expected_code} as expected"
        return False, f"Expected exit code {expected_code}, got {result.returncode}"
    _check.__name__ = f"cli_exitcode({path}, {args}, {expected_code})"
    return _check


def check_bug_fixed(path: str, bad_pattern: str, good_pattern: str):
    """Check that a specific bug pattern was replaced with the fix."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        content = f.read_text(errors="replace")
        if bad_pattern in content:
            return False, f"Bug still present: '{bad_pattern[:40]}'"
        if good_pattern in content:
            return True, f"Bug fixed: found '{good_pattern[:40]}'"
        return False, f"Neither bug pattern nor fix found"
    _check.__name__ = f"bug_fixed({path}, '{bad_pattern[:20]}')"
    return _check


def check_class_exists(path: str, class_name: str):
    """Check that a class is defined in a file."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        content = f.read_text(errors="replace")
        if f"class {class_name}" in content:
            return True, f"Class '{class_name}' found in {path}"
        return False, f"Class '{class_name}' not found in {path}"
    _check.__name__ = f"class_exists({path}, {class_name})"
    return _check


def check_import_works(path: str, import_name: str):
    """Check that a module can be imported successfully."""
    def _check(workspace: Path) -> tuple[bool, str]:
        f = workspace / path
        if not f.exists():
            return False, f"File missing: {path}"
        module_name = path.replace("/", ".").replace("\\", ".").removesuffix(".py")
        result = subprocess.run(
            [sys.executable, "-c", f"import sys; sys.path.insert(0, '.'); import {module_name}"],
            capture_output=True, text=True, timeout=10, cwd=str(workspace)
        )
        if result.returncode == 0:
            return True, f"Import {module_name} succeeded"
        return False, f"Import {module_name} failed: {result.stderr[:100]}"
    _check.__name__ = f"import_works({path})"
    return _check


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# -- E1: CLI Calculator (Comparative) --

TASK_E1 = DeepEvalTask(
    id="E1",
    category="E",
    description="CLI Calculator — single vs swarm comparison",
    task_prompt=textwrap.dedent("""\
        Create a command-line calculator `calc.py` that:
        1. Uses argparse with subcommands: add, sub, mul, div
        2. Each subcommand takes two float arguments: a and b
        3. Prints the result to stdout (just the number)
        4. Division by zero prints "Error: division by zero" and exits with code 1
        5. --help shows usage information
        6. Supports negative numbers and floating point
        7. Returns exit code 0 on success, 1 on error

        Examples:
          python calc.py add 2 3      -> 5.0
          python calc.py sub 10 3     -> 7.0
          python calc.py mul 4 5      -> 20.0
          python calc.py div 10 3     -> 3.3333333333333335
          python calc.py div 1 0      -> Error: division by zero (exit 1)
          python calc.py add -5 3     -> -2.0
    """),
    setup_files={},
    expert="coder",
    checks=[
        # File basics (correctness)
        wcheck("file_exists", check_file_exists("calc.py"), 1.0, "correctness"),
        wcheck("compiles", check_python_compiles("calc.py"), 2.0, "correctness"),
        wcheck("has_argparse", check_file_contains("calc.py", "argparse"), 1.0, "correctness"),
        # Operations (correctness)
        wcheck("add_works", check_cli_args("calc.py", ["add", "2", "3"], "5"), 2.0, "correctness"),
        wcheck("sub_works", check_cli_args("calc.py", ["sub", "10", "3"], "7"), 2.0, "correctness"),
        wcheck("mul_works", check_cli_args("calc.py", ["mul", "4", "5"], "20"), 2.0, "correctness"),
        wcheck("div_works", check_cli_args("calc.py", ["div", "10", "2"], "5"), 2.0, "correctness"),
        # Error handling (edge_cases)
        wcheck("div_zero_msg", check_cli_args("calc.py", ["div", "1", "0"], "Error"), 2.0, "edge_cases"),
        wcheck("div_zero_exit", check_cli_exitcode("calc.py", ["div", "1", "0"], 1), 1.5, "edge_cases"),
        wcheck("no_args_exit", check_cli_exitcode("calc.py", [], 2), 1.0, "edge_cases"),
        # Quality features
        wcheck("help_text", check_cli_args("calc.py", ["--help"], "usage"), 1.0, "quality"),
        wcheck("negative_numbers", check_cli_args("calc.py", ["add", "--", "-5", "3"], "-2"), 1.5, "quality"),
        wcheck("float_support", check_cli_args("calc.py", ["div", "10", "3"], "3.333"), 1.5, "quality"),
        # Completeness
        wcheck("has_main_guard", check_file_contains("calc.py", '__name__'), 0.5, "completeness"),
        wcheck("exit_code_success", check_cli_exitcode("calc.py", ["add", "1", "1"], 0), 1.0, "completeness"),
    ],
)

# -- E2: Four Independent Processor Modules (Parallel-Friendly Comparison) --

TASK_E2 = DeepEvalTask(
    id="E2",
    category="E",
    description="Four independent processor modules — parallel-friendly comparison",
    task_prompt=textwrap.dedent("""\
        Create four independent Python processor modules, each with its own test file.
        These are four separate, independent modules with no shared dependencies between them.

        1. `csv_processor.py` with functions:
           - parse_csv(text: str) -> list[list[str]]  (split lines by comma, strip whitespace)
           - filter_rows(rows: list[list[str]], column: int, value: str) -> list[list[str]]
           - to_csv(rows: list[list[str]]) -> str  (join back to CSV text)
           Create `test_csv_processor.py` with at least 4 tests.

        2. `text_processor.py` with functions:
           - word_frequency(text: str) -> dict[str, int]  (case-insensitive)
           - find_longest_word(text: str) -> str
           - truncate(text: str, max_len: int) -> str  (add "..." if truncated)
           Create `test_text_processor.py` with at least 4 tests.

        3. `json_processor.py` with functions:
           - flatten(obj: dict, prefix: str = "") -> dict  (flatten nested dicts with dot notation keys)
           - merge(a: dict, b: dict) -> dict  (deep merge, b overrides a)
           - extract_keys(obj: dict, keys: list[str]) -> dict  (pick only listed keys)
           Create `test_json_processor.py` with at least 4 tests.

        4. `date_processor.py` with functions:
           - parse_date(s: str) -> datetime.date  (accept "YYYY-MM-DD" format)
           - days_between(a: str, b: str) -> int  (absolute difference)
           - add_days(date_str: str, n: int) -> str  (return "YYYY-MM-DD")
           Create `test_date_processor.py` with at least 4 tests.

        Run all tests to verify they pass.
    """),
    setup_files={},
    expert="coder",
    checks=[
        # csv_processor (correctness)
        wcheck("csv_exists", check_file_exists("csv_processor.py"), 1.0, "correctness"),
        wcheck("csv_compiles", check_python_compiles("csv_processor.py"), 1.5, "correctness"),
        wcheck("has_parse_csv", check_function_exists("csv_processor.py", "parse_csv"), 1.0, "correctness"),
        wcheck("has_filter_rows", check_function_exists("csv_processor.py", "filter_rows"), 1.0, "correctness"),
        wcheck("has_to_csv", check_function_exists("csv_processor.py", "to_csv"), 1.0, "correctness"),
        wcheck("test_csv_exists", check_file_exists("test_csv_processor.py"), 0.5, "completeness"),
        wcheck("test_csv_pass", check_pytest_passes("test_csv_processor.py"), 2.0, "correctness"),
        # text_processor (correctness)
        wcheck("text_exists", check_file_exists("text_processor.py"), 1.0, "correctness"),
        wcheck("text_compiles", check_python_compiles("text_processor.py"), 1.5, "correctness"),
        wcheck("has_word_frequency", check_function_exists("text_processor.py", "word_frequency"), 1.0, "correctness"),
        wcheck("has_find_longest_word", check_function_exists("text_processor.py", "find_longest_word"), 1.0, "correctness"),
        wcheck("has_truncate", check_function_exists("text_processor.py", "truncate"), 1.0, "correctness"),
        wcheck("test_text_exists", check_file_exists("test_text_processor.py"), 0.5, "completeness"),
        wcheck("test_text_pass", check_pytest_passes("test_text_processor.py"), 2.0, "correctness"),
        # json_processor (correctness)
        wcheck("json_exists", check_file_exists("json_processor.py"), 1.0, "correctness"),
        wcheck("json_compiles", check_python_compiles("json_processor.py"), 1.5, "correctness"),
        wcheck("has_flatten", check_function_exists("json_processor.py", "flatten"), 1.0, "correctness"),
        wcheck("has_merge", check_function_exists("json_processor.py", "merge"), 1.0, "correctness"),
        wcheck("has_extract_keys", check_function_exists("json_processor.py", "extract_keys"), 1.0, "correctness"),
        wcheck("test_json_exists", check_file_exists("test_json_processor.py"), 0.5, "completeness"),
        wcheck("test_json_pass", check_pytest_passes("test_json_processor.py"), 2.0, "correctness"),
        # date_processor (correctness)
        wcheck("date_exists", check_file_exists("date_processor.py"), 1.0, "correctness"),
        wcheck("date_compiles", check_python_compiles("date_processor.py"), 1.5, "correctness"),
        wcheck("has_parse_date", check_function_exists("date_processor.py", "parse_date"), 1.0, "correctness"),
        wcheck("has_days_between", check_function_exists("date_processor.py", "days_between"), 1.0, "correctness"),
        wcheck("has_add_days", check_function_exists("date_processor.py", "add_days"), 1.0, "correctness"),
        wcheck("test_date_exists", check_file_exists("test_date_processor.py"), 0.5, "completeness"),
        wcheck("test_date_pass", check_pytest_passes("test_date_processor.py"), 2.0, "correctness"),
        # Quality: all modules importable
        wcheck("csv_importable", check_import_works("csv_processor.py", "csv_processor"), 0.5, "quality"),
        wcheck("text_importable", check_import_works("text_processor.py", "text_processor"), 0.5, "quality"),
        wcheck("json_importable", check_import_works("json_processor.py", "json_processor"), 0.5, "quality"),
        wcheck("date_importable", check_import_works("date_processor.py", "date_processor"), 0.5, "quality"),
    ],
)

# -- E3: Three Independent Algorithm Modules (Heavy Parallel Comparison) --

TASK_E3 = DeepEvalTask(
    id="E3",
    category="E",
    description="Three independent algorithm modules — heavy parallel comparison",
    task_prompt=textwrap.dedent("""\
        Create three independent Python algorithm modules, each with its own test file.
        These are three separate, independent modules with no shared dependencies.
        Each module must also work as a CLI tool.

        1. `expr_eval.py` — Mathematical expression evaluator
           Functions:
           - tokenize(expr: str) -> list[str]: Split expression into tokens
             (numbers, operators +, -, *, /, and parentheses).
           - evaluate(expr: str, variables: dict | None = None) -> float:
             Evaluate a mathematical expression string.
             Supports: +, -, *, /, parentheses for grouping, float literals,
             and variable names that are looked up in the variables dict.
             Operator precedence: * and / bind tighter than + and -.
             Raises ValueError for: division by zero, unbalanced parentheses,
             undefined variables, empty or invalid expressions.
           CLI usage: python expr_eval.py "<expression>" [--var NAME=VALUE ...]
             Prints the numeric result to stdout.
             Exit code 1 on any error, with message to stderr.
           Examples:
             python expr_eval.py "2 + 3"          -> 5.0
             python expr_eval.py "2 + 3 * 4"      -> 14.0
             python expr_eval.py "(2 + 3) * 4"    -> 20.0
             python expr_eval.py --var x=5 "x * 2 + 1"  -> 11.0
             python expr_eval.py "1 / 0"          -> error, exit 1
           Create `test_expr_eval.py` with at least 6 tests covering precedence,
           parentheses, variables, and error cases.

        2. `pattern_match.py` — Glob-style pattern matcher
           Functions:
           - match(pattern: str, text: str) -> bool: Match text against a
             glob-style pattern. Supports these special characters:
               *     — matches zero or more of any character
               ?     — matches exactly one of any character
               [abc] — matches any single character in the set
               [!abc] — matches any single character NOT in the set
             All other characters match literally. Matching is case-sensitive.
           - filter_matches(pattern: str, texts: list[str]) -> list[str]:
             Return only texts that match the pattern.
           IMPORTANT: Do NOT use the `re` module. Implement matching from scratch
           using your own algorithm (e.g., dynamic programming or recursive backtracking).
           CLI usage: python pattern_match.py PATTERN TEXT [TEXT ...]
             Prints matching texts, one per line.
           Examples:
             python pattern_match.py "*.py" test.py readme.md  -> test.py
             python pattern_match.py "file?.txt" file1.txt fileAB.txt -> file1.txt
             python pattern_match.py "[abc]at" cat dog rat     -> cat
           Create `test_pattern_match.py` with at least 6 tests covering
           star, question mark, character classes, negated classes, and no-match cases.

        3. `graph_utils.py` — Weighted directed graph algorithms
           Graph format: dict[str, dict[str, float]] — adjacency dict with edge weights.
           Example: {"a": {"b": 1.0, "c": 4.0}, "b": {"c": 2.0}, "c": {}}
           Functions:
           - shortest_path(graph: dict, start: str, end: str) -> tuple[list[str], float]:
             Dijkstra's algorithm. Returns (path_as_node_list, total_distance).
             Raises ValueError if no path exists or start/end not in graph.
           - topological_sort(graph: dict) -> list[str]:
             Kahn's algorithm or DFS-based topological sort for DAGs.
             Raises ValueError if the graph contains a cycle.
           - has_cycle(graph: dict) -> bool:
             Return True if the directed graph contains a cycle, False otherwise.
           CLI usage:
             python graph_utils.py shortest '<json_graph>' START END
             python graph_utils.py topo '<json_graph>'
             Exit code 1 on error (no path found, cycle in topo sort, etc.)
           Examples:
             python graph_utils.py shortest '{"a":{"b":1,"c":10},"b":{"c":2},"c":{}}' a c
               -> prints path and distance 3
             python graph_utils.py topo '{"a":{"b":1},"b":{"c":1},"c":{}}'
               -> prints: a, b, c (or similar valid topological order)
           Create `test_graph_utils.py` with at least 6 tests covering shortest path,
           topological sort, cycle detection, and error cases.

        Run ALL tests to verify they pass.
    """),
    setup_files={},
    expert="coder",
    checks=[
        # --- expr_eval (13 checks) ---
        wcheck("expr_exists", check_file_exists("expr_eval.py"), 1.0, "correctness"),
        wcheck("expr_compiles", check_python_compiles("expr_eval.py"), 1.5, "correctness"),
        wcheck("has_tokenize", check_function_exists("expr_eval.py", "tokenize"), 1.0, "correctness"),
        wcheck("has_evaluate", check_function_exists("expr_eval.py", "evaluate"), 1.0, "correctness"),
        wcheck("test_expr_exists", check_file_exists("test_expr_eval.py"), 0.5, "completeness"),
        wcheck("test_expr_pass", check_pytest_passes("test_expr_eval.py"), 2.5, "correctness"),
        wcheck("expr_basic_add", check_cli_args("expr_eval.py", ["2 + 3"], "5"), 1.5, "correctness"),
        wcheck("expr_precedence", check_cli_args("expr_eval.py", ["2 + 3 * 4"], "14"), 2.0, "correctness"),
        wcheck("expr_parens", check_cli_args("expr_eval.py", ["(2 + 3) * 4"], "20"), 2.0, "correctness"),
        wcheck("expr_variables", check_cli_args("expr_eval.py", ["--var", "x=5", "x * 2 + 1"], "11"), 1.5, "quality"),
        wcheck("expr_div_zero_exit", check_cli_exitcode("expr_eval.py", ["1 / 0"], 1), 1.0, "edge_cases"),
        wcheck("expr_unbalanced_exit", check_cli_exitcode("expr_eval.py", ["(2 + 3"], 1), 1.0, "edge_cases"),
        wcheck("expr_importable", check_import_works("expr_eval.py", "expr_eval"), 0.5, "quality"),
        # --- pattern_match (11 checks) ---
        wcheck("pat_exists", check_file_exists("pattern_match.py"), 1.0, "correctness"),
        wcheck("pat_compiles", check_python_compiles("pattern_match.py"), 1.5, "correctness"),
        wcheck("has_match", check_function_exists("pattern_match.py", "match"), 1.0, "correctness"),
        wcheck("has_filter_matches", check_function_exists("pattern_match.py", "filter_matches"), 1.0, "correctness"),
        wcheck("no_re_import", check_file_not_contains("pattern_match.py", "import re"), 1.0, "quality"),
        wcheck("test_pat_exists", check_file_exists("test_pattern_match.py"), 0.5, "completeness"),
        wcheck("test_pat_pass", check_pytest_passes("test_pattern_match.py"), 2.5, "correctness"),
        wcheck("pat_star", check_cli_args("pattern_match.py", ["*.py", "test.py", "readme.md", "main.py"], "test.py"), 1.5, "correctness"),
        wcheck("pat_question", check_cli_args("pattern_match.py", ["file?.txt", "file1.txt", "fileAB.txt"], "file1.txt"), 1.5, "correctness"),
        wcheck("pat_charset", check_cli_args("pattern_match.py", ["[abc]at", "cat", "bat", "dog", "rat"], "cat"), 1.5, "edge_cases"),
        wcheck("pat_importable", check_import_works("pattern_match.py", "pattern_match"), 0.5, "quality"),
        # --- graph_utils (11 checks) ---
        wcheck("graph_exists", check_file_exists("graph_utils.py"), 1.0, "correctness"),
        wcheck("graph_compiles", check_python_compiles("graph_utils.py"), 1.5, "correctness"),
        wcheck("has_shortest_path", check_function_exists("graph_utils.py", "shortest_path"), 1.0, "correctness"),
        wcheck("has_topological_sort", check_function_exists("graph_utils.py", "topological_sort"), 1.0, "correctness"),
        wcheck("has_has_cycle", check_function_exists("graph_utils.py", "has_cycle"), 1.0, "correctness"),
        wcheck("test_graph_exists", check_file_exists("test_graph_utils.py"), 0.5, "completeness"),
        wcheck("test_graph_pass", check_pytest_passes("test_graph_utils.py"), 2.5, "correctness"),
        wcheck("graph_shortest", check_cli_args("graph_utils.py",
            ["shortest", '{"a":{"b":1,"c":10},"b":{"c":2},"c":{}}', "a", "c"], "3"), 1.5, "correctness"),
        wcheck("graph_topo", check_cli_args("graph_utils.py",
            ["topo", '{"a":{"b":1},"b":{"c":1},"c":{}}'], "a"), 1.5, "correctness"),
        wcheck("graph_cycle_exit", check_cli_exitcode("graph_utils.py",
            ["topo", '{"a":{"b":1},"b":{"a":1}}'], 1), 1.0, "edge_cases"),
        wcheck("graph_importable", check_import_works("graph_utils.py", "graph_utils"), 0.5, "quality"),
    ],
)

# -- F1: Three Independent Modules (Parallel Speedup) --

TASK_F1 = DeepEvalTask(
    id="F1",
    category="F",
    description="Three independent utility modules — parallel speedup test",
    task_prompt=textwrap.dedent("""\
        Create three independent Python utility modules with tests:

        1. `string_utils.py` with functions:
           - reverse_string(s: str) -> str
           - is_palindrome(s: str) -> bool  (case-insensitive, ignoring spaces)
           - count_words(s: str) -> int
           Create `test_string_utils.py` with at least 4 tests.

        2. `math_utils.py` with functions:
           - factorial(n: int) -> int  (raise ValueError for negative)
           - is_prime(n: int) -> bool
           - gcd(a: int, b: int) -> int
           Create `test_math_utils.py` with at least 4 tests.

        3. `file_utils.py` with functions:
           - count_lines(path: str) -> int
           - find_files(directory: str, extension: str) -> list[str]
           - read_json(path: str) -> dict
           Create `test_file_utils.py` with at least 3 tests.

        Run all tests to verify they pass.
    """),
    setup_files={},
    expert="coder",
    checks=[
        # string_utils (correctness)
        wcheck("string_utils_exists", check_file_exists("string_utils.py"), 1.0, "correctness"),
        wcheck("string_utils_compiles", check_python_compiles("string_utils.py"), 1.5, "correctness"),
        wcheck("has_reverse_string", check_function_exists("string_utils.py", "reverse_string"), 1.0, "correctness"),
        wcheck("has_is_palindrome", check_function_exists("string_utils.py", "is_palindrome"), 1.0, "correctness"),
        wcheck("has_count_words", check_function_exists("string_utils.py", "count_words"), 1.0, "correctness"),
        wcheck("test_string_utils_pass", check_pytest_passes("test_string_utils.py"), 2.0, "correctness"),
        # math_utils (correctness)
        wcheck("math_utils_exists", check_file_exists("math_utils.py"), 1.0, "correctness"),
        wcheck("math_utils_compiles", check_python_compiles("math_utils.py"), 1.5, "correctness"),
        wcheck("has_factorial", check_function_exists("math_utils.py", "factorial"), 1.0, "correctness"),
        wcheck("has_is_prime", check_function_exists("math_utils.py", "is_prime"), 1.0, "correctness"),
        wcheck("has_gcd", check_function_exists("math_utils.py", "gcd"), 1.0, "correctness"),
        wcheck("test_math_utils_pass", check_pytest_passes("test_math_utils.py"), 2.0, "correctness"),
        # file_utils (correctness)
        wcheck("file_utils_exists", check_file_exists("file_utils.py"), 1.0, "correctness"),
        wcheck("file_utils_compiles", check_python_compiles("file_utils.py"), 1.5, "correctness"),
        wcheck("has_count_lines", check_function_exists("file_utils.py", "count_lines"), 1.0, "correctness"),
        wcheck("has_find_files", check_function_exists("file_utils.py", "find_files"), 1.0, "correctness"),
        wcheck("test_file_utils_pass", check_pytest_passes("test_file_utils.py"), 2.0, "correctness"),
    ],
)

# -- G1: Seeded Bug Hunt (Adversarial Quality) --

INVENTORY_CODE = textwrap.dedent("""\
    from dataclasses import dataclass

    @dataclass
    class Product:
        name: str
        price: float
        stock: int

    class Inventory:
        def __init__(self):
            self.products: dict[str, Product] = {}
            self.restock_threshold: int = 10

        def add_product(self, name: str, price: float, stock: int = 0):
            self.products[name] = Product(name, price, stock)

        def purchase(self, name: str, amount: int, budget: float) -> tuple[int, float]:
            \"\"\"Buy as many units as possible within budget.
            Returns (units_bought, total_cost).\"\"\"
            if name not in self.products:
                raise ValueError(f"Product '{name}' not found")
            product = self.products[name]
            if product.stock <= 0:
                return 0, 0.0
            # Bug 1: integer division truncates — should use min(amount, affordable)
            affordable = int(budget // product.price)
            can_buy = min(amount, affordable, product.stock)
            total = can_buy * product.price
            product.stock -= can_buy
            return can_buy, total

        def needs_restock(self, name: str) -> bool:
            \"\"\"Check if product stock is at or below restock threshold.\"\"\"
            if name not in self.products:
                raise ValueError(f"Product '{name}' not found")
            # Bug 2: should be <= not <  (off-by-one: exactly at threshold should trigger)
            return self.products[name].stock < self.restock_threshold

        def apply_discount(self, name: str, pct: float) -> float:
            \"\"\"Apply a percentage discount (0.0-1.0) and return new price.\"\"\"
            if name not in self.products:
                raise ValueError(f"Product '{name}' not found")
            product = self.products[name]
            # Bug 3: 1 + pct makes it a MARKUP instead of a discount
            product.price = round(product.price * (1 + pct), 2)
            return product.price

        def total_value(self) -> float:
            \"\"\"Total inventory value (sum of price * stock for all products).\"\"\"
            return sum(p.price * p.stock for p in self.products.values())
""")

TASK_G1 = DeepEvalTask(
    id="G1",
    category="G",
    description="Seeded bug hunt — find 3 subtle bugs in inventory system",
    task_prompt=textwrap.dedent("""\
        The file `inventory.py` implements an inventory management system.
        It has several subtle bugs. Your task:

        1. Read and understand the code carefully
        2. Find and fix ALL bugs
        3. Write `test_inventory.py` with comprehensive tests that:
           - Verify each bug is actually fixed
           - Test edge cases (zero price, zero stock, exact threshold, etc.)
           - Test error handling (missing products, etc.)
        4. Run the tests to confirm they pass

        Hints:
        - Look at the purchase() method: what happens with fractional results?
        - Look at needs_restock(): what happens when stock equals the threshold exactly?
        - Look at apply_discount(): does it actually discount, or does it do something else?
    """),
    setup_files={"inventory.py": INVENTORY_CODE},
    expert="coder",
    checks=[
        # Bug fixes (correctness — 3x multiplier)
        wcheck("bug1_int_div_fixed",
               check_bug_fixed("inventory.py", "budget // product.price", "budget / product.price"),
               3.0, "correctness"),
        wcheck("bug2_off_by_one_fixed",
               check_bug_fixed("inventory.py", ".stock < self.restock", ".stock <= self.restock"),
               3.0, "correctness"),
        wcheck("bug3_discount_fixed",
               check_bug_fixed("inventory.py", "1 + pct", "1 - pct"),
               3.0, "correctness"),
        wcheck("still_compiles", check_python_compiles("inventory.py"), 2.0, "correctness"),
        # Tests written (quality — 2x)
        wcheck("test_file_exists", check_file_exists("test_inventory.py"), 1.0, "quality"),
        wcheck("test_compiles", check_python_compiles("test_inventory.py"), 1.0, "quality"),
        wcheck("tests_pass", check_pytest_passes("test_inventory.py"), 2.0, "quality"),
        # Edge case coverage (edge_cases — 2x)
        wcheck("tests_purchase", check_file_contains("test_inventory.py", "purchase"), 1.0, "edge_cases"),
        wcheck("tests_restock", check_file_contains("test_inventory.py", "restock"), 1.0, "edge_cases"),
        wcheck("tests_discount", check_file_contains("test_inventory.py", "discount"), 1.0, "edge_cases"),
        wcheck("tests_edge_zero", check_file_contains("test_inventory.py", "0"), 0.5, "edge_cases"),
        # Completeness
        wcheck("dataclass_intact", check_file_contains("inventory.py", "@dataclass"), 0.5, "completeness"),
        wcheck("total_value_intact", check_function_exists("inventory.py", "total_value"), 0.5, "completeness"),
    ],
)

# -- H1: Large Multi-File Project (Scale Beyond Context) --

H1_SETUP_FILES = {
    "framework/__init__.py": "",
    "framework/app.py": textwrap.dedent("""\
        from framework.router import Router
        from framework.request import Request
        from framework.response import Response

        class App:
            def __init__(self):
                self.router = Router()
                self.middleware = []

            def add_middleware(self, mw):
                self.middleware.append(mw)

            def route(self, path, method="GET"):
                def decorator(handler):
                    self.router.add_route(path, method, handler)
                    return handler
                return decorator

            async def handle(self, request: Request) -> Response:
                # Run middleware chain
                for mw in self.middleware:
                    result = await mw(request)
                    if isinstance(result, Response):
                        return result
                # Dispatch to route handler
                handler = self.router.match(request.path, request.method)
                if handler is None:
                    return Response(status=404, body="Not Found")
                return await handler(request)
    """),
    "framework/router.py": textwrap.dedent("""\
        class Router:
            def __init__(self):
                self.routes: dict[tuple[str, str], callable] = {}

            def add_route(self, path: str, method: str, handler):
                self.routes[(path, method.upper())] = handler

            def match(self, path: str, method: str):
                return self.routes.get((path, method.upper()))
    """),
    "framework/request.py": textwrap.dedent("""\
        from dataclasses import dataclass, field

        @dataclass
        class Request:
            path: str
            method: str = "GET"
            headers: dict = field(default_factory=dict)
            body: str = ""
            user: object = None  # Set by auth middleware
    """),
    "framework/response.py": textwrap.dedent("""\
        from dataclasses import dataclass
        import json

        @dataclass
        class Response:
            status: int = 200
            body: str = ""
            headers: dict = None

            def __post_init__(self):
                if self.headers is None:
                    self.headers = {"Content-Type": "text/plain"}

            @classmethod
            def json(cls, data, status=200):
                return cls(
                    status=status,
                    body=json.dumps(data),
                    headers={"Content-Type": "application/json"}
                )
    """),
    "framework/database.py": textwrap.dedent("""\
        class InMemoryDB:
            \"\"\"Simple in-memory key-value store for demo purposes.\"\"\"
            def __init__(self):
                self._tables: dict[str, dict] = {}

            def create_table(self, name: str):
                if name not in self._tables:
                    self._tables[name] = {}

            def insert(self, table: str, key: str, data: dict):
                if table not in self._tables:
                    self.create_table(table)
                self._tables[table][key] = data

            def get(self, table: str, key: str) -> dict | None:
                return self._tables.get(table, {}).get(key)

            def find(self, table: str, **filters) -> list[dict]:
                rows = self._tables.get(table, {}).values()
                results = []
                for row in rows:
                    if all(row.get(k) == v for k, v in filters.items()):
                        results.append(row)
                return results

            def delete(self, table: str, key: str) -> bool:
                if table in self._tables and key in self._tables[table]:
                    del self._tables[table][key]
                    return True
                return False
    """),
    "framework/config.py": textwrap.dedent("""\
        class Config:
            SECRET_KEY = "change-me-in-production"
            DB_TYPE = "memory"
            SESSION_TIMEOUT = 3600
    """),
    "framework/utils.py": textwrap.dedent("""\
        import hashlib
        import time

        def hash_password(password: str, salt: str = "") -> str:
            \"\"\"Hash a password with optional salt using SHA-256.\"\"\"
            return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

        def verify_password(password: str, hashed: str, salt: str = "") -> bool:
            \"\"\"Verify a password against its hash.\"\"\"
            return hash_password(password, salt) == hashed

        def timestamp() -> float:
            return time.time()
    """),
    "main_app.py": textwrap.dedent("""\
        import asyncio
        from framework.app import App
        from framework.request import Request
        from framework.response import Response
        from framework.database import InMemoryDB

        app = App()
        db = InMemoryDB()

        @app.route("/", "GET")
        async def index(request: Request) -> Response:
            return Response.json({"message": "Welcome to the framework"})

        @app.route("/health", "GET")
        async def health(request: Request) -> Response:
            return Response.json({"status": "ok"})

        if __name__ == "__main__":
            async def demo():
                r = await app.handle(Request(path="/", method="GET"))
                print(f"Status: {r.status}, Body: {r.body}")
            asyncio.run(demo())
    """),
}

TASK_H1 = DeepEvalTask(
    id="H1",
    category="H",
    description="Add auth to multi-file web framework — scale beyond context",
    task_prompt=textwrap.dedent("""\
        This project has a mini web framework in `framework/`. Your task is to add
        authentication support:

        1. Create `framework/auth.py` with:
           - A `User` dataclass (username, password_hash, role)
           - A `register_user(db, username, password, role="user")` function
             that hashes the password and stores the user in the database
           - A `authenticate(db, username, password)` function that verifies
             credentials and returns the User or None
           - A `require_auth` async middleware function for the App that:
             * Checks for "Authorization" header (format: "Basic username:password")
             * Authenticates the user
             * Sets request.user if valid
             * Returns 401 Response if invalid/missing

        2. Add a login endpoint to `main_app.py`:
           - POST /login accepts JSON body with "username" and "password"
           - Returns 200 with user info if valid, 401 if invalid

        3. Create `test_auth.py` with tests for:
           - User registration
           - Authentication (valid + invalid credentials)
           - Middleware (authorized + unauthorized requests)
           - Login endpoint

        4. Run the tests to verify everything works together.

        Use the existing framework/utils.py hash_password and verify_password functions.
        Use the existing framework/database.py InMemoryDB for storage.
    """),
    setup_files=H1_SETUP_FILES,
    expert="coder",
    max_rounds=25,
    timeout=240,
    checks=[
        # Auth module (correctness)
        wcheck("auth_module_exists", check_file_exists("framework/auth.py"), 2.0, "correctness"),
        wcheck("auth_compiles", check_python_compiles("framework/auth.py"), 2.0, "correctness"),
        wcheck("has_user_class", check_class_exists("framework/auth.py", "User"), 1.5, "correctness"),
        wcheck("has_register_user", check_function_exists("framework/auth.py", "register_user"), 1.5, "correctness"),
        wcheck("has_authenticate", check_function_exists("framework/auth.py", "authenticate"), 1.5, "correctness"),
        wcheck("has_require_auth", check_function_exists("framework/auth.py", "require_auth"), 1.5, "correctness"),
        # Integration (correctness)
        wcheck("uses_hash_password",
               check_file_contains("framework/auth.py", "hash_password"), 1.0, "correctness"),
        wcheck("uses_verify_password",
               check_file_contains("framework/auth.py", "verify_password"), 1.0, "correctness"),
        wcheck("main_app_has_login",
               check_file_contains("main_app.py", "/login"), 1.5, "correctness"),
        # Tests (quality)
        wcheck("test_file_exists", check_file_exists("test_auth.py"), 1.0, "quality"),
        wcheck("test_compiles", check_python_compiles("test_auth.py"), 1.0, "quality"),
        wcheck("tests_pass", check_pytest_passes("test_auth.py"), 3.0, "quality"),
        # Edge cases
        wcheck("tests_invalid_creds",
               check_file_contains("test_auth.py", "invalid"), 1.0, "edge_cases"),
        wcheck("tests_unauthorized",
               check_file_contains("test_auth.py", "401"), 1.0, "edge_cases"),
        wcheck("password_not_plain",
               check_file_not_contains("framework/auth.py", "password_plain"), 0.5, "edge_cases"),
        # Completeness
        wcheck("framework_init_intact",
               check_file_exists("framework/__init__.py"), 0.5, "completeness"),
        wcheck("app_still_compiles",
               check_python_compiles("framework/app.py"), 1.0, "completeness"),
        wcheck("main_still_compiles",
               check_python_compiles("main_app.py"), 1.0, "completeness"),
    ],
)

# -- I1: Learning Eval (Learning Over Time) --

PAGINATE_CODE = textwrap.dedent("""\
    def paginate(items: list, page: int, per_page: int = 10) -> dict:
        \"\"\"Return a page of items with pagination metadata.

        Args:
            items: Full list of items
            page: Page number (1-based)
            per_page: Items per page (default 10)

        Returns:
            dict with keys: items, page, per_page, total, total_pages
        \"\"\"
        total = len(items)
        total_pages = (total + per_page - 1) // per_page

        # Bug: off-by-one in start index (page 1 should start at index 0)
        start = page * per_page  # Should be (page - 1) * per_page
        end = start + per_page

        return {
            "items": items[start:end],
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        }


    def search(items: list[dict], field: str, query: str) -> list[dict]:
        \"\"\"Search items where field contains query (case-insensitive).\"\"\"
        return [item for item in items if query.lower() in str(item.get(field, "")).lower()]
""")

# The "similar but different" bug for Run 2
PAGINATE_CODE_V2 = textwrap.dedent("""\
    def paginate_v2(items: list, page: int, per_page: int = 10) -> dict:
        \"\"\"Return a page of items with pagination metadata.

        Args:
            items: Full list of items
            page: Page number (1-based)
            per_page: Items per page (default 10)

        Returns:
            dict with keys: items, page, per_page, total, total_pages
        \"\"\"
        total = len(items)
        # Bug: off-by-one in total_pages calculation (fails when total is exact multiple)
        total_pages = total // per_page  # Should be (total + per_page - 1) // per_page

        start = (page - 1) * per_page
        end = start + per_page

        return {
            "items": items[start:end],
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        }


    def filter_items(items: list[dict], **criteria) -> list[dict]:
        \"\"\"Filter items matching all criteria.\"\"\"
        return [item for item in items
                if all(item.get(k) == v for k, v in criteria.items())]
""")

TASK_I1 = DeepEvalTask(
    id="I1",
    category="I",
    description="Learning eval — does LessonsDB improve second attempt?",
    task_prompt=textwrap.dedent("""\
        The file `data.py` has a bug in the `paginate()` function.
        Page numbers are 1-based, but the function produces wrong results
        for page 1. Fix the bug and add tests in `test_data.py` that verify:
        - Page 1 returns the first `per_page` items
        - Page 2 returns the next batch
        - Empty list returns empty items
        - Total pages calculation is correct
        Run the tests.
    """),
    setup_files={"data.py": PAGINATE_CODE},
    expert="coder",
    use_swarm=False,  # Learning eval compares run1 vs run2 of single agent
    learning_seed={
        "error_sig": "off-by-one in pagination start index",
        "fix": "Changed 'start = page * per_page' to 'start = (page - 1) * per_page' — page is 1-based so subtract 1 for 0-based index",
        "files": ["data.py"],
        "expert": "coder",
    },
    checks=[
        # Fix (correctness)
        wcheck("bug_fixed",
               check_bug_fixed("data.py", "start = page * per_page", "start = (page - 1) * per_page"),
               3.0, "correctness"),
        wcheck("still_compiles", check_python_compiles("data.py"), 1.5, "correctness"),
        wcheck("search_intact", check_function_exists("data.py", "search"), 1.0, "correctness"),
        # Tests (quality)
        wcheck("test_file_exists", check_file_exists("test_data.py"), 1.0, "quality"),
        wcheck("test_compiles", check_python_compiles("test_data.py"), 1.0, "quality"),
        wcheck("tests_pass", check_pytest_passes("test_data.py"), 2.0, "quality"),
        # Edge case tests
        wcheck("tests_page_1", check_file_contains("test_data.py", "page"), 1.0, "edge_cases"),
        wcheck("tests_empty", check_file_contains("test_data.py", "[]"), 0.5, "edge_cases"),
    ],
)

# I1 checks for Run 2 (the similar bug)
I1_RUN2_CHECKS = [
    wcheck("bug_fixed_v2",
           check_bug_fixed("data.py", "total // per_page", "(total + per_page - 1) // per_page"),
           3.0, "correctness"),
    wcheck("still_compiles", check_python_compiles("data.py"), 1.5, "correctness"),
    wcheck("filter_intact", check_function_exists("data.py", "filter_items"), 1.0, "correctness"),
    wcheck("test_file_exists", check_file_exists("test_data.py"), 1.0, "quality"),
    wcheck("test_compiles", check_python_compiles("test_data.py"), 1.0, "quality"),
    wcheck("tests_pass", check_pytest_passes("test_data.py"), 2.0, "quality"),
    wcheck("tests_total_pages", check_file_contains("test_data.py", "total_pages"), 1.0, "edge_cases"),
]

# All deep eval tasks
DEEP_EVAL_TASKS: list[DeepEvalTask] = [TASK_E1, TASK_E2, TASK_E3, TASK_F1, TASK_G1, TASK_H1, TASK_I1]


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _setup_deep_workspace(task: DeepEvalTask, workspace: Path):
    """Create the workspace directory and seed files."""
    workspace.mkdir(parents=True, exist_ok=True)
    for rel_path, content in task.setup_files.items():
        f = workspace / rel_path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")


async def _run_single_agent(task: DeepEvalTask, workspace: Path) -> RunMetrics:
    """Run a task through a single agent and return metrics."""
    _setup_deep_workspace(task, workspace)

    old_project_dir = shared.PROJECT_DIR
    shared.PROJECT_DIR = workspace
    bus = SwarmBus(":memory:")

    t0 = time.monotonic()
    tokens_before = shared.state.global_tokens_used
    cost_before = shared.state.global_cost_usd
    shared._last_deliberation_time = 0.0
    agent_name = f"deep_single_{task.id}"
    error = ""

    try:
        await run_expert(
            task.expert,
            task.task_prompt,
            bus=bus,
            agent_name=agent_name,
        )
    except Exception as e:
        error = str(e)
    finally:
        elapsed = round(time.monotonic() - t0, 2)
        tokens = shared.state.global_tokens_used - tokens_before
        cost = round(shared.state.global_cost_usd - cost_before, 6)

        # Wait for background agents
        for name, bg_task in list(shared._background_tasks.items()):
            if not bg_task.done():
                try:
                    await asyncio.wait_for(bg_task, timeout=60)
                except Exception:
                    bg_task.cancel()
        shared._background_tasks.clear()

        shared.PROJECT_DIR = old_project_dir
        bus.close()
        if shared._bus_instance is not None:
            try:
                shared._bus_instance.close()
            except Exception:
                pass
            shared._bus_instance = None

        # Clean up agents
        agents_to_remove = [n for n in shared.state.agents if n.startswith("deep_")]
        for n in agents_to_remove:
            shared.state.agents.pop(n, None)

    # Get rounds
    rounds = 0
    agent = shared.state.get_agent(agent_name)
    if agent:
        rounds = len(agent.tool_call_log)

    # Run weighted checks
    metrics = _run_weighted_checks(task.checks, workspace)
    metrics.time_seconds = elapsed
    metrics.deliberation_time = shared._last_deliberation_time
    metrics.cost_usd = cost
    metrics.tokens_used = tokens
    metrics.rounds_used = rounds
    metrics.error = error
    return metrics


async def _run_swarm(task: DeepEvalTask, workspace: Path) -> RunMetrics:
    """Run a task through the Orchestrator and return metrics."""
    _setup_deep_workspace(task, workspace)

    old_project_dir = shared.PROJECT_DIR
    shared.PROJECT_DIR = workspace
    bus = SwarmBus(":memory:")

    t0 = time.monotonic()
    tokens_before = shared.state.global_tokens_used
    cost_before = shared.state.global_cost_usd
    shared._last_planning_time = 0.0
    shared._last_deliberation_time = 0.0
    error = ""

    try:
        await Orchestrator.run(task.task_prompt, bus, use_worktrees=False)
    except Exception as e:
        error = str(e)
    finally:
        elapsed = round(time.monotonic() - t0, 2)
        planning_time = shared._last_planning_time
        tokens = shared.state.global_tokens_used - tokens_before
        cost = round(shared.state.global_cost_usd - cost_before, 6)

        # Wait for background agents
        for name, bg_task in list(shared._background_tasks.items()):
            if not bg_task.done():
                try:
                    await asyncio.wait_for(bg_task, timeout=60)
                except Exception:
                    bg_task.cancel()
        shared._background_tasks.clear()

        shared.PROJECT_DIR = old_project_dir
        bus.close()
        if shared._bus_instance is not None:
            try:
                shared._bus_instance.close()
            except Exception:
                pass
            shared._bus_instance = None

        # Clean up agents
        agents_to_remove = [n for n in shared.state.agents if n.startswith(("deep_", "orch_"))]
        for n in agents_to_remove:
            shared.state.agents.pop(n, None)

    # Run weighted checks
    metrics = _run_weighted_checks(task.checks, workspace)
    metrics.time_seconds = elapsed
    metrics.planning_time = planning_time
    metrics.deliberation_time = shared._last_deliberation_time
    metrics.cost_usd = cost
    metrics.tokens_used = tokens
    metrics.error = error
    return metrics


# ---------------------------------------------------------------------------
# Comparative runner
# ---------------------------------------------------------------------------

async def run_comparative(task: DeepEvalTask) -> ComparativeResult:
    """Run a task through both single-agent and swarm, compare results."""
    result = ComparativeResult(
        task_id=task.id,
        category=task.category,
        description=task.description,
    )

    with tempfile.TemporaryDirectory(prefix=f"deep_single_{task.id}_") as tmp1, \
         tempfile.TemporaryDirectory(prefix=f"deep_swarm_{task.id}_") as tmp2:

        ws_single = Path(tmp1)
        ws_swarm = Path(tmp2)

        # Run single-agent path
        print(f"  [Single Agent] Running {task.id}...")
        result.single = await _run_single_agent(task, ws_single)
        s_exec = result.single.time_seconds - result.single.deliberation_time
        print(f"  [Single Agent] Score: {result.single.quality_score:.2%} | "
              f"Exec: {s_exec:.1f}s | Delib: {result.single.deliberation_time:.1f}s | "
              f"Cost: ${result.single.cost_usd:.4f}")

        if task.use_swarm:
            # Run swarm path
            print(f"  [Swarm]        Running {task.id}...")
            result.swarm = await _run_swarm(task, ws_swarm)
            w_exec = result.swarm.time_seconds - result.swarm.deliberation_time
            print(f"  [Swarm]        Score: {result.swarm.quality_score:.2%} | "
                  f"Exec: {w_exec:.1f}s | Delib: {result.swarm.deliberation_time:.1f}s | "
                  f"Cost: ${result.swarm.cost_usd:.4f}")
        else:
            # No swarm comparison (e.g., learning eval)
            result.swarm = RunMetrics()

    result.verdict, result.quality_delta, result.speedup, result.cost_ratio = \
        _compute_verdict(result.single, result.swarm)
    _compute_efficiency_scores(result)

    return result


async def run_learning_eval(task: DeepEvalTask) -> ComparativeResult:
    """Run a learning evaluation: Run 1 fresh, Run 2 with seeded LessonsDB.

    Returns ComparativeResult where single=Run1(fresh) and swarm=Run2(with lessons).
    """
    result = ComparativeResult(
        task_id=task.id,
        category=task.category,
        description=f"{task.description} (learning comparison)",
    )

    with tempfile.TemporaryDirectory(prefix=f"deep_learn1_{task.id}_") as tmp1, \
         tempfile.TemporaryDirectory(prefix=f"deep_learn2_{task.id}_") as tmp2:

        ws1 = Path(tmp1)
        ws2 = Path(tmp2)

        # Run 1: Fresh (no lessons)
        print(f"  [Run 1: Fresh] Running {task.id}...")
        result.single = await _run_single_agent(task, ws1)
        r1_exec = result.single.time_seconds - result.single.deliberation_time
        print(f"  [Run 1: Fresh] Score: {result.single.quality_score:.2%} | "
              f"Exec: {r1_exec:.1f}s | Delib: {result.single.deliberation_time:.1f}s | "
              f"Cost: ${result.single.cost_usd:.4f}")

        # Seed LessonsDB for Run 2
        if task.learning_seed:
            lessons_dir = ws2 / ".grokswarm"
            lessons_dir.mkdir(parents=True, exist_ok=True)
            lessons_db = LessonsDB(lessons_dir / "lessons_learned.yaml")
            lessons_db.record_lesson(
                error_signature=task.learning_seed["error_sig"],
                fix_description=task.learning_seed["fix"],
                files_involved=task.learning_seed.get("files", []),
                expert=task.learning_seed.get("expert", ""),
            )

        # Run 2: With lessons, similar bug (use modified task)
        run2_task = DeepEvalTask(
            id=f"{task.id}_run2",
            category=task.category,
            description=task.description,
            task_prompt=textwrap.dedent("""\
                The file `data.py` has a bug in the `paginate_v2()` function.
                The total_pages calculation is wrong — it fails when the total
                number of items is an exact multiple of per_page.
                Fix the bug and write tests in `test_data.py` covering:
                - Exact multiples (e.g., 20 items, 10 per page = 2 pages)
                - Non-exact (e.g., 21 items, 10 per page = 3 pages)
                - Empty list
                - Single page
                Run the tests.
            """),
            setup_files={"data.py": PAGINATE_CODE_V2},
            expert=task.expert,
            checks=I1_RUN2_CHECKS,
        )

        print(f"  [Run 2: With Lessons] Running {task.id}...")
        result.swarm = await _run_single_agent(run2_task, ws2)
        r2_exec = result.swarm.time_seconds - result.swarm.deliberation_time
        print(f"  [Run 2: With Lessons] Score: {result.swarm.quality_score:.2%} | "
              f"Exec: {r2_exec:.1f}s | Delib: {result.swarm.deliberation_time:.1f}s | "
              f"Cost: ${result.swarm.cost_usd:.4f}")

    result.verdict, result.quality_delta, result.speedup, result.cost_ratio = \
        _compute_verdict(result.single, result.swarm)
    _compute_efficiency_scores(result)

    return result


# ---------------------------------------------------------------------------
# Statistical runner (N runs)
# ---------------------------------------------------------------------------

async def run_statistical(task: DeepEvalTask, n_runs: int = 3) -> StatisticalResult:
    """Run a task N times through both paths, aggregate statistics."""
    stat = StatisticalResult(task_id=task.id, n_runs=n_runs)

    for i in range(n_runs):
        print(f"  [Run {i+1}/{n_runs}] {task.id}...")
        if task.category == "I":
            result = await run_learning_eval(task)
        elif task.use_swarm:
            result = await run_comparative(task)
        else:
            with tempfile.TemporaryDirectory(prefix=f"stat_{task.id}_{i}_") as tmp:
                ws = Path(tmp)
                single_metrics = await _run_single_agent(task, ws)
                result = ComparativeResult(
                    task_id=task.id, category=task.category,
                    description=task.description, single=single_metrics,
                )

        stat.single_scores.append(result.single.quality_score)
        stat.single_times.append(result.single.time_seconds)
        stat.single_costs.append(result.single.cost_usd)

        if result.swarm.checks_total > 0:
            stat.swarm_scores.append(result.swarm.quality_score)
            stat.swarm_times.append(result.swarm.time_seconds)
            stat.swarm_costs.append(result.swarm.cost_usd)

    # Compute statistics
    stat.single_mean = round(_mean(stat.single_scores), 4)
    stat.single_stddev = round(_stddev(stat.single_scores), 4)
    stat.swarm_mean = round(_mean(stat.swarm_scores), 4)
    stat.swarm_stddev = round(_stddev(stat.swarm_scores), 4)

    # Significance: delta must exceed max of both stddevs
    delta = stat.swarm_mean - stat.single_mean
    threshold = max(stat.single_stddev, stat.swarm_stddev, 0.05)
    stat.significant = abs(delta) > threshold

    if stat.significant and delta > 0:
        stat.verdict = "swarm_better"
    elif stat.significant and delta < 0:
        stat.verdict = "single_better"
    else:
        stat.verdict = "inconclusive"

    return stat


def format_statistical_report(results: list[StatisticalResult]) -> str:
    """Format statistical results from N-run evaluation."""
    lines = []
    lines.append("=" * 110)
    lines.append("GROKSWARM STATISTICAL EVAL REPORT")
    lines.append("=" * 110)
    lines.append("")
    lines.append(f"{'Task':<8} {'N':>3} {'S.Mean':>8} {'S.Std':>7} "
                 f"{'W.Mean':>8} {'W.Std':>7} {'Delta':>8} "
                 f"{'Signif':>7} {'Verdict':<16}")
    lines.append("-" * 110)

    for r in results:
        s_mean = f"{r.single_mean:.0%}"
        s_std = f"{r.single_stddev:.3f}"
        if r.swarm_scores:
            w_mean = f"{r.swarm_mean:.0%}"
            w_std = f"{r.swarm_stddev:.3f}"
            delta = f"{r.swarm_mean - r.single_mean:+.3f}"
        else:
            w_mean = w_std = delta = "N/A"
        sig = "Yes" if r.significant else "No"
        verdict_display = r.verdict.replace("_", " ").title()
        lines.append(f"{r.task_id:<8} {r.n_runs:>3} {s_mean:>8} {s_std:>7} "
                     f"{w_mean:>8} {w_std:>7} {delta:>8} "
                     f"{sig:>7} {verdict_display:<16}")

        # Per-run details
        for i, score in enumerate(r.single_scores):
            swarm_score = r.swarm_scores[i] if i < len(r.swarm_scores) else None
            s_time = r.single_times[i] if i < len(r.single_times) else 0
            s_cost = r.single_costs[i] if i < len(r.single_costs) else 0
            detail = f"    Run {i+1}: Single={score:.2%} ({s_time:.1f}s, ${s_cost:.4f})"
            if swarm_score is not None:
                w_time = r.swarm_times[i] if i < len(r.swarm_times) else 0
                w_cost = r.swarm_costs[i] if i < len(r.swarm_costs) else 0
                detail += f"  Swarm={swarm_score:.2%} ({w_time:.1f}s, ${w_cost:.4f})"
            lines.append(detail)

    lines.append("-" * 110)
    lines.append("")
    lines.append("=" * 110)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_deep_report(results: list[ComparativeResult]) -> str:
    """Format comparative results as a readable report."""
    lines = []
    lines.append("=" * 100)
    lines.append("GROKSWARM DEEP COMPARATIVE EVAL REPORT")
    lines.append("=" * 100)
    lines.append("")

    # Header
    lines.append(f"{'Task':<8} {'Single':>8} {'Swarm':>8} {'Delta':>8} "
                 f"{'S.Time':>7} {'W.Total':>7} {'W.Plan':>7} {'W.Exec':>7} "
                 f"{'Speedup':>8} {'ExecSpd':>8} {'CostRat':>8} {'Verdict':<14}")
    lines.append("-" * 120)

    swarm_wins = single_wins = ties = 0
    total_delta = 0.0
    total_speedup = 0.0
    count = 0

    for r in results:
        single_str = f"{r.single.quality_score:.0%}"
        swarm_str = f"{r.swarm.quality_score:.0%}" if r.swarm.checks_total > 0 else "N/A"
        delta_str = f"{r.quality_delta:+.2f}" if r.swarm.checks_total > 0 else "N/A"
        s_time = f"{r.single.time_seconds:.1f}s"
        w_time = f"{r.swarm.time_seconds:.1f}s" if r.swarm.checks_total > 0 else "N/A"

        # Plan/exec breakdown
        plan_t = r.swarm.planning_time
        exec_t = max(0, r.swarm.time_seconds - plan_t) if r.swarm.checks_total > 0 else 0
        w_plan = f"{plan_t:.1f}s" if r.swarm.checks_total > 0 else "N/A"
        w_exec = f"{exec_t:.1f}s" if r.swarm.checks_total > 0 else "N/A"

        speedup_str = f"{r.speedup:.2f}x" if r.swarm.checks_total > 0 else "N/A"
        # Exec-only speedup: single time / swarm exec time
        if r.swarm.checks_total > 0 and exec_t > 0:
            exec_speedup = r.single.time_seconds / exec_t
            exec_spd_str = f"{exec_speedup:.2f}x"
        else:
            exec_spd_str = "N/A"
        cost_str = f"{r.cost_ratio:.2f}x" if r.swarm.checks_total > 0 else "N/A"

        verdict_display = r.verdict.replace("_", " ").title()
        lines.append(f"{r.task_id:<8} {single_str:>8} {swarm_str:>8} {delta_str:>8} "
                     f"{s_time:>7} {w_time:>7} {w_plan:>7} {w_exec:>7} "
                     f"{speedup_str:>8} {exec_spd_str:>8} {cost_str:>8} {verdict_display:<14}")

        if r.verdict == "swarm_better":
            swarm_wins += 1
        elif r.verdict == "single_better":
            single_wins += 1
        else:
            ties += 1

        total_delta += r.quality_delta
        total_speedup += r.speedup
        count += 1

    lines.append("-" * 120)

    # Efficiency scores table
    lines.append("")
    lines.append("EFFICIENCY SCORES (quality + actual cost/time, overall normalized)")
    lines.append("-" * 100)
    lines.append(f"{'Task':<8} {'S.Qual':>7} {'S.Cost':>8} {'S.Time':>8} {'S.Ovrll':>8}"
                 f" | {'W.Qual':>7} {'W.Cost':>8} {'W.Time':>8} {'W.Ovrll':>8}")
    lines.append("-" * 100)

    total_single_overall = 0.0
    total_swarm_overall = 0.0
    swarm_count = 0

    for r in results:
        sq = f"{r.single.quality_score:.0%}"
        sc = f"${r.single.cost_usd:.3f}"
        st = f"{r.single.time_seconds:.1f}s"
        so = f"{r.single_overall:.0%}"
        if r.swarm.checks_total > 0:
            wq = f"{r.swarm.quality_score:.0%}"
            wc = f"${r.swarm.cost_usd:.3f}"
            wt = f"{r.swarm.time_seconds:.1f}s"
            wo = f"{r.swarm_overall:.0%}"
            total_swarm_overall += r.swarm_overall
            swarm_count += 1
        else:
            wq = wc = wt = wo = "N/A"
        total_single_overall += r.single_overall
        lines.append(f"{r.task_id:<8} {sq:>7} {sc:>8} {st:>8} {so:>8}"
                     f" | {wq:>7} {wc:>8} {wt:>8} {wo:>8}")

    lines.append("-" * 100)

    # Summary
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Swarm Wins:    {swarm_wins}")
    lines.append(f"  Single Wins:   {single_wins}")
    lines.append(f"  Ties:          {ties}")
    if count > 0:
        lines.append(f"  Avg Quality Δ: {total_delta / count:+.3f}")
        lines.append(f"  Avg Speedup:   {total_speedup / count:.2f}x")
        lines.append(f"  Avg S.Overall: {total_single_overall / count:.0%}")
        if swarm_count > 0:
            lines.append(f"  Avg W.Overall: {total_swarm_overall / swarm_count:.0%}")

    # Category breakdowns
    lines.append("")
    lines.append("CATEGORY BREAKDOWN")
    lines.append("-" * 40)
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    cat_names = {
        "E": "Comparative",
        "F": "Parallel Speedup",
        "G": "Adversarial Quality",
        "H": "Scale Beyond Context",
        "I": "Learning Over Time",
    }
    for cat, cat_results in sorted(categories.items()):
        name = cat_names.get(cat, cat)
        avg_single = sum(r.single.quality_score for r in cat_results) / len(cat_results)
        has_swarm = any(r.swarm.checks_total > 0 for r in cat_results)
        if has_swarm:
            avg_swarm = sum(r.swarm.quality_score for r in cat_results) / len(cat_results)
            lines.append(f"  {cat} ({name}): Single={avg_single:.0%} Swarm={avg_swarm:.0%}")
        else:
            lines.append(f"  {cat} ({name}): Run1={avg_single:.0%}")

    # Per-task check details
    lines.append("")
    lines.append("CHECK DETAILS")
    lines.append("-" * 40)
    for r in results:
        lines.append(f"\n  {r.task_id} — {r.description}")
        if r.single.check_details:
            s_exec = r.single.time_seconds - r.single.deliberation_time
            lines.append(f"    [Single Agent] Score: {r.single.quality_score:.0%} | "
                        f"Exec: {s_exec:.1f}s | Delib: {r.single.deliberation_time:.1f}s | "
                        f"Cost: ${r.single.cost_usd:.4f}")
            for d in r.single.check_details:
                status = "PASS" if d["passed"] else "FAIL"
                lines.append(f"      [{status}] {d['check']} ({d['category']} w={d['weight']:.1f}): "
                            f"{d['message'][:60]}")
            if r.single.error:
                lines.append(f"      ERROR: {r.single.error[:80]}")
        if r.swarm.check_details:
            w_exec = r.swarm.time_seconds - r.swarm.deliberation_time
            lines.append(f"    [Swarm] Score: {r.swarm.quality_score:.0%} | "
                        f"Exec: {w_exec:.1f}s | Delib: {r.swarm.deliberation_time:.1f}s | "
                        f"Cost: ${r.swarm.cost_usd:.4f}")
            for d in r.swarm.check_details:
                status = "PASS" if d["passed"] else "FAIL"
                lines.append(f"      [{status}] {d['check']} ({d['category']} w={d['weight']:.1f}): "
                            f"{d['message'][:60]}")
            if r.swarm.error:
                lines.append(f"      ERROR: {r.swarm.error[:80]}")

    lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pytest unit tests
# ---------------------------------------------------------------------------

class TestDataStructures:
    """Verify data structures are well-formed."""

    def test_weighted_check_defaults(self):
        wc = WeightedCheck(name="test", check_fn=lambda w: (True, "ok"))
        assert wc.weight == 1.0
        assert wc.category == "correctness"

    def test_deep_eval_task_defaults(self):
        t = DeepEvalTask(id="X1", category="X", description="test", task_prompt="do it",
                        setup_files={})
        assert t.expert == "coder"
        assert t.max_rounds == 20
        assert t.timeout == 180
        assert t.use_swarm is True
        assert t.learning_seed is None

    def test_run_metrics_defaults(self):
        m = RunMetrics()
        assert m.quality_score == 0.0
        assert m.checks_passed == 0

    def test_comparative_result_defaults(self):
        r = ComparativeResult(task_id="X1")
        assert r.verdict == "tie"
        assert r.quality_delta == 0.0
        assert r.speedup == 1.0

    def test_statistical_result_defaults(self):
        s = StatisticalResult(task_id="X1")
        assert s.n_runs == 0
        assert s.significant is False
        assert s.verdict == "inconclusive"

    def test_deep_eval_task_has_baseline_scores(self):
        t = DeepEvalTask(id="X1", category="X", description="test", task_prompt="do it",
                        setup_files={}, baseline_scores={"claude_code": 0.85})
        assert t.baseline_scores["claude_code"] == 0.85

    def test_all_tasks_have_checks(self):
        for task in DEEP_EVAL_TASKS:
            assert len(task.checks) > 0, f"Task {task.id} has no checks"

    def test_all_tasks_have_unique_ids(self):
        ids = [t.id for t in DEEP_EVAL_TASKS]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs: {ids}"

    def test_all_tasks_have_valid_category(self):
        for task in DEEP_EVAL_TASKS:
            assert task.category in ("E", "F", "G", "H", "I"), f"Bad category: {task.category}"

    def test_all_checks_have_valid_category(self):
        for task in DEEP_EVAL_TASKS:
            for wc in task.checks:
                assert wc.category in CATEGORY_MULTIPLIERS, \
                    f"Task {task.id}, check {wc.name}: invalid category '{wc.category}'"


class TestWeightedScoring:
    """Verify weighted scoring math."""

    def test_all_pass(self):
        checks = [
            WeightedCheck("a", lambda w: (True, "ok"), 1.0, "correctness"),
            WeightedCheck("b", lambda w: (True, "ok"), 1.0, "quality"),
        ]
        metrics = _run_weighted_checks(checks, Path("."))
        assert metrics.quality_score == 1.0
        assert metrics.checks_passed == 2
        assert metrics.checks_total == 2

    def test_all_fail(self):
        checks = [
            WeightedCheck("a", lambda w: (False, "bad"), 1.0, "correctness"),
            WeightedCheck("b", lambda w: (False, "bad"), 1.0, "quality"),
        ]
        metrics = _run_weighted_checks(checks, Path("."))
        assert metrics.quality_score == 0.0
        assert metrics.checks_passed == 0

    def test_weighted_scoring_correctness_heavier(self):
        """Correctness checks (3x) should outweigh completeness (1x)."""
        checks = [
            WeightedCheck("pass_completeness", lambda w: (True, "ok"), 1.0, "completeness"),
            WeightedCheck("fail_correctness", lambda w: (False, "bad"), 1.0, "correctness"),
        ]
        metrics = _run_weighted_checks(checks, Path("."))
        # completeness passes: 1.0 * 1.0 = 1.0 earned
        # correctness fails: 1.0 * 3.0 = 3.0 total but 0 earned
        # total_weight = 1.0 + 3.0 = 4.0, earned = 1.0
        assert metrics.quality_score == pytest.approx(0.25)

    def test_weighted_scoring_all_correctness(self):
        """Two correctness checks, one passes."""
        checks = [
            WeightedCheck("a", lambda w: (True, "ok"), 2.0, "correctness"),
            WeightedCheck("b", lambda w: (False, "bad"), 1.0, "correctness"),
        ]
        metrics = _run_weighted_checks(checks, Path("."))
        # a: 2.0 * 3.0 = 6.0 earned
        # b: 1.0 * 3.0 = 3.0 not earned
        # score = 6.0 / 9.0 = 0.6667
        assert metrics.quality_score == pytest.approx(6.0 / 9.0, abs=0.001)

    def test_empty_checks(self):
        metrics = _run_weighted_checks([], Path("."))
        assert metrics.quality_score == 0.0
        assert metrics.checks_total == 0

    def test_check_exception_handled(self):
        def bad_check(w):
            raise RuntimeError("boom")
        checks = [WeightedCheck("explode", bad_check, 1.0, "correctness")]
        metrics = _run_weighted_checks(checks, Path("."))
        assert metrics.quality_score == 0.0
        assert metrics.check_details[0]["passed"] is False
        assert "boom" in metrics.check_details[0]["message"]


class TestComputeVerdict:
    """Verify verdict computation."""

    def test_swarm_better(self):
        single = RunMetrics(quality_score=0.5, time_seconds=10, cost_usd=0.01)
        swarm = RunMetrics(quality_score=0.8, time_seconds=5, cost_usd=0.03)
        verdict, delta, speedup, cost_ratio = _compute_verdict(single, swarm)
        assert verdict == "swarm_better"
        assert delta == pytest.approx(0.3)
        assert speedup == 2.0
        assert cost_ratio == 3.0

    def test_single_better(self):
        single = RunMetrics(quality_score=0.9, time_seconds=10, cost_usd=0.01)
        swarm = RunMetrics(quality_score=0.5, time_seconds=15, cost_usd=0.02)
        verdict, delta, speedup, cost_ratio = _compute_verdict(single, swarm)
        assert verdict == "single_better"
        assert delta == pytest.approx(-0.4)

    def test_tie(self):
        single = RunMetrics(quality_score=0.75, time_seconds=10, cost_usd=0.01)
        swarm = RunMetrics(quality_score=0.78, time_seconds=10, cost_usd=0.01)
        verdict, delta, speedup, cost_ratio = _compute_verdict(single, swarm)
        assert verdict == "tie"

    def test_zero_time_handling(self):
        single = RunMetrics(quality_score=0.5, time_seconds=0, cost_usd=0)
        swarm = RunMetrics(quality_score=0.5, time_seconds=0, cost_usd=0)
        verdict, delta, speedup, cost_ratio = _compute_verdict(single, swarm)
        assert speedup == 1.0
        assert cost_ratio == 1.0


class TestCostAdjustedVerdict:
    """Verify cost-adjusted verdict computation."""

    def test_swarm_efficient(self):
        single = RunMetrics(quality_score=0.6, time_seconds=10, cost_usd=0.02)
        swarm = RunMetrics(quality_score=0.9, time_seconds=8, cost_usd=0.03)
        verdict, delta, speedup, cost_ratio = _compute_verdict_v2(single, swarm)
        assert verdict == "swarm_efficient"

    def test_swarm_better_but_costly(self):
        single = RunMetrics(quality_score=0.6, time_seconds=10, cost_usd=0.01)
        swarm = RunMetrics(quality_score=0.8, time_seconds=8, cost_usd=0.10)
        verdict, delta, speedup, cost_ratio = _compute_verdict_v2(single, swarm)
        assert verdict == "swarm_better_but_costly"

    def test_single_efficient(self):
        single = RunMetrics(quality_score=0.9, time_seconds=10, cost_usd=0.01)
        swarm = RunMetrics(quality_score=0.5, time_seconds=15, cost_usd=0.05)
        verdict, delta, speedup, cost_ratio = _compute_verdict_v2(single, swarm)
        assert verdict == "single_efficient"

    def test_tie(self):
        single = RunMetrics(quality_score=0.75, time_seconds=10, cost_usd=0.01)
        swarm = RunMetrics(quality_score=0.78, time_seconds=10, cost_usd=0.01)
        verdict, delta, speedup, cost_ratio = _compute_verdict_v2(single, swarm)
        assert verdict == "tie"


class TestStatisticalHelpers:
    """Verify mean, stddev, efficiency helpers."""

    def test_mean_basic(self):
        assert _mean([1, 2, 3, 4, 5]) == 3.0

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_stddev_basic(self):
        result = _stddev([2, 4, 4, 4, 5, 5, 7, 9])
        assert 2.0 < result < 2.2  # sample stddev ~2.14

    def test_stddev_single(self):
        assert _stddev([5]) == 0.0

    def test_stddev_empty(self):
        assert _stddev([]) == 0.0

    def test_efficiency_basic(self):
        assert _compute_efficiency(0.9, 0.01) == pytest.approx(90.0)

    def test_efficiency_zero_cost(self):
        result = _compute_efficiency(0.9, 0.0)
        assert result == pytest.approx(900.0)  # 0.9 / 0.001


class TestStatisticalReport:
    """Verify statistical report formatting."""

    def test_format_empty(self):
        report = format_statistical_report([])
        assert "STATISTICAL EVAL REPORT" in report

    def test_format_with_results(self):
        stat = StatisticalResult(
            task_id="E1", n_runs=3,
            single_scores=[0.8, 0.85, 0.82],
            swarm_scores=[0.9, 0.88, 0.91],
            single_mean=0.8233, single_stddev=0.025,
            swarm_mean=0.8967, swarm_stddev=0.015,
            single_times=[10, 12, 11], swarm_times=[8, 7, 9],
            single_costs=[0.01, 0.01, 0.01], swarm_costs=[0.03, 0.03, 0.03],
            significant=True, verdict="swarm_better",
        )
        report = format_statistical_report([stat])
        assert "E1" in report
        assert "Swarm Better" in report
        assert "Run 1" in report


class TestCheckFunctionsDeep:
    """Verify the new check function builders."""

    def test_check_cli_args_pass(self, tmp_path):
        (tmp_path / "hello.py").write_text('import sys; print(f"Hello {sys.argv[1]}")')
        ok, msg = check_cli_args("hello.py", ["World"], "Hello World")(tmp_path)
        assert ok is True

    def test_check_cli_args_fail(self, tmp_path):
        (tmp_path / "hello.py").write_text('print("Goodbye")')
        ok, msg = check_cli_args("hello.py", [], "Hello")(tmp_path)
        assert ok is False

    def test_check_cli_exitcode_pass(self, tmp_path):
        (tmp_path / "exit1.py").write_text("import sys; sys.exit(1)")
        ok, msg = check_cli_exitcode("exit1.py", [], 1)(tmp_path)
        assert ok is True

    def test_check_cli_exitcode_fail(self, tmp_path):
        (tmp_path / "exit0.py").write_text("pass")
        ok, msg = check_cli_exitcode("exit0.py", [], 1)(tmp_path)
        assert ok is False

    def test_check_bug_fixed_pass(self, tmp_path):
        (tmp_path / "code.py").write_text("x = a - b")
        ok, msg = check_bug_fixed("code.py", "a + b", "a - b")(tmp_path)
        assert ok is True

    def test_check_bug_fixed_still_buggy(self, tmp_path):
        (tmp_path / "code.py").write_text("x = a + b")
        ok, msg = check_bug_fixed("code.py", "a + b", "a - b")(tmp_path)
        assert ok is False

    def test_check_class_exists_pass(self, tmp_path):
        (tmp_path / "m.py").write_text("class Foo:\n    pass")
        ok, msg = check_class_exists("m.py", "Foo")(tmp_path)
        assert ok is True

    def test_check_class_exists_fail(self, tmp_path):
        (tmp_path / "m.py").write_text("x = 1")
        ok, msg = check_class_exists("m.py", "Foo")(tmp_path)
        assert ok is False

    def test_check_import_works_pass(self, tmp_path):
        (tmp_path / "simple.py").write_text("x = 42")
        ok, msg = check_import_works("simple.py", "simple")(tmp_path)
        assert ok is True

    def test_check_import_works_fail(self, tmp_path):
        (tmp_path / "broken.py").write_text("def f(\n")
        ok, msg = check_import_works("broken.py", "broken")(tmp_path)
        assert ok is False


class TestSetupFiles:
    """Verify task setup files create correctly."""

    def test_e1_no_setup(self, tmp_path):
        _setup_deep_workspace(TASK_E1, tmp_path)
        # E1 has no setup files
        assert tmp_path.exists()

    def test_g1_inventory_exists(self, tmp_path):
        _setup_deep_workspace(TASK_G1, tmp_path)
        f = tmp_path / "inventory.py"
        assert f.exists()
        content = f.read_text()
        assert "class Inventory" in content
        # Verify bugs are present
        assert "budget // product.price" in content  # Bug 1
        assert ".stock < self.restock" in content     # Bug 2
        assert "1 + pct" in content                   # Bug 3

    def test_h1_framework_exists(self, tmp_path):
        _setup_deep_workspace(TASK_H1, tmp_path)
        assert (tmp_path / "framework" / "__init__.py").exists()
        assert (tmp_path / "framework" / "app.py").exists()
        assert (tmp_path / "framework" / "router.py").exists()
        assert (tmp_path / "framework" / "request.py").exists()
        assert (tmp_path / "framework" / "response.py").exists()
        assert (tmp_path / "framework" / "database.py").exists()
        assert (tmp_path / "framework" / "utils.py").exists()
        assert (tmp_path / "main_app.py").exists()

    def test_i1_paginate_exists(self, tmp_path):
        _setup_deep_workspace(TASK_I1, tmp_path)
        f = tmp_path / "data.py"
        assert f.exists()
        content = f.read_text()
        assert "start = page * per_page" in content  # Bug present

    def test_g1_bugs_actually_fail(self, tmp_path):
        """Verify the seeded bugs cause incorrect behavior."""
        _setup_deep_workspace(TASK_G1, tmp_path)
        # Bug 2: needs_restock should return True when stock == threshold
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(f"""\
                import sys; sys.path.insert(0, r'{tmp_path}')
                from inventory import Inventory
                inv = Inventory()
                inv.add_product("widget", 10.0, 10)
                # stock=10, threshold=10 — should need restock but bug says no
                assert inv.needs_restock("widget") == False, "Bug should make this False"
                print("BUG_CONFIRMED")
            """)],
            capture_output=True, text=True, timeout=10
        )
        assert "BUG_CONFIRMED" in result.stdout

    def test_i1_bug_actually_fails(self, tmp_path):
        """Verify the paginate bug causes wrong results."""
        _setup_deep_workspace(TASK_I1, tmp_path)
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(f"""\
                import sys; sys.path.insert(0, r'{tmp_path}')
                from data import paginate
                result = paginate(list(range(100)), page=1, per_page=10)
                # Bug: page 1 returns items[10:20] instead of items[0:10]
                assert result["items"] == list(range(10, 20)), "Bug confirmed"
                print("BUG_CONFIRMED")
            """)],
            capture_output=True, text=True, timeout=10
        )
        assert "BUG_CONFIRMED" in result.stdout


class TestReportFormatting:
    """Verify report formatting."""

    def test_format_empty(self):
        report = format_deep_report([])
        assert "DEEP COMPARATIVE EVAL REPORT" in report
        assert "SUMMARY" in report

    def test_format_with_results(self):
        results = [
            ComparativeResult(
                task_id="E1",
                category="E",
                description="Test task",
                single=RunMetrics(quality_score=0.8, time_seconds=10, cost_usd=0.01,
                                  checks_passed=4, checks_total=5, check_details=[
                                      {"check": "a", "category": "correctness", "weight": 1.0,
                                       "multiplier": 3.0, "passed": True, "message": "ok"}
                                  ]),
                swarm=RunMetrics(quality_score=0.9, time_seconds=8, cost_usd=0.03,
                                checks_passed=5, checks_total=5, check_details=[
                                    {"check": "a", "category": "correctness", "weight": 1.0,
                                     "multiplier": 3.0, "passed": True, "message": "ok"}
                                ]),
                verdict="swarm_better",
                quality_delta=0.1,
                speedup=1.25,
                cost_ratio=3.0,
            ),
        ]
        report = format_deep_report(results)
        assert "E1" in report
        assert "Swarm Better" in report
        assert "Swarm Wins:    1" in report

    def test_format_category_breakdown(self):
        results = [
            ComparativeResult(task_id="E1", category="E", single=RunMetrics(quality_score=0.8),
                             swarm=RunMetrics(quality_score=0.9, checks_total=5), verdict="swarm_better"),
            ComparativeResult(task_id="F1", category="F", single=RunMetrics(quality_score=0.7),
                             swarm=RunMetrics(quality_score=0.6, checks_total=5), verdict="single_better"),
        ]
        report = format_deep_report(results)
        assert "Comparative" in report
        assert "Parallel Speedup" in report


class TestEfficiencyScores:
    """Verify efficiency score computation."""

    def test_single_only(self):
        """When only single ran, cost/time scores equal quality."""
        r = ComparativeResult(
            task_id="X1",
            single=RunMetrics(quality_score=0.85, cost_usd=0.01, time_seconds=10.0),
            swarm=RunMetrics(),  # no swarm
        )
        _compute_efficiency_scores(r)
        assert r.single_cost_score == pytest.approx(0.85)
        assert r.single_time_score == pytest.approx(0.85)
        assert r.single_overall == pytest.approx(0.85)
        assert r.swarm_cost_score == 0.0
        assert r.swarm_time_score == 0.0
        assert r.swarm_overall == 0.0

    def test_swarm_same_cost_time(self):
        """When costs and times are equal, scores equal quality."""
        r = ComparativeResult(
            task_id="X1",
            single=RunMetrics(quality_score=0.9, cost_usd=0.01, time_seconds=10.0, checks_total=5),
            swarm=RunMetrics(quality_score=0.9, cost_usd=0.01, time_seconds=10.0, checks_total=5),
        )
        _compute_efficiency_scores(r)
        assert r.single_cost_score == pytest.approx(0.9)
        assert r.swarm_cost_score == pytest.approx(0.9)
        assert r.single_overall == pytest.approx(0.9)
        assert r.swarm_overall == pytest.approx(0.9)

    def test_swarm_expensive_penalized(self):
        """Swarm with 10x cost gets penalized on cost score."""
        r = ComparativeResult(
            task_id="X1",
            single=RunMetrics(quality_score=0.92, cost_usd=0.01, time_seconds=5.0, checks_total=5),
            swarm=RunMetrics(quality_score=0.89, cost_usd=0.10, time_seconds=30.0, checks_total=5),
        )
        _compute_efficiency_scores(r)
        # Single cost score: 0.92 * (0.01 / 0.01) = 0.92
        assert r.single_cost_score == pytest.approx(0.92)
        # Swarm cost score: 0.89 * (0.01 / 0.10) = 0.089
        assert r.swarm_cost_score == pytest.approx(0.089)
        # Single time score: 0.92 * (5/5) = 0.92
        assert r.single_time_score == pytest.approx(0.92)
        # Swarm time score: 0.89 * (5/30) ≈ 0.1483
        assert r.swarm_time_score == pytest.approx(0.89 * 5 / 30, abs=0.001)
        # Single overall much higher than swarm
        assert r.single_overall > r.swarm_overall

    def test_zero_cost_defaults_to_quality(self):
        """When costs are zero, scores default to quality."""
        r = ComparativeResult(
            task_id="X1",
            single=RunMetrics(quality_score=0.8, cost_usd=0.0, time_seconds=0.0, checks_total=5),
            swarm=RunMetrics(quality_score=0.7, cost_usd=0.0, time_seconds=0.0, checks_total=5),
        )
        _compute_efficiency_scores(r)
        assert r.single_cost_score == pytest.approx(0.8)
        assert r.swarm_cost_score == pytest.approx(0.7)

    def test_efficiency_in_report(self):
        """Efficiency scores table appears in formatted report."""
        r = ComparativeResult(
            task_id="E1", category="E", description="test",
            single=RunMetrics(quality_score=0.92, cost_usd=0.01, time_seconds=5.0, checks_total=5),
            swarm=RunMetrics(quality_score=0.89, cost_usd=0.10, time_seconds=30.0, checks_total=5),
        )
        _compute_efficiency_scores(r)
        report = format_deep_report([r])
        assert "EFFICIENCY SCORES" in report
        assert "S.Qual" in report
        assert "W.Cost" in report
        assert "$0.010" in report  # actual single cost
        assert "$0.100" in report  # actual swarm cost


class TestScorePersistence:
    """Verify score save/load."""

    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".grokswarm").mkdir()
        results = [
            ComparativeResult(
                task_id="E1", category="E", description="test",
                single=RunMetrics(quality_score=0.9, cost_usd=0.01, time_seconds=5.0),
                single_cost_score=0.9, single_time_score=0.9, single_overall=0.9,
            ),
        ]
        path = _save_eval_scores(results)
        assert path.exists()
        loaded = _load_eval_scores()
        assert "E1" in loaded
        assert loaded["E1"]["single_quality"] == 0.9
        assert loaded["E1"]["single_overall"] == 0.9

    def test_upsert_preserves_existing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".grokswarm").mkdir()
        r1 = [ComparativeResult(task_id="E1", single=RunMetrics(quality_score=0.8),
                                single_overall=0.8)]
        _save_eval_scores(r1)
        r2 = [ComparativeResult(task_id="F1", single=RunMetrics(quality_score=0.7),
                                single_overall=0.7)]
        _save_eval_scores(r2)
        loaded = _load_eval_scores()
        assert "E1" in loaded
        assert "F1" in loaded

    def test_load_missing_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _load_eval_scores() == {}

    def test_save_includes_notes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".grokswarm").mkdir()
        results = [
            ComparativeResult(
                task_id="E1", category="E", description="test",
                single=RunMetrics(quality_score=0.9, check_details=[
                    {"check": "exists", "category": "correctness", "weight": 1.0,
                     "passed": True, "message": "ok"},
                    {"check": "exit_code", "category": "edge_cases", "weight": 1.0,
                     "passed": False, "message": "Expected exit 2"},
                ]),
            ),
        ]
        _save_eval_scores(results)
        loaded = _load_eval_scores()
        notes = loaded["E1"]["notes"]
        assert len(notes["weaknesses"]) == 1
        assert "exit_code" in notes["weaknesses"][0]
        assert len(notes["suggestions"]) >= 1


class TestGenerateNotes:
    """Verify _generate_notes() improvement analysis."""

    def test_all_passing(self):
        checks = [
            {"check": "a", "category": "correctness", "weight": 1.0, "passed": True, "message": "ok"},
            {"check": "b", "category": "correctness", "weight": 1.0, "passed": True, "message": "ok"},
        ]
        notes = _generate_notes(checks, [])
        assert any("All 2 correctness" in s for s in notes["strengths"])
        assert len(notes["weaknesses"]) == 0
        assert len(notes["suggestions"]) == 0

    def test_failures_generate_weaknesses(self):
        checks = [
            {"check": "div_zero_msg", "category": "edge_cases", "weight": 2.0,
             "passed": False, "message": "No error message on division by zero"},
        ]
        notes = _generate_notes(checks, [])
        assert len(notes["weaknesses"]) == 1
        assert "div_zero_msg" in notes["weaknesses"][0]
        assert "[Single]" in notes["weaknesses"][0]
        assert len(notes["suggestions"]) >= 1

    def test_both_paths_differences(self):
        single = [
            {"check": "a", "category": "correctness", "weight": 1.0, "passed": True, "message": "ok"},
            {"check": "b", "category": "correctness", "weight": 1.0, "passed": False, "message": "fail"},
        ]
        swarm = [
            {"check": "a", "category": "correctness", "weight": 1.0, "passed": False, "message": "fail"},
            {"check": "b", "category": "correctness", "weight": 1.0, "passed": True, "message": "ok"},
        ]
        notes = _generate_notes(single, swarm)
        assert any("Single passed" in s for s in notes["strengths"])
        assert any("Swarm passed" in s for s in notes["strengths"])

    def test_empty_checks(self):
        notes = _generate_notes([], [])
        assert notes == {"strengths": [], "weaknesses": [], "suggestions": []}

    def test_deduplicates_suggestions(self):
        checks = [
            {"check": "err1", "category": "correctness", "weight": 1.0,
             "passed": False, "message": "error in output"},
            {"check": "err1", "category": "edge_cases", "weight": 1.0,
             "passed": False, "message": "error in output"},
        ]
        notes = _generate_notes(checks, [])
        # Same check name + message → same suggestion, should deduplicate
        assert len(notes["suggestions"]) == 1

    def test_suggest_fix_patterns(self):
        assert "exit code" in _suggest_fix("exit_code", "wrong exit").lower()
        assert "error" in _suggest_fix("div_zero", "error message").lower()
        assert "missing" in _suggest_fix("file_check", "file missing").lower() or \
               "exist" in _suggest_fix("file_check", "file missing").lower()
        assert _suggest_fix("", "") == ""


# ---------------------------------------------------------------------------
# CLI: Live eval runner
# ---------------------------------------------------------------------------

async def _run_live_deep_eval(task_ids: list[str] | None = None):
    """Run deep eval tasks with real API calls."""
    tasks = DEEP_EVAL_TASKS
    if task_ids:
        tasks = [t for t in DEEP_EVAL_TASKS if t.id in task_ids]
        if not tasks:
            print(f"No tasks found matching: {task_ids}")
            return

    results = []
    for task in tasks:
        print(f"\n{'='*70}")
        print(f"Deep Eval: {task.id} ({task.category}) — {task.description}")
        print(f"{'='*70}")

        if task.category == "I":
            result = await run_learning_eval(task)
        elif task.use_swarm:
            result = await run_comparative(task)
        else:
            # Single-agent only
            with tempfile.TemporaryDirectory(prefix=f"deep_{task.id}_") as tmp:
                ws = Path(tmp)
                single_metrics = await _run_single_agent(task, ws)
                result = ComparativeResult(
                    task_id=task.id,
                    category=task.category,
                    description=task.description,
                    single=single_metrics,
                )
                _compute_efficiency_scores(result)

        results.append(result)

    report = format_deep_report(results)
    print(f"\n{report}")

    # Save report
    report_path = Path(".grokswarm") / "deep_eval_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    # Save JSON
    json_path = Path(".grokswarm") / "deep_eval_results.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"Results saved to: {json_path}")

    # Save efficiency scores
    scores_path = _save_eval_scores(results)
    print(f"Scores saved to: {scores_path}")


async def _run_statistical_eval(task_ids: list[str] | None, n_runs: int):
    """Run statistical evaluation with N runs per task."""
    tasks = DEEP_EVAL_TASKS
    if task_ids:
        tasks = [t for t in DEEP_EVAL_TASKS if t.id in task_ids]
        if not tasks:
            print(f"No tasks found matching: {task_ids}")
            return

    results = []
    for task in tasks:
        print(f"\n{'='*70}")
        print(f"Statistical Eval: {task.id} ({n_runs} runs)")
        print(f"{'='*70}")
        stat = await run_statistical(task, n_runs)
        results.append(stat)
        print(f"  Single: {stat.single_mean:.0%} (std={stat.single_stddev:.3f})")
        if stat.swarm_scores:
            print(f"  Swarm:  {stat.swarm_mean:.0%} (std={stat.swarm_stddev:.3f})")
        print(f"  Verdict: {stat.verdict} (significant={stat.significant})")

    report = format_statistical_report(results)
    print(f"\n{report}")

    report_path = Path(".grokswarm") / "statistical_eval_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    # Build ComparativeResults from stat means for score saving
    task_map = {t.id: t for t in DEEP_EVAL_TASKS}
    score_results = []
    for stat in results:
        t = task_map.get(stat.task_id)
        cr = ComparativeResult(
            task_id=stat.task_id,
            category=t.category if t else "",
            description=t.description if t else "",
            single=RunMetrics(
                quality_score=stat.single_mean,
                cost_usd=_mean(stat.single_costs),
                time_seconds=_mean(stat.single_times),
            ),
            verdict=stat.verdict,
        )
        if stat.swarm_scores:
            cr.swarm = RunMetrics(
                quality_score=stat.swarm_mean,
                cost_usd=_mean(stat.swarm_costs),
                time_seconds=_mean(stat.swarm_times),
                checks_total=1,  # signal that swarm ran
            )
        _compute_efficiency_scores(cr)
        score_results.append(cr)
    scores_path = _save_eval_scores(score_results)
    print(f"Scores saved to: {scores_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GrokSwarm Deep Comparative Eval")
    parser.add_argument("--live", action="store_true", help="Run with real API calls")
    parser.add_argument("--task", nargs="*", help="Specific task IDs to run (e.g., E1 G1)")
    parser.add_argument("--list", action="store_true", help="List available deep eval tasks")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per task for statistical eval (default 1)")
    parser.add_argument("--claude", action="store_true",
                        help="Route agent execution through Claude Code CLI")
    parser.add_argument("--dualhead", action="store_true",
                        help="Enable dualhead: Grok plans, Claude reviews before execution")
    args = parser.parse_args()

    if args.list:
        print(f"{'ID':<8} {'Cat':>4} {'Checks':>7} {'Swarm':>6} {'Expert':<10} Description")
        print("-" * 90)
        for t in DEEP_EVAL_TASKS:
            swarm = "Yes" if t.use_swarm else "No"
            print(f"{t.id:<8} {t.category:>4} {len(t.checks):>7} {swarm:>6} {t.expert:<10} {t.description}")
        return

    if args.claude:
        import shutil
        if not shutil.which("claude"):
            print("Error: Claude Code CLI not found in PATH")
            sys.exit(1)
        shared.state.claude_mode = True
        print("Claude Code mode: ON — agents will execute via claude -p")

    if args.dualhead:
        import shutil
        if not shutil.which("claude"):
            print("Error: Claude Code CLI not found in PATH (needed for dualhead review)")
            sys.exit(1)
        shared.state.dualhead_mode = True
        print("Dualhead mode: ON — Grok plans, Claude reviews before execution")

    if args.live:
        task_ids = args.task if args.task else None
        if args.runs > 1:
            asyncio.run(_run_statistical_eval(task_ids, args.runs))
        else:
            asyncio.run(_run_live_deep_eval(task_ids))
    else:
        print("Running pytest deep eval suite...")
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


if __name__ == "__main__":
    main()
