# Deep Eval V2 — Real-World Evaluation Design

## Overview

V1 (eval_deep.py) proves the machinery works with toy problems.
V2 proves GrokSwarm is **genuinely useful** with tasks that mirror real work.

Progression: V1 passes → V2 mechanical → V2 hard → V2 adversarial

---

## Gap 1: Real-World Task Complexity

Tasks that take a human 20-60 minutes, not 2 minutes.

### R1: Refactor a 400-Line Module

**Setup**: A single `monolith.py` (~400 lines) containing an HTTP API client
with mixed concerns — auth, request building, response parsing, retry logic,
logging, and caching all tangled in one class.

**Task**: "Refactor `monolith.py` into a clean package structure:
`client/auth.py`, `client/http.py`, `client/cache.py`, `client/retry.py`,
`client/models.py`. Keep all existing functionality working. The existing
`test_monolith.py` (provided) must still pass without modification."

**Why it's hard**: Requires understanding dependency order, knowing what to
extract first, keeping imports consistent, and not breaking the test suite.
A bad decomposition creates circular imports.

**Checks (~25)**:
- All 5 new files exist and compile (correctness)
- Original test suite passes unchanged (correctness, 3x weight)
- No circular imports — `python -c "from client import ..."` works (correctness)
- Each module is <100 lines (quality)
- monolith.py either deleted or is a thin re-export facade (quality)
- No duplicated functions across modules (quality)
- Has `__init__.py` with clean public API (completeness)

### R2: Debug a Race Condition

**Setup**: An async task queue (`task_queue.py`, ~200 lines) with a worker pool.
Contains a deliberate race condition: the `completed_count` is incremented
without a lock, and `results` dict is mutated from multiple coroutines.
Plus a second bug: tasks can be added after `shutdown()` is called.

**Task**: "The async task queue in `task_queue.py` has reliability issues.
Sometimes results go missing and the completed count is wrong when running
multiple workers. Find and fix the concurrency bugs. Write stress tests
that reliably reproduce the issues before the fix."

**Why it's hard**: Race conditions don't manifest in simple tests. The agent
needs to understand asyncio concurrency, write a stress test that actually
triggers the race, then apply the right fix (asyncio.Lock, not threading.Lock).

**Checks (~18)**:
- Original bugs are fixed — Lock around shared state (correctness)
- Shutdown prevents new task submission (correctness)
- Stress test exists and passes (correctness)
- No threading.Lock used (should be asyncio.Lock) (edge_cases)
- Worker count is configurable (completeness)
- Tests run under 10 seconds (quality)

### R3: Implement Against a Failing Test Suite

**Setup**: A `test_markdown.py` with 15 tests for a Markdown-to-HTML converter.
No implementation file exists.

**Task**: "Implement `markdown.py` that makes all tests in `test_markdown.py`
pass. Do NOT modify the tests."

Tests cover: headings (h1-h3), bold, italic, code blocks, links, unordered
lists, ordered lists, paragraphs, escaping, nested formatting.

**Why it's hard**: 15 specific test expectations to satisfy simultaneously.
Requires careful parsing logic, not just string replacement. Nested
formatting (`**bold _and italic_**`) is genuinely tricky.

**Checks (~18)**:
- markdown.py exists and compiles (correctness)
- All 15 tests pass (correctness, weight 5.0 — this IS the task)
- test_markdown.py is unmodified — hash check (correctness)
- No external dependencies used (edge_cases)
- Handles empty input (edge_cases)

---

## Gap 2: Baseline Comparison Against Claude Code

Not implementable as automated checks — this is a **process** recommendation.

**Design**: Create a shared eval corpus (`eval_corpus/`) with task descriptions
that both GrokSwarm and Claude Code can run. Output goes to separate dirs.
Same check functions score both.

For now: track GrokSwarm scores. When we have Claude Code scores for the
same tasks (run manually), add them as `baseline_scores` in each task
definition for comparison.

```python
@dataclass
class DeepEvalTask:
    ...
    baseline_scores: dict | None = None  # {"claude_code": 0.85, "cursor": 0.72}
```

Report would then show: "GrokSwarm scored 0.82 vs Claude Code baseline 0.85 (Δ -0.03)"

### B1: Side-by-Side Scaffold

**Task**: Same as R3 (implement against failing tests).
Record: score, time, cost for GrokSwarm.
Manually run the same task in Claude Code, record results.
The report compares both.

This doesn't need new code — just a convention for storing baseline numbers
and a report column for "vs baseline".

---

## Gap 3: Reliability Over N Runs

### Statistical Runner

Add `--runs N` flag to eval_deep.py. For each task, run N times, report:

```
Task  | Mean Score | StdDev | Min  | Max  | Mean Time | Mean Cost
E1    | 0.87       | 0.04   | 0.80 | 0.93 | 12.3s     | $0.015
```

**Verdict rules change**: "swarm_better" only if mean swarm score > mean single
score by more than 1 standard deviation. This prevents lucky-roll conclusions.

```python
@dataclass
class StatisticalResult:
    task_id: str
    n_runs: int
    single_scores: list[float]
    swarm_scores: list[float]
    single_mean: float
    single_stddev: float
    swarm_mean: float
    swarm_stddev: float
    significant: bool  # delta > max(single_stddev, swarm_stddev)
```

Implementation: ~60 lines wrapping `run_comparative()` in a loop.

---

## Gap 4: Cost-Adjusted Quality

### Efficiency Score

Add to verdict computation:

```python
efficiency = quality_score / max(cost_usd, 0.001)  # quality per dollar
```

New verdict categories:
- `swarm_efficient` — better quality AND better or equal cost-efficiency
- `swarm_better_but_costly` — better quality but >2x cost
- `single_efficient` — single agent wins on efficiency
- `tie`

### C1: Cost-Efficiency Task

**Task**: Same as E1 (CLI calculator) but verdict uses efficiency score.
A swarm that scores 0.95 at $0.08 loses to a single agent scoring 0.90 at $0.01.

No new task code — just a modified verdict function that factors in cost.

---

## Gap 5: User-Judgment Checks (Semi-Automated)

### Heuristic Quality Checks

Things we CAN check mechanically as proxies for "good code":

```python
def check_no_god_functions(path: str, max_lines: int = 50):
    """No function longer than max_lines — proxy for readability."""

def check_has_docstrings(path: str, min_functions: int = 3):
    """Public functions have docstrings — proxy for documentation quality."""

def check_cyclomatic_complexity(path: str, max_complexity: int = 10):
    """No function exceeds complexity threshold — proxy for maintainability."""

def check_no_hardcoded_values(path: str, patterns: list[str]):
    """Magic numbers/strings extracted to constants."""

def check_consistent_style(path: str):
    """Passes flake8/ruff with default rules."""
```

### J1: Code Quality Eval

**Setup**: Ask agent to build a non-trivial module (URL shortener with
storage, validation, collision handling, stats tracking).

**Task**: "Build a URL shortener library..."

**Checks split**:
- 40% correctness (it works)
- 30% quality (readable, documented, no god functions, passes linter)
- 20% edge cases (invalid URLs, collisions, concurrent access)
- 10% completeness (all features present)

### J2: Architecture Decision Eval

**Setup**: Provide a requirements doc with ambiguous parts.

**Task**: "Build a plugin system for a CLI tool. Requirements:
- Plugins are Python files in a directory
- Each plugin can register commands
- Plugins can depend on other plugins
- Plugins can be enabled/disabled"

**Check for reasonable decisions**:
- Did it handle circular plugin dependencies? (edge_cases)
- Did it validate plugin structure? (quality)
- Did it document the plugin API? (quality)
- Does it actually work? (correctness)

---

## Gap 6: Failure Mode / Resilience Testing

### K1: Partial Failure Recovery

**Setup**: 3 files need fixing. One fix is straightforward, one is medium,
one is near-impossible (obfuscated code that can't realistically be fixed).

**Task**: "Fix all bugs in the project."

**What we measure**: Does the agent:
- Fix the easy and medium bugs? (correctness)
- Recognize the hard one is unfixable and say so? (quality)
- NOT break the easy fixes while attempting the hard one? (edge_cases)
- Report what it accomplished vs what it couldn't? (completeness)

### K2: Bad Orchestrator Decomposition Recovery

**Setup**: A task that the Orchestrator will likely decompose poorly
(e.g., a task with hidden dependencies between subtasks that aren't
obvious from the description).

**Task**: "Add pagination to the user list endpoint. The endpoint uses
a custom query builder that needs to be extended first."

Hidden dependency: the query builder doesn't support LIMIT/OFFSET,
so the pagination implementation will fail until that's fixed.

**Checks**:
- Did it eventually figure out the dependency? (correctness)
- Did it fix the query builder before/during pagination? (quality)
- Or did it fail and report why? (acceptable, but lower score)

### K3: Graceful Degradation Under Token Limits

**Task**: Same as H1 but with `max_rounds=5` (very tight budget).

**Checks**:
- Did it accomplish SOMETHING useful? (correctness)
- Did it prioritize the most important parts? (quality)
- Did it not crash or loop? (edge_cases)
- Did it report what it couldn't finish? (completeness)

### K4: Conflicting Instructions

**Setup**: A requirements file that contradicts itself.
"The function should return None on error" vs test that expects it to raise.

**Task**: "Implement the function per the requirements. Tests must pass."

**Checks**:
- Did it follow the tests (the ground truth) over the prose? (correctness)
- Did it note the contradiction? (quality)

---

## Implementation Order

```
Phase 1 (Current — V1):
  E1, F1, G1, H1, I1 — toy problems, prove machinery works
  STATUS: 37 tests passing, ready for --live

Phase 2 (Mechanical V2):
  Statistical runner (--runs N)
  Cost-adjusted verdict
  Heuristic quality checks (linter, complexity, docstrings)
  ~100 lines new code

Phase 3 (Real-World V2):
  R1 (refactor 400-line module)
  R2 (debug race condition)
  R3 (implement against failing tests)
  J1 (code quality eval)
  ~300 lines: setup files + checks

Phase 4 (Adversarial V2):
  K1 (partial failure recovery)
  K2 (bad decomposition recovery)
  K3 (graceful degradation)
  K4 (conflicting instructions)
  ~200 lines: setup files + checks

Phase 5 (Baselines):
  Run tasks manually in Claude Code
  Record baseline_scores
  Add comparison column to report
```

## File Plan

| File | Action |
|------|--------|
| `eval_deep.py` | Add statistical runner, cost-adjusted verdict, quality checks |
| `eval_deep_v2.py` | NEW — Phase 3+4 tasks (keeps eval_deep.py focused) |
| `eval_corpus/` | NEW dir — shared setup files for larger tasks |
| `eval_baselines.json` | NEW — recorded baseline scores from other tools |
