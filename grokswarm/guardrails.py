"""Guardrails: PlanGate, GoalVerifier, LoopDetector, EvidenceTracker, Orchestrator, ToolFilter,
TaskComplexity, LessonsDB, CostGuard, DynamicTools."""

import json
import asyncio
import hashlib
import re
import time
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

import yaml

import grokswarm.shared as shared
from grokswarm.models import AgentState, SubTask, TaskDAG
from grokswarm.tools_registry import READ_ONLY_TOOLS

# 4-tier model routing:
#   fast       — exploration, read-only, simple tool calls (non-reasoning, cheapest)
#   reasoning  — standard agent execution (default)
#   hardcore   — complex planning, decomposition, difficult debugging
#   multi_agent — orchestrator task decomposition
_DEFAULT_MODEL_ROUTING = {
    "fast":        "grok-4-1-fast-non-reasoning",
    "reasoning":   "grok-4-1-fast-reasoning",
    "hardcore":    "grok-4.20-experimental-beta-latest",
    "multi_agent": "grok-4.20-multi-agent-experimental-beta-latest",
}
MODEL_ROUTING = dict(_DEFAULT_MODEL_ROUTING)

_VALID_TIERS = frozenset(_DEFAULT_MODEL_ROUTING)


def set_model_tier(tier: str, model: str) -> None:
    """Update a tier's model name. Raises ValueError for invalid tiers."""
    if tier not in _VALID_TIERS:
        raise ValueError(f"Invalid tier '{tier}'. Must be one of: {', '.join(sorted(_VALID_TIERS))}")
    MODEL_ROUTING[tier] = model
    if tier == "reasoning":
        shared.MODEL = model


def get_model_tiers() -> dict[str, str]:
    """Return a copy of the current MODEL_ROUTING."""
    return dict(MODEL_ROUTING)


def reset_model_tiers() -> None:
    """Restore MODEL_ROUTING to defaults."""
    MODEL_ROUTING.clear()
    MODEL_ROUTING.update(_DEFAULT_MODEL_ROUTING)
    shared.MODEL = _DEFAULT_MODEL_ROUTING["reasoning"]


# ---------------------------------------------------------------------------
# Feature 1: Plan-Then-Execute with Approval Gate
# ---------------------------------------------------------------------------

class PlanGate:
    """Enforces plan-then-execute workflow for agents."""

    PLAN_PHASE_TOOLS: set[str] = READ_ONLY_TOOLS | {"update_plan"}

    @staticmethod
    def check_tool_allowed(agent, tool_name: str) -> str | None:
        """Returns error message if tool is blocked in current phase, None if allowed."""
        if agent.phase == "planning" and tool_name not in PlanGate.PLAN_PHASE_TOOLS:
            return (
                f"[BLOCKED] Tool '{tool_name}' not available during planning phase. "
                "Finish your plan first using update_plan, then tools will be unlocked."
            )
        return None

    @staticmethod
    def check_plan_ready(agent) -> bool:
        """Returns True if agent has a non-empty plan. Single-step plans are fine for simple tasks."""
        return bool(agent.plan) and len(agent.plan) >= 1

    @staticmethod
    def transition_to_executing(agent):
        """Freeze the plan copy, unlock write tools, transition phase."""
        agent.approved_plan = [dict(s) for s in agent.plan]
        agent.plan_files_allowed = _extract_files_from_plan(agent.plan)
        agent.phase = "executing"

    @staticmethod
    def check_plan_deviation(agent, tool_name: str, tool_args: dict) -> str | None:
        """Warns if agent is editing files not mentioned in its plan."""
        if tool_name in ("edit_file", "write_file"):
            path = tool_args.get("path", "")
            if agent.plan_files_allowed and path not in agent.plan_files_allowed:
                return (
                    f"[WARNING] Editing '{path}' which was not in your approved plan. "
                    "If this is intentional, update your plan first."
                )
        return None


def _extract_files_from_plan(plan: list[dict]) -> set[str]:
    """Extract file paths mentioned in plan step descriptions."""
    files: set[str] = set()
    # Match common file patterns: word.ext, path/to/file.ext
    pattern = re.compile(r'[\w./\\-]+\.(?:py|json|jsx|tsx|js|ts|yaml|yml|toml|cfg|ini|md|txt|html|css|sql|sh|rs|go)')
    for step in plan:
        text = step.get("step", "")
        for match in pattern.finditer(text):
            files.add(match.group(0))
    return files


# ---------------------------------------------------------------------------
# Feature 2: Goal Verification Loop
# ---------------------------------------------------------------------------

class GoalVerifier:
    """Ensures agents actually solve the stated goal before declaring done."""

    REFLECTION_PROMPT = """[SYSTEM -- MANDATORY SELF-CHECK]
Before finishing, verify your work against the original goal:

ORIGINAL GOAL: {goal}

For each part of the goal:
1. State what was asked
2. State what you did
3. Cite the specific tool output that proves it works (e.g., "run_tests returned [PASS] with 12 tests")
4. If you cannot cite evidence, you MUST run the verification now

Call update_plan to mark all completed steps, then provide your final summary."""

    @staticmethod
    def build_reflection_prompt(original_goal: str) -> str:
        """Builds the reflection message injected before the final round."""
        return GoalVerifier.REFLECTION_PROMPT.format(goal=original_goal)

    @staticmethod
    def validate_completion(agent, tool_actions: list[str], full_output: str) -> dict:
        """Cross-references agent claims against actual tool results.
        Returns {valid: bool, issues: list[str]}"""
        issues = []
        if agent.plan:
            incomplete = [s for s in agent.plan if s["status"] not in ("done", "skipped")]
            if incomplete:
                issues.append(
                    f"{len(incomplete)} plan steps not marked done: "
                    f"{[s['step'] for s in incomplete]}"
                )
        # Check if any file mutations happened without subsequent test run
        has_mutations = any("edit_file" in a or "write_file" in a for a in tool_actions)
        has_tests = any("run_tests" in a for a in tool_actions)
        if has_mutations and not has_tests:
            issues.append("Code was modified but tests were never run")
        return {"valid": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
# Feature 3: Loop Detection + Escalation
# ---------------------------------------------------------------------------

class LoopDetector:
    """Detects when an agent is repeating the same failing pattern."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.tool_history: list[tuple[str, str]] = []  # (tool_name, key_arg_hash)
        self.edit_targets: dict[str, int] = {}  # file_path -> edit_count
        self.test_failures: list[str] = []  # list of test error signatures

    def record_tool_call(self, tool_name: str, args: dict, result: str):
        """Record a tool call and its result for pattern detection."""
        sig = (tool_name, self._hash_key_args(tool_name, args))
        self.tool_history.append(sig)

        if tool_name in ("edit_file", "write_file"):
            path = args.get("path", "")
            self.edit_targets[path] = self.edit_targets.get(path, 0) + 1

        if tool_name == "run_tests" and "[FAIL]" in result:
            error_sig = self._extract_error_signature(result)
            self.test_failures.append(error_sig)

    def check_loop(self) -> str | None:
        """Returns escalation message if a loop is detected, None otherwise."""
        # Pattern 1: Same file edited 4+ times
        for path, count in self.edit_targets.items():
            if count >= 4:
                return (
                    f"[LOOP DETECTED] You've edited '{path}' {count} times. "
                    "Step back -- your current approach isn't working. "
                    "Read the file fresh, reconsider your assumptions, "
                    "and try a fundamentally different fix."
                )

        # Pattern 2: Same test error 3+ times in a row
        if len(self.test_failures) >= 3:
            recent = self.test_failures[-3:]
            if len(set(recent)) == 1:
                return (
                    "[LOOP DETECTED] The same test error has occurred 3 times. "
                    "Your fix attempts aren't addressing the root cause. "
                    "Stop editing and re-read the error message carefully. "
                    "What assumption are you making that might be wrong?"
                )

        # Pattern 3: Same tool+args sequence repeated
        if len(self.tool_history) >= 6:
            last_3 = self.tool_history[-3:]
            prev_3 = self.tool_history[-6:-3]
            if last_3 == prev_3:
                return (
                    "[LOOP DETECTED] You're repeating the same sequence of actions. "
                    "This pattern is not making progress. Try a completely different "
                    "approach or ask for help via send_message."
                )

        return None

    def _hash_key_args(self, tool_name: str, args: dict) -> str:
        """Hash the key arguments that identify a unique tool invocation."""
        if tool_name in ("edit_file", "write_file", "read_file"):
            return args.get("path", "")
        if tool_name == "run_tests":
            return args.get("command", "default")
        if tool_name == "run_shell":
            return args.get("command", "")[:50]
        return str(sorted(args.keys()))

    def _extract_error_signature(self, result: str) -> str:
        """Extract a stable signature from a test failure for comparison."""
        lines = result.strip().split("\n")
        for line in reversed(lines):
            if "Error" in line or "FAILED" in line or "assert" in line.lower():
                return line.strip()[:100]
        return result[-100:]


# ---------------------------------------------------------------------------
# Feature 4: Claim-Evidence Tracking
# ---------------------------------------------------------------------------

class EvidenceTracker:
    """Tracks tool results as evidence and detects unsupported claims."""

    def __init__(self):
        self.file_reads: dict[str, int] = {}  # path -> last_read_round
        self.file_writes: dict[str, int] = {}  # path -> last_write_round
        self.test_results: list[tuple[int, str]] = []  # (round, PASS/FAIL)
        self.shell_results: list[tuple[int, str, str]] = []  # (round, cmd, truncated_output)
        self.current_round: int = 0
        self.models_used: dict[str, int] = {}  # model -> round_count

    def record_model(self, model: str):
        """Track which model was used for a round."""
        self.models_used[model] = self.models_used.get(model, 0) + 1

    def record_tool(self, round_num: int, tool_name: str, args: dict, result: str):
        self.current_round = round_num
        if tool_name == "read_file":
            self.file_reads[args.get("path", "")] = round_num
        elif tool_name in ("edit_file", "write_file"):
            self.file_writes[args.get("path", "")] = round_num
        elif tool_name == "run_tests":
            status = "PASS" if "[PASS]" in result else "FAIL" if "[FAIL]" in result else "UNKNOWN"
            self.test_results.append((round_num, status))
        elif tool_name == "run_shell":
            self.shell_results.append((round_num, args.get("command", ""), result[:200]))

    def check_stale_reads(self) -> list[str]:
        """Returns warnings for files that were written after last read."""
        warnings = []
        for path, write_round in self.file_writes.items():
            read_round = self.file_reads.get(path, -1)
            if read_round < write_round:
                warnings.append(
                    f"'{path}' was modified in round {write_round} but not re-read since. "
                    "Consider re-reading before making claims about its contents."
                )
        return warnings

    def get_evidence_summary(self) -> dict:
        """Returns a structured summary for the completion report."""
        last_test = self.test_results[-1] if self.test_results else None
        return {
            "files_read": len(self.file_reads),
            "files_written": len(self.file_writes),
            "test_runs": len(self.test_results),
            "last_test_status": last_test[1] if last_test else "never_run",
            "shell_commands": len(self.shell_results),
            "models_used": dict(self.models_used),
        }


# ---------------------------------------------------------------------------
# Feature 6: Persistent Orchestrator with Task DAG
# ---------------------------------------------------------------------------

class Orchestrator:
    """Persistent orchestrator that decomposes, sequences, and validates multi-agent work."""

    DECOMPOSITION_PROMPT = """Break this task into ordered sub-tasks. For each sub-task specify:
- id: short unique identifier (e.g., "t1", "t2")
- description: what specifically needs to be done
- expert: which expert should handle it (from: {experts})
- depends_on: list of sub-task ids that must complete first (empty list if none)
- deliverables: expected output files or verifiable outcomes

Output JSON: {{"subtasks": [...]}}

RULES:
- Order matters: put foundational work first
- Each sub-task should be independently verifiable
- Include a final verification sub-task that runs tests and checks all deliverables
- Task: {task}"""

    # Model fallback chain for decomposition
    _DECOMPOSE_MODELS = [
        MODEL_ROUTING["multi_agent"],
        MODEL_ROUTING["hardcore"],
        MODEL_ROUTING["reasoning"],
    ]

    @staticmethod
    async def decompose(task: str, experts: list[str]) -> TaskDAG:
        """Use LLM to decompose task into a TaskDAG. Tries multi-agent -> hardcore -> reasoning fallback."""
        prompt = Orchestrator.DECOMPOSITION_PROMPT.format(
            experts=experts, task=task
        )
        messages = [
            {"role": "system", "content": "You are a task decomposition engine. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        from grokswarm import llm

        for decompose_model in Orchestrator._DECOMPOSE_MODELS:
            try:
                chat = llm.create_chat(decompose_model, response_format="json_object")
                llm.populate_chat(chat, messages)
                response = await shared._api_call_with_retry(
                    lambda: chat.sample(),
                    label=f"Orchestrator:decompose({decompose_model})"
                )
                usage = llm.extract_usage(response)
                if usage["prompt_tokens"] or usage["completion_tokens"]:
                    from grokswarm.agents import _record_usage
                    _record_usage(decompose_model, usage["prompt_tokens"], usage["completion_tokens"],
                                  usage["cached_tokens"])

                data = json.loads((response.content or "").strip())
                subtasks = []
                for item in data.get("subtasks", []):
                    subtasks.append(SubTask(
                        id=item.get("id", f"t{len(subtasks)+1}"),
                        description=item.get("description", ""),
                        expert=item.get("expert", "assistant"),
                        depends_on=item.get("depends_on", []),
                        deliverables=item.get("deliverables", []),
                    ))
                if subtasks:
                    return TaskDAG(goal=task, subtasks=subtasks)
                # Empty subtasks — try next model
                shared.console.print(f"[swarm.warning]Model {decompose_model} returned empty subtasks, trying fallback...[/swarm.warning]")
            except Exception as e:
                shared.console.print(f"[swarm.warning]Orchestrator model {decompose_model} failed: {e}. Trying fallback...[/swarm.warning]")
                continue

        # All models failed — single-task fallback
        shared.console.print("[swarm.warning]All orchestrator models failed. Using single-task fallback.[/swarm.warning]")
        return TaskDAG(goal=task, subtasks=[
            SubTask(id="t1", description=task, expert=experts[0] if experts else "assistant")
        ])

    @staticmethod
    async def validate_phase_results(subtask: SubTask) -> tuple[bool, str]:
        """After a sub-task's agent finishes, verify deliverables exist."""
        issues = []
        for deliverable in subtask.deliverables:
            if deliverable.endswith((".py", ".js", ".ts", ".jsx", ".tsx")):
                full_path = shared.PROJECT_DIR / deliverable
                if not full_path.exists():
                    issues.append(f"Expected file '{deliverable}' not created")
                else:
                    # Quick syntax check for Python files
                    if deliverable.endswith(".py"):
                        try:
                            from grokswarm.tools_test import _lint_file
                            lint_err = _lint_file(str(full_path))
                            if lint_err:
                                issues.append(f"'{deliverable}' has syntax errors: {lint_err[:200]}")
                        except ImportError:
                            pass
        return (len(issues) == 0, "; ".join(issues))

    @staticmethod
    async def run(task: str, bus):
        """Main orchestration loop."""
        from grokswarm.agents import run_expert
        from grokswarm.registry_helpers import list_experts

        experts = list_experts()
        dag = await Orchestrator.decompose(task, experts)

        notify(f"Orchestrator: decomposed into {len(dag.subtasks)} sub-tasks")
        bus.post("orchestrator", json.dumps([asdict(t) for t in dag.subtasks]), kind="plan")

        # Store the DAG on shared state so /tasks can display it
        shared._current_dag = dag

        while not dag.is_complete():
            ready = dag.ready_tasks()
            if not ready and dag.failed_tasks():
                # All remaining tasks are blocked by failures
                failed_info = "; ".join(f"{t.id}: {t.result_summary}" for t in dag.failed_tasks())
                notify(f"Orchestrator: blocked by failures -- {failed_info}", level="warning")
                break

            if not ready:
                # Nothing ready and nothing failed -- shouldn't happen, but guard
                break

            # Launch ready tasks as agents
            tasks = []
            for subtask in ready:
                subtask.status = "running"
                agent_name = f"{subtask.expert}_{subtask.id}"
                subtask.agent_name = agent_name
                task_coro = run_expert(
                    subtask.expert,
                    subtask.description,
                    agent_name=agent_name,
                    bus=bus
                )
                tasks.append((subtask, asyncio.create_task(task_coro)))

            # Wait for this batch to complete
            await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

            # Validate results
            for subtask, _ in tasks:
                agent = shared.state.get_agent(subtask.agent_name)
                if agent and agent.state == AgentState.DONE:
                    valid, issues = await Orchestrator.validate_phase_results(subtask)
                    if valid:
                        subtask.status = "done"
                        notify(f"Orchestrator: [{subtask.id}] {subtask.description[:40]}... DONE")
                    else:
                        subtask.status = "failed"
                        subtask.result_summary = issues
                        notify(f"Orchestrator: [{subtask.id}] FAILED -- {issues[:80]}", level="warning")
                elif agent and agent.state == AgentState.ERROR:
                    subtask.status = "failed"
                    subtask.result_summary = "Agent error"
                else:
                    # Treat unknown states as failed
                    subtask.status = "failed"
                    subtask.result_summary = f"Agent state: {agent.state.value if agent else 'unknown'}"

        # Final summary
        done_count = sum(1 for t in dag.subtasks if t.status == "done")
        total = len(dag.subtasks)
        notify(f"Orchestrator: {done_count}/{total} sub-tasks complete")


# ---------------------------------------------------------------------------
# Feature 7: Tool Filtering + Model Routing
# ---------------------------------------------------------------------------

TASK_TYPE_TOOLS = {
    "code_change": {
        "read_file", "write_file", "edit_file", "list_directory", "search_files",
        "grep_files", "run_shell", "run_tests", "git_status", "git_diff",
        "git_commit", "git_branch", "git_log", "git_checkout", "find_symbol",
        "find_references", "update_plan",
    },
    "research": {
        "read_file", "list_directory", "search_files", "grep_files", "web_search",
        "x_search", "fetch_page", "extract_links", "find_symbol", "find_references",
        "update_plan",
    },
    "testing": {
        "read_file", "list_directory", "grep_files", "run_tests", "run_shell",
        "run_app_capture", "capture_tui_screenshot", "analyze_image", "git_status",
        "git_diff", "find_symbol", "update_plan",
    },
    "git_ops": {
        "git_status", "git_diff", "git_log", "git_commit", "git_checkout",
        "git_branch", "git_show_file", "git_blame", "git_stash", "git_init",
        "read_file", "list_directory", "update_plan",
    },
}


class ToolFilter:
    @staticmethod
    def get_tools_for_expert(expert_yaml: dict) -> set[str] | None:
        """Returns the set of allowed tool names for an expert, or None for all tools."""
        allowed = expert_yaml.get("tools")
        if allowed:
            return set(allowed)
        return None

    @staticmethod
    def get_model_for_phase(expert_yaml: dict, phase: str) -> str:
        """Returns the appropriate model for the current phase.

        Planning uses hardcore model (reasoning matters most when forming a plan).
        Execution uses the expert's preferred tier (default: reasoning).
        """
        pref = expert_yaml.get("model_preference", "reasoning")
        if phase == "planning":
            # Planning needs strong reasoning — use hardcore for better plans
            if pref == "fast":
                # Fast experts still get reasoning for planning (not non-reasoning)
                return MODEL_ROUTING["reasoning"]
            return MODEL_ROUTING["hardcore"]
        # Execution uses the expert's preference
        return MODEL_ROUTING.get(pref, MODEL_ROUTING["reasoning"])

    @staticmethod
    def get_model_for_escalation() -> str:
        """Returns the hardcore model for loop-detector escalation."""
        return MODEL_ROUTING["hardcore"]

    @staticmethod
    def get_orchestrator_model() -> str:
        """Returns the multi-agent model for orchestrator decomposition."""
        return MODEL_ROUTING["multi_agent"]


# ---------------------------------------------------------------------------
# Feature 8: Task Complexity Classifier
# ---------------------------------------------------------------------------

class TaskComplexity:
    """Classifies task complexity to decide whether to skip planning."""

    # Patterns that indicate a simple, single-action task
    SIMPLE_PATTERNS = [
        re.compile(r'\b(add|write|insert)\s+(a\s+)?docstring', re.I),
        re.compile(r'\b(fix|correct)\s+(a\s+)?(typo|spelling|whitespace)', re.I),
        re.compile(r'\brename\s+\w+\s+to\s+\w+', re.I),
        re.compile(r'\b(delete|remove)\s+(the\s+)?(unused|dead)\s+(code|import|variable|function)', re.I),
        re.compile(r'\b(add|insert)\s+(a\s+)?(comment|type\s*hint|annotation)', re.I),
        re.compile(r'\b(update|change|set)\s+(the\s+)?version', re.I),
        re.compile(r'\b(add|append)\s+.{0,20}\s+to\s+.{0,30}\.(txt|md|cfg|ini|toml|yaml|yml)', re.I),
        re.compile(r'\bformat\s+(the\s+)?(code|file)', re.I),
    ]

    # Patterns that indicate a complex, multi-step task
    COMPLEX_INDICATORS = [
        re.compile(r'\b(refactor|redesign|rewrite|overhaul|migrate)\b', re.I),
        re.compile(r'\b(implement|build|create)\s+(a\s+)?(new|full|complete)', re.I),
        re.compile(r'\band\s+(then|also|additionally)\b', re.I),  # multi-part task
        re.compile(r'\b(multiple|several|all)\s+(files?|modules?|components?)', re.I),
        re.compile(r'\b(test|verify|validate)\s+(and|then)\s+(fix|update|change)', re.I),
        re.compile(r'\bintegrat(e|ion)\b', re.I),
    ]

    @staticmethod
    def classify(task: str) -> str:
        """Returns 'simple', 'moderate', or 'complex'."""
        # Very short tasks are likely simple
        word_count = len(task.split())

        # Check explicit simple patterns
        for pat in TaskComplexity.SIMPLE_PATTERNS:
            if pat.search(task):
                return "simple"

        # Check explicit complex patterns
        for pat in TaskComplexity.COMPLEX_INDICATORS:
            if pat.search(task):
                return "complex"

        # Heuristic: short tasks (under 15 words) without complex indicators -> simple
        if word_count <= 12:
            return "simple"

        # Medium-length tasks
        if word_count <= 30:
            return "moderate"

        return "complex"

    @staticmethod
    def should_skip_planning(task: str) -> bool:
        """Returns True if the task is simple enough to skip the planning phase."""
        return TaskComplexity.classify(task) == "simple"


# ---------------------------------------------------------------------------
# Feature 9: Cross-Session Learning (LessonsDB)
# ---------------------------------------------------------------------------

class LessonsDB:
    """Persistent store of failure patterns and their solutions."""

    def __init__(self, path: Path | None = None):
        self._path = path or (shared.PROJECT_DIR / ".grokswarm" / "lessons_learned.yaml")

    def _load(self) -> list[dict]:
        if self._path.exists():
            try:
                data = yaml.safe_load(self._path.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    def _save(self, lessons: list[dict]):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Keep at most 50 lessons, newest first
        lessons = lessons[:50]
        self._path.write_text(yaml.dump(lessons, default_flow_style=False, allow_unicode=True), encoding="utf-8")

    def record_lesson(self, error_signature: str, fix_description: str,
                      files_involved: list[str], expert: str = ""):
        """Record a lesson learned from a loop recovery."""
        lessons = self._load()
        # Don't duplicate similar lessons
        for existing in lessons:
            if existing.get("error_sig", "")[:60] == error_signature[:60]:
                # Update existing lesson
                existing["fix"] = fix_description
                existing["count"] = existing.get("count", 1) + 1
                existing["last_seen"] = datetime.now().isoformat()
                self._save(lessons)
                return
        lessons.insert(0, {
            "error_sig": error_signature[:200],
            "fix": fix_description[:500],
            "files": files_involved[:10],
            "expert": expert,
            "count": 1,
            "created": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
        })
        self._save(lessons)

    def find_relevant(self, error_signature: str = "", files: list[str] | None = None,
                      limit: int = 3) -> list[dict]:
        """Find lessons relevant to the current error or files."""
        lessons = self._load()
        if not lessons:
            return []

        scored: list[tuple[float, dict]] = []
        for lesson in lessons:
            score = 0.0
            # Error signature overlap
            if error_signature and lesson.get("error_sig"):
                sig_words = set(error_signature.lower().split())
                lesson_words = set(lesson["error_sig"].lower().split())
                overlap = len(sig_words & lesson_words)
                if overlap > 2:
                    score += overlap * 2.0
            # File overlap
            if files and lesson.get("files"):
                file_overlap = len(set(files) & set(lesson["files"]))
                if file_overlap > 0:
                    score += file_overlap * 3.0
            # Frequency bonus (only as tiebreaker when already relevant)
            if score > 0:
                score += min(lesson.get("count", 1), 5) * 0.5
            if score > 0:
                scored.append((score, lesson))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [lesson for _, lesson in scored[:limit]]

    def format_for_prompt(self, lessons: list[dict]) -> str:
        """Format lessons into a system prompt injection."""
        if not lessons:
            return ""
        lines = ["[SYSTEM -- LESSONS FROM PREVIOUS SESSIONS]",
                 "These issues were encountered before in this project:"]
        for i, lesson in enumerate(lessons, 1):
            lines.append(f"  {i}. Error: {lesson['error_sig'][:100]}")
            lines.append(f"     Fix: {lesson['fix'][:200]}")
            if lesson.get("files"):
                lines.append(f"     Files: {', '.join(lesson['files'][:5])}")
        lines.append("Use this knowledge to avoid repeating the same mistakes.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature 10: Cost Guard (session limits + rate alarm)
# ---------------------------------------------------------------------------

class CostGuard:
    """Monitors session spending and enforces cost limits."""

    def __init__(self):
        self.session_budget_usd: float = 0.0  # 0 = no limit
        self.cost_timestamps: list[tuple[float, float]] = []  # (time, cost_delta)
        self._warned_thresholds: set[float] = set()
        self.WARN_THRESHOLDS = [1.0, 5.0, 10.0, 25.0, 50.0]
        self.RATE_LIMIT_PER_MIN: float = 2.0  # dollars per minute

    def set_budget(self, budget_usd: float):
        self.session_budget_usd = budget_usd

    def record_cost(self, cost_delta: float):
        """Record a cost event for rate tracking."""
        self.cost_timestamps.append((time.time(), cost_delta))
        # Trim old entries (keep last 5 minutes)
        cutoff = time.time() - 300
        self.cost_timestamps = [(t, c) for t, c in self.cost_timestamps if t > cutoff]

    def get_rate_per_min(self) -> float:
        """Calculate current spending rate in $/min over the last 60 seconds."""
        now = time.time()
        cutoff = now - 60
        recent = sum(c for t, c in self.cost_timestamps if t > cutoff)
        return recent  # already per-minute since window is 60s

    def check(self, total_session_cost: float) -> list[str]:
        """Returns list of actions to take: 'warn', 'pause_all', or empty."""
        actions = []

        # Threshold warnings
        for threshold in self.WARN_THRESHOLDS:
            if total_session_cost >= threshold and threshold not in self._warned_thresholds:
                self._warned_thresholds.add(threshold)
                actions.append(f"warn:${threshold:.0f}")

        # Budget exceeded
        if self.session_budget_usd > 0 and total_session_cost >= self.session_budget_usd:
            actions.append("pause_all")

        # Rate alarm
        rate = self.get_rate_per_min()
        if rate > self.RATE_LIMIT_PER_MIN:
            actions.append(f"rate_alarm:${rate:.2f}/min")

        return actions


# ---------------------------------------------------------------------------
# Feature 11: Dynamic Tool Granting
# ---------------------------------------------------------------------------

# Task keywords -> additional tools to grant
_DYNAMIC_TOOL_RULES: list[tuple[re.Pattern, set[str]]] = [
    (re.compile(r'\b(screenshot|visual|ui|tui|display|render|screen)\b', re.I),
     {"capture_tui_screenshot", "analyze_image", "run_app_capture"}),
    (re.compile(r'\b(tests?|spec|assert|verify|check)\b', re.I),
     {"run_tests"}),
    (re.compile(r'\b(run|execute|launch|start|deploy|install|build)\b', re.I),
     {"run_shell"}),
    (re.compile(r'\b(search|find|look\s*up|google|web)\b', re.I),
     {"web_search", "fetch_page"}),
    (re.compile(r'\b(git|commit|branch|merge|push|pull|diff|log|stash)\b', re.I),
     {"git_status", "git_diff", "git_log", "git_commit", "git_branch", "git_checkout"}),
    (re.compile(r'\b(image|picture|photo|png|jpg|svg)\b', re.I),
     {"analyze_image"}),
]


class DynamicTools:
    """Grants additional tools based on task content."""

    @staticmethod
    def infer_extra_tools(task: str) -> set[str]:
        """Returns set of additional tool names the task likely needs."""
        extras: set[str] = set()
        for pattern, tools in _DYNAMIC_TOOL_RULES:
            if pattern.search(task):
                extras |= tools
        return extras

    @staticmethod
    def merge_tools(expert_tools: set[str] | None, task: str) -> set[str] | None:
        """Merge expert's static tools with dynamically inferred ones.
        Returns None (all tools) if expert has no whitelist."""
        extras = DynamicTools.infer_extra_tools(task)
        if expert_tools is None:
            return None  # expert already has all tools
        if not extras:
            return expert_tools
        merged = expert_tools | extras
        return merged


# ---------------------------------------------------------------------------
# Feature 5: Notification system
# ---------------------------------------------------------------------------

_notification_queue: asyncio.Queue | None = None


def _get_notification_queue() -> asyncio.Queue:
    """Lazily create the notification queue (avoids issues with asyncio event loops)."""
    global _notification_queue
    if _notification_queue is None:
        _notification_queue = asyncio.Queue()
    return _notification_queue


def notify(message: str, level: str = "info"):
    """Queue a notification for display in the REPL."""
    try:
        _get_notification_queue().put_nowait((level, message))
    except Exception:
        pass  # Don't crash if queue isn't ready


def drain_notifications() -> list[tuple[str, str]]:
    """Drain all pending notifications. Returns list of (level, message)."""
    items = []
    try:
        q = _get_notification_queue()
        while not q.empty():
            items.append(q.get_nowait())
    except Exception:
        pass
    return items


def _auto_print(message: str, level: str = "info"):
    """Print a notification directly to the console AND queue it."""
    notify(message, level)
    try:
        if level == "error":
            shared.console.print(f"[bold red]{message}[/bold red]")
        elif level == "warning":
            shared.console.print(f"[bold yellow]{message}[/bold yellow]")
        else:
            shared.console.print(f"[dim]{message}[/dim]")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GuardrailPipeline: unified hook system for run_expert()
# ---------------------------------------------------------------------------

class GuardrailPipeline:
    """Encapsulates all guardrail state and provides clean hook methods for the agent loop.

    Usage in run_expert():
        pipeline = GuardrailPipeline(agent, display_name, task_desc, data, bus)
        # Before loop:
        pipeline.setup(conversation)
        # Each round:
        pipeline.on_round_start(round_num, max_rounds, conversation)
        model = pipeline.select_model()
        # After API call:
        should_stop = pipeline.on_api_response(response, round_num)
        # Before each tool:
        error = pipeline.check_tool(tool_name, args)
        # After each tool:
        pipeline.on_tool_result(tool_name, args, result, round_num, tc_id)
        # After all tools in a round:
        should_stop = pipeline.on_round_end(round_num, conversation)
        # After loop:
        pipeline.on_completion(tool_actions, full_output, rounds_used, max_rounds)
    """

    def __init__(self, agent, display_name: str, task_desc: str,
                 expert_data: dict, bus=None):
        self.agent = agent
        self.display_name = display_name
        self.task_desc = task_desc
        self.expert_data = expert_data
        self.bus = bus

        # Sub-systems
        self.loop_detector = LoopDetector()
        self.evidence_tracker = EvidenceTracker()
        self.lessons_db = LessonsDB()

        # State
        self.loop_escalation_count = 0
        self.model_escalated = False
        self.loop_error_at_escalation = ""
        self.made_file_mutations = False
        self.ran_tests = False
        self.verification_prompted = False
        self._step_notified: set[str] = set()

        # Model routing
        self.planning_model = ToolFilter.get_model_for_phase(expert_data, "planning")
        self.execution_model = ToolFilter.get_model_for_phase(expert_data, "executing")
        self.escalation_model = ToolFilter.get_model_for_escalation()

    # -- Setup (before loop) --

    def setup(self, conversation: list[dict]):
        """Initialize guardrails: complexity skip, lessons injection."""
        # Complexity-based planning skip
        if TaskComplexity.should_skip_planning(self.task_desc):
            self.agent.phase = "executing"
            shared._log(f"agent {self.display_name}: simple task, skipping planning phase")

        # Inject lessons from previous sessions
        relevant_lessons = self.lessons_db.find_relevant(files=[])
        if relevant_lessons:
            lesson_text = self.lessons_db.format_for_prompt(relevant_lessons)
            conversation.append({"role": "user", "content": lesson_text})

    def get_tool_schemas(self):
        """Get filtered tool schemas based on expert + dynamic tools."""
        from grokswarm.tools_registry import get_agent_tool_schemas
        allowed = ToolFilter.get_tools_for_expert(self.expert_data)
        allowed = DynamicTools.merge_tools(allowed, self.task_desc)
        return get_agent_tool_schemas(allowed_tools=allowed)

    # -- Round hooks --

    def on_round_start(self, round_num: int, max_rounds: int, conversation: list[dict]):
        """Inject system messages at appropriate round milestones."""
        pct = (round_num + 1) / max_rounds

        # Goal anchoring at ~40%
        if round_num > 0 and abs(pct - 0.4) < (1.0 / max_rounds):
            conversation.append({
                "role": "user",
                "content": f'[SYSTEM] Checkpoint -- original goal: "{self.task_desc[:200]}". Are you on track? If not, adjust your plan.',
            })

        # Reflection at ~80%
        reflection_round = int(max_rounds * 0.8)
        if round_num == reflection_round and round_num > 2:
            conversation.append({
                "role": "user",
                "content": GoalVerifier.build_reflection_prompt(self.task_desc),
            })

        # Final round warning
        if round_num == max_rounds - 1:
            conversation.append({
                "role": "user",
                "content": "[SYSTEM] This is your FINAL round. Wrap up now: summarize what you accomplished and what remains unfinished.",
            })

        # Stale read warnings every 5 rounds
        if round_num > 0 and round_num % 5 == 0:
            stale_warnings = self.evidence_tracker.check_stale_reads()
            if stale_warnings:
                conversation.append({
                    "role": "user",
                    "content": "[SYSTEM] " + " ".join(stale_warnings),
                })

    def select_model(self) -> str:
        """Select the right model based on phase + escalation state."""
        if self.model_escalated:
            model = self.escalation_model
        elif self.agent.phase == "planning":
            model = self.planning_model
        else:
            model = self.execution_model
        self.evidence_tracker.record_model(model)
        self.agent.current_model = model
        return model

    def check_cost_limits(self, round_num: int) -> bool:
        """Check cost guard and agent budget. Returns True if agent should stop."""
        # Cost guard: session-level spending checks
        cost_delta = self.agent.cost_usd
        _cost_guard.record_cost(cost_delta / max(round_num + 1, 1))
        cost_actions = _cost_guard.check(shared.state.global_cost_usd)
        for action in cost_actions:
            if action.startswith("warn:"):
                _auto_print(f"COST WARNING: session spending passed {action[5:]}", level="warning")
            elif action == "pause_all":
                _auto_print(f"COST LIMIT: session budget ${_cost_guard.session_budget_usd:.2f} exceeded -- pausing all agents", level="error")
                self.agent.transition(AgentState.PAUSED)
                if self.bus:
                    self.bus.post(self.display_name, f"Session budget exceeded (${shared.state.global_cost_usd:.2f}/${_cost_guard.session_budget_usd:.2f})", kind="status")
                return True
            elif action.startswith("rate_alarm:"):
                _auto_print(f"COST RATE ALARM: spending {action[11:]} -- consider pausing agents", level="warning")

        # Per-agent budget check
        if not self.agent.check_budget():
            self.agent.transition(AgentState.PAUSED)
            if self.bus:
                self.bus.post(self.display_name, f"Agent over budget after round {round_num + 1}", recipient="*", kind="status")
            return True

        return False

    # -- Tool hooks --

    def check_tool(self, tool_name: str, args: dict) -> str | None:
        """Check if a tool call is allowed. Returns error message or None."""
        gate_error = PlanGate.check_tool_allowed(self.agent, tool_name)
        if gate_error:
            shared._log(f"agent {self.display_name}: PlanGate blocked {tool_name}")
        return gate_error

    def _log_tool_call(self, tool_name: str, args: dict, result: str, round_num: int):
        """Append to agent's rolling tool call log for /peek visibility."""
        args_summary = ""
        if tool_name in ("read_file", "write_file", "edit_file"):
            args_summary = args.get("path", "")
        elif tool_name == "run_tests":
            args_summary = args.get("command", "default")
        elif tool_name == "run_shell":
            args_summary = args.get("command", "")[:60]
        elif tool_name == "update_plan":
            args_summary = f"{len(args.get('steps', []))} steps"
        else:
            args_summary = str(list(args.keys()))[:40]
        result_preview = result[:120].replace("\n", " ")
        self.agent.tool_call_log.append({
            "tool": tool_name, "args": args_summary,
            "result": result_preview, "round": round_num,
        })
        if len(self.agent.tool_call_log) > 10:
            self.agent.tool_call_log = self.agent.tool_call_log[-10:]

    def on_tool_result(self, tool_name: str, args: dict, result: str, round_num: int):
        """Record tool result for loop detection, evidence tracking, and live log."""
        # Plan deviation warning (prepended to result by caller)
        deviation = PlanGate.check_plan_deviation(self.agent, tool_name, args)

        # Record in all trackers
        self.loop_detector.record_tool_call(tool_name, args, result)
        self.evidence_tracker.record_tool(round_num, tool_name, args, result)
        self._log_tool_call(tool_name, args, result, round_num)

        # Track mutations and test runs
        if tool_name in ("write_file", "edit_file"):
            self.made_file_mutations = True
        elif tool_name == "run_tests":
            self.ran_tests = True

        # Auto-transition from planning to executing after update_plan
        if tool_name == "update_plan" and self.agent.phase == "planning":
            if PlanGate.check_plan_ready(self.agent):
                if shared.state.trust_mode or shared.state.agent_mode > 1:
                    PlanGate.transition_to_executing(self.agent)
                    _auto_print(f"[{self.display_name}] Plan approved -- executing ({len(self.agent.plan)} steps)")
                    shared._log(f"agent {self.display_name}: auto-transitioned to executing phase")
                else:
                    plan_text = "\n".join(f"  {i+1}. {s['step']}" for i, s in enumerate(self.agent.plan))
                    if self.bus:
                        self.bus.post(self.display_name, f"Plan ready:\n{plan_text}", kind="plan")
                    _auto_print(f"[{self.display_name}] Plan: {len(self.agent.plan)} steps -- /reject {self.display_name} <feedback> to revise")
                    PlanGate.transition_to_executing(self.agent)
                    shared._log(f"agent {self.display_name}: plan posted, auto-transitioned to executing")

        return deviation

    def check_verification_gate(self, round_num: int, max_rounds: int, conversation: list[dict]) -> bool:
        """Check if agent should be forced to run tests. Returns True to continue loop."""
        if self.made_file_mutations and not self.ran_tests and not self.verification_prompted and round_num < max_rounds - 1:
            self.verification_prompted = True
            conversation.append({
                "role": "user",
                "content": "[SYSTEM] You modified code files but have not run tests yet. "
                           "Run run_tests now to verify your changes work before finishing.",
            })
            shared._log(f"agent {self.display_name}: verification gate triggered")
            return True
        return False

    def on_round_end(self, round_num: int, conversation: list[dict]) -> bool:
        """Check for loops after each round. Returns True if agent should stop."""
        # Loop detection
        loop_msg = self.loop_detector.check_loop()
        if loop_msg:
            self.loop_escalation_count += 1
            shared._log(f"agent {self.display_name}: loop detected (escalation #{self.loop_escalation_count})")
            _auto_print(f"WARNING: {self.display_name} stuck in loop (escalation #{self.loop_escalation_count})", level="warning")

            # Auto-log to bug tracker
            from grokswarm.bugs import log_loop_detection
            error_sig = self.loop_detector.test_failures[-1] if self.loop_detector.test_failures else ""
            log_loop_detection(self.display_name, self.loop_escalation_count, error_sig)

            if self.loop_escalation_count >= 3:
                self.agent.transition(AgentState.PAUSED)
                pause_msg = f"Agent {self.display_name} paused: stuck in loop after 3 escalations. Use /tell {self.display_name} to provide guidance or /resume {self.display_name}."
                if self.bus:
                    self.bus.post(self.display_name, pause_msg, kind="status")
                _auto_print(pause_msg, level="error")
                return True
            elif self.loop_escalation_count == 1:
                self.model_escalated = True
                if self.loop_detector.test_failures:
                    self.loop_error_at_escalation = self.loop_detector.test_failures[-1]
                elif self.loop_detector.edit_targets:
                    self.loop_error_at_escalation = f"repeated edits to {list(self.loop_detector.edit_targets.keys())[-1]}"
                relevant = self.lessons_db.find_relevant(
                    error_signature=self.loop_error_at_escalation,
                    files=list(self.loop_detector.edit_targets.keys()),
                )
                lesson_hint = ""
                if relevant:
                    lesson_hint = "\n\n" + self.lessons_db.format_for_prompt(relevant)
                _auto_print(f"[{self.display_name}] Escalating to hardcore model after loop detection", level="warning")
                conversation.append({
                    "role": "user",
                    "content": f"[SYSTEM] {loop_msg}\n\n[MODEL ESCALATED] You are now running on a more powerful reasoning model. Use this opportunity to think more carefully about the problem.{lesson_hint}",
                })
            else:
                conversation.append({
                    "role": "user",
                    "content": f"[SYSTEM] {loop_msg}\n\n[FINAL WARNING] One more loop and you will be paused. Ask for help via send_message if needed.",
                })

        # Milestone notifications for plan steps
        if self.agent.plan:
            done_count = sum(1 for s in self.agent.plan if s["status"] == "done")
            total_count = len(self.agent.plan)
            for step in self.agent.plan:
                if step["status"] == "done":
                    step_key = step['step'][:30]
                    if step_key not in self._step_notified:
                        self._step_notified.add(step_key)
                        _auto_print(f"[{self.display_name}] completed: {step['step'][:60]} ({done_count}/{total_count})")

        return False

    # -- Completion --

    def on_completion(self, tool_actions: list[str], full_output: str,
                      rounds_used: int, max_rounds: int) -> tuple[dict, list[str]]:
        """Run post-completion verification. Returns (evidence_summary, verification_issues)."""
        import sys

        verification_result = GoalVerifier.validate_completion(self.agent, tool_actions, full_output)
        verification_issues = verification_result.get("issues", [])

        # Post-completion smoke test
        if self.made_file_mutations and not self.ran_tests and "pytest" not in sys.modules:
            shared._log(f"agent {self.display_name}: post-completion smoke test")
            try:
                from grokswarm.tools_test import run_tests as _run_tests_fn
                test_output = _run_tests_fn(None, None)
                if "[FAIL]" in test_output:
                    verification_issues.append("[REGRESSION WARNING] Post-completion test run failed")
                    _auto_print(f"[{self.display_name}] REGRESSION WARNING: post-completion tests failed", level="warning")
            except Exception:
                pass

        ev_summary = self.evidence_tracker.get_evidence_summary()

        # Cross-session learning
        if self.loop_escalation_count > 0 and ev_summary.get("last_test_status") == "PASS":
            fix_desc = full_output[-500:] if full_output else "Unknown fix"
            files_involved = list(self.loop_detector.edit_targets.keys())
            try:
                self.lessons_db.record_lesson(
                    error_signature=self.loop_error_at_escalation,
                    fix_description=fix_desc,
                    files_involved=files_involved,
                    expert=self.expert_data.get('name', ''),
                )
                shared._log(f"agent {self.display_name}: recorded lesson for '{self.loop_error_at_escalation[:50]}'")
            except Exception:
                pass

        # Completion notification
        test_status = ev_summary.get("last_test_status", "never_run")
        models_info = ", ".join(f"{m}x{c}" for m, c in ev_summary.get("models_used", {}).items())
        _auto_print(f"[{self.display_name}] DONE -- {ev_summary.get('files_written', 0)} files changed, tests {test_status}, ${self.agent.cost_usd:.4f}, {rounds_used} rounds")

        return ev_summary, verification_issues


# Singleton cost guard for session-level cost tracking
_cost_guard = CostGuard()
