"""Guardrails: PlanGate, GoalVerifier, LoopDetector, EvidenceTracker, Orchestrator, ToolFilter."""

import json
import asyncio
import hashlib
import re
from dataclasses import asdict
from pathlib import Path

import grokswarm.shared as shared
from grokswarm.models import AgentState, SubTask, TaskDAG
from grokswarm.tools_registry import READ_ONLY_TOOLS


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
        """Returns True if agent has a non-empty plan with at least 2 steps."""
        return bool(agent.plan) and len(agent.plan) >= 2

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

    @staticmethod
    async def decompose(task: str, experts: list[str]) -> TaskDAG:
        """Use LLM to decompose task into a TaskDAG."""
        prompt = Orchestrator.DECOMPOSITION_PROMPT.format(
            experts=experts, task=task
        )
        try:
            response = await shared._api_call_with_retry(
                lambda: shared.client.chat.completions.create(
                    model=shared.MODEL,
                    messages=[
                        {"role": "system", "content": "You are a task decomposition engine. Output valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                ),
                label="Orchestrator:decompose"
            )
            if hasattr(response, 'usage') and response.usage:
                from grokswarm.agents import _record_usage
                _record_usage(shared.MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)

            data = json.loads(response.choices[0].message.content.strip())
            subtasks = []
            for item in data.get("subtasks", []):
                subtasks.append(SubTask(
                    id=item.get("id", f"t{len(subtasks)+1}"),
                    description=item.get("description", ""),
                    expert=item.get("expert", "assistant"),
                    depends_on=item.get("depends_on", []),
                    deliverables=item.get("deliverables", []),
                ))
            return TaskDAG(goal=task, subtasks=subtasks)
        except Exception as e:
            shared.console.print(f"[swarm.warning]Orchestrator decomposition failed: {e}. Using single-task fallback.[/swarm.warning]")
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

MODEL_ROUTING = {
    "fast": "grok-3-fast",
    "reasoning": None,  # uses configured MODEL
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
        """Returns the appropriate model for the current phase."""
        pref = expert_yaml.get("model_preference", "reasoning")
        # Planning phase always uses fast model (read-only, cheaper)
        if phase == "planning":
            return MODEL_ROUTING["fast"] or shared.MODEL
        # Execution uses the expert's preference
        return MODEL_ROUTING.get(pref) or shared.MODEL


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
