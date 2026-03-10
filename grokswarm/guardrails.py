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
import os
from grokswarm.models import AgentState, DeliberationRound, SubTask, TaskDAG
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
        """Blocks file mutations not mentioned in the agent's approved plan."""
        if tool_name in ("edit_file", "write_file"):
            path = tool_args.get("path", "")
            if agent.plan_files_allowed and path not in agent.plan_files_allowed:
                return (
                    f"[BLOCKED] Editing '{path}' which was not in your approved plan. "
                    "Use update_plan to add this file to your plan first, then retry."
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
        Returns {valid: bool, issues: list[str]}

        Note: plan-step completion is NOT checked here. Agents frequently
        forget to call update_plan even when work is done, causing false
        positives that trigger wasteful verification rounds.
        """
        issues = []
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
    """Detects when an agent is repeating the same failing pattern.

    Uses content-aware detection: tracks what changed (not just what file),
    parses pytest output specifically, and requires repeated identical content
    to declare a loop — not just repeated tool calls.
    """

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.tool_history: list[tuple[str, str]] = []  # (tool_name, content_hash)
        self.edit_targets: dict[str, int] = {}  # file_path -> edit_count
        self.edit_content_sigs: dict[str, list[str]] = {}  # file_path -> [content_hashes]
        self.test_failures: list[str] = []  # list of test error signatures

    def record_tool_call(self, tool_name: str, args: dict, result: str):
        """Record a tool call and its result for pattern detection."""
        sig = (tool_name, self._hash_key_args(tool_name, args))
        self.tool_history.append(sig)

        if tool_name in ("edit_file", "write_file"):
            path = args.get("path", "")
            self.edit_targets[path] = self.edit_targets.get(path, 0) + 1
            # Track content hash to distinguish "same edit repeated" from "making progress"
            content = args.get("content", "") or args.get("new_text", "") or args.get("old_text", "")
            content_hash = hashlib.md5(content.encode(errors="replace")).hexdigest()[:8]
            self.edit_content_sigs.setdefault(path, []).append(content_hash)

        if tool_name == "run_tests" and "[FAIL]" in result:
            error_sig = self._extract_error_signature(result)
            self.test_failures.append(error_sig)

    def check_loop(self) -> str | None:
        """Returns escalation message if a loop is detected, None otherwise."""
        # Pattern 1: Same file edited many times WITH stagnant content
        # Only triggers if recent edits are repetitive (same content hash),
        # not if each edit is making new changes (forward progress).
        for path, sigs in self.edit_content_sigs.items():
            count = len(sigs)
            if count >= 5 and self.test_failures:
                # With test failures: check if last 4 edits are repetitive (<=2 unique)
                if len(set(sigs[-4:])) <= 2:
                    return (
                        f"[LOOP DETECTED] You've edited '{path}' {count} times "
                        "with similar content each time. Your approach isn't working. "
                        "Read the file fresh, reconsider your assumptions, "
                        "and try a fundamentally different fix."
                    )
            elif count >= 8:
                # Without test failures: higher threshold, check last 5 for repetition
                if len(set(sigs[-5:])) <= 2:
                    return (
                        f"[LOOP DETECTED] You've edited '{path}' {count} times "
                        "with repetitive changes. Step back and try a different approach."
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

        # Pattern 3: Same tool+args sequence repeated 3 times (not just 2)
        if len(self.tool_history) >= 9:
            last_3 = self.tool_history[-3:]
            mid_3 = self.tool_history[-6:-3]
            first_3 = self.tool_history[-9:-6]
            if last_3 == mid_3 == first_3:
                return (
                    "[LOOP DETECTED] You're repeating the same sequence of actions. "
                    "This pattern is not making progress. Try a completely different "
                    "approach or ask for help via send_message."
                )

        return None

    def _hash_key_args(self, tool_name: str, args: dict) -> str:
        """Hash key arguments including content for edit/write tools."""
        if tool_name in ("edit_file", "write_file"):
            path = args.get("path", "")
            content = args.get("content", "") or args.get("new_text", "")
            content_hash = hashlib.md5(content.encode(errors="replace")).hexdigest()[:8]
            return f"{path}:{content_hash}"
        if tool_name == "read_file":
            return args.get("path", "")
        if tool_name == "run_tests":
            return args.get("command", "default")
        if tool_name == "run_shell":
            return args.get("command", "")[:50]
        return str(sorted(args.keys()))

    def _extract_error_signature(self, result: str) -> str:
        """Extract a stable signature from a test failure.

        Prefers pytest short-summary FAILED lines (most specific),
        falls back to lines containing Error/assert.
        """
        lines = result.strip().split("\n")
        # Prefer pytest short test summary: "FAILED test_x.py::test_name - reason"
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("FAILED ") and "::" in stripped:
                return stripped[:150]
        # Fallback: scan from bottom for error lines
        for line in reversed(lines):
            stripped = line.strip()
            if "Error" in stripped or "FAILED" in stripped or "assert" in stripped.lower():
                return stripped[:100]
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

    DECOMPOSITION_PROMPT = """Break this task into sub-tasks for parallel execution where possible.

For each sub-task specify:
- id: short unique identifier (e.g., "t1", "t2")
- description: what specifically needs to be done
- expert: which expert should handle it (from: {experts})
- depends_on: list of sub-task ids that must complete first (empty list [] if independent)
- deliverables: expected output files or verifiable outcomes

Output JSON: {{"subtasks": [...]}}

RULES:
- Each sub-task must be a COMPLETE deliverable: one agent implements AND tests its own module — NEVER split code and tests into separate sub-tasks
- Prefer fewer, larger sub-tasks — only split work that produces genuinely independent deliverables (separate files, no shared state)
- Maximize parallelism: independent deliverables should have empty depends_on so they run simultaneously
- Only add a dependency when a sub-task truly needs another's output
- Include a final verification sub-task that depends on all others and runs all tests together
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
                full_path = shared.get_project_dir() / deliverable
                if not full_path.exists():
                    issues.append(f"Expected file '{deliverable}' not created")
                else:
                    # Quick syntax check for Python files
                    if deliverable.endswith(".py"):
                        try:
                            from grokswarm.tools_test import _lint_file
                            lint_err = _lint_file(full_path)
                            if lint_err:
                                issues.append(f"'{deliverable}' has syntax errors: {lint_err[:200]}")
                        except ImportError:
                            pass
        return (len(issues) == 0, "; ".join(issues))

    @staticmethod
    def _consolidate_dag(dag: TaskDAG, task: str, expert: str) -> TaskDAG:
        """Collapse DAGs where parallelism doesn't help.

        If the decomposition only has 0-1 truly parallel work tasks
        (excluding verification/final tasks), single-agent is better.
        """
        def _is_verify(st: SubTask) -> bool:
            d = st.description.lower()
            return any(k in d for k in (
                "verify", "verification", "run all tests",
                "run tests", "final check", "validate all",
            ))

        work_tasks = [t for t in dag.subtasks if not _is_verify(t)]
        # Root tasks = work tasks with no dependencies on other work tasks
        work_ids = {t.id for t in work_tasks}
        parallel_roots = sum(
            1 for t in work_tasks
            if not any(dep in work_ids for dep in (t.depends_on or []))
        )

        if parallel_roots <= 1:
            notify(
                f"Orchestrator: consolidating {len(dag.subtasks)} subtasks "
                f"→ single agent (only {parallel_roots} parallel root)"
            )
            return TaskDAG(goal=task, subtasks=[
                SubTask(id="t1", description=task, expert=expert)
            ])

        notify(f"Orchestrator: keeping {len(dag.subtasks)} subtasks ({parallel_roots} parallel)")
        return dag

    @staticmethod
    def _is_git_repo() -> bool:
        """Check if the project directory is a git repository."""
        return (shared.PROJECT_DIR / ".git").exists()

    @staticmethod
    async def run(task: str, bus, *, use_worktrees: bool = True):
        """Main orchestration loop with per-agent branch isolation.

        When use_worktrees=True and the project is a git repo, each parallel
        agent batch gets its own git worktree (branch). After agents complete,
        their branches are merged back into the main branch.
        """
        from grokswarm.agents import run_expert, run_claude_expert
        from grokswarm.registry_helpers import list_experts
        from grokswarm.tools_git import git_worktree_add, git_worktree_remove, _run_git

        experts = list_experts()

        best_expert = experts[0] if experts else "assistant"

        _plan_t0 = time.monotonic()
        if TaskComplexity.should_decompose(task):
            dag = await Orchestrator.decompose(task, experts)
            # Dualhead deliberation: have external reviewer validate the plan
            if shared.state.dualhead_mode:
                reviewer = ClaudeReviewer()
                deliberator = Deliberator(reviewer)
                dag = await deliberator.deliberate(task, dag)
            dag = Orchestrator._consolidate_dag(dag, task, best_expert)
        else:
            notify("Orchestrator: fast-path — single-agent execution (no decomposition needed)")
            dag = TaskDAG(goal=task, subtasks=[
                SubTask(id="t1", description=task, expert=best_expert)
            ])
        shared._last_planning_time = round(time.monotonic() - _plan_t0, 2)

        notify(f"Orchestrator: decomposed into {len(dag.subtasks)} sub-tasks")
        bus.post("orchestrator", json.dumps([asdict(t) for t in dag.subtasks]), kind="plan")

        # Store the DAG on shared state so /tasks can display it
        shared._current_dag = dag

        is_git = Orchestrator._is_git_repo()
        enable_worktrees = use_worktrees and is_git

        # Track branches created for merging
        agent_branches: list[tuple[SubTask, str]] = []  # (subtask, branch_name)

        while not dag.is_complete():
            ready = dag.ready_tasks()
            if not ready and dag.failed_tasks():
                failed_info = "; ".join(f"{t.id}: {t.result_summary}" for t in dag.failed_tasks())
                notify(f"Orchestrator: blocked by failures -- {failed_info}", level="warning")
                break

            if not ready:
                break

            # Determine if we need worktrees: only when >1 task in a batch
            batch_worktrees = enable_worktrees and len(ready) > 1

            # Launch ready tasks as agents
            tasks = []
            batch_branches = []
            for subtask in ready:
                subtask.status = "running"
                agent_name = f"{subtask.expert}_{subtask.id}"
                subtask.agent_name = agent_name

                # Prepend results from dependency tasks so agent has context
                dep_context = ""
                for dep_id in (subtask.depends_on or []):
                    dep_task = dag.get_task(dep_id) if hasattr(dag, 'get_task') else None
                    if dep_task and getattr(dep_task, 'result_summary', None):
                        dep_context += f"\n[Prior result from {dep_id}]: {dep_task.result_summary}\n"
                full_desc = dep_context + subtask.description if dep_context else subtask.description

                workspace = None
                if batch_worktrees:
                    branch_name = f"agent/{agent_name}"
                    worktree_result = git_worktree_add(branch_name)
                    if not worktree_result.startswith("Error"):
                        workspace = Path(worktree_result)
                        batch_branches.append((subtask, branch_name))
                        notify(f"Orchestrator: [{subtask.id}] worktree created: {branch_name}")
                    else:
                        notify(f"Orchestrator: [{subtask.id}] worktree failed ({worktree_result}), running in shared workspace", level="warning")

                if shared.state.claude_mode:
                    task_coro = run_claude_expert(
                        task_desc=full_desc,
                        bus=bus,
                        agent_name=agent_name,
                        workspace_dir=workspace,
                        expert_name=subtask.expert,
                        is_sub_agent=True,
                    )
                else:
                    task_coro = run_expert(
                        subtask.expert,
                        full_desc,
                        agent_name=agent_name,
                        bus=bus,
                        workspace_dir=workspace,
                        is_sub_agent=True,
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
                    subtask.status = "failed"
                    subtask.result_summary = f"Agent state: {agent.state.value if agent else 'unknown'}"

            # Merge successful agent branches back into main
            if batch_branches:
                # Commit any uncommitted work in each worktree before merging
                for subtask, branch_name in batch_branches:
                    if subtask.status != "done":
                        continue
                    wt_path = shared.PROJECT_DIR / ".grokswarm" / "worktrees" / branch_name
                    if wt_path.exists():
                        _run_git("add", "-A", cwd=wt_path)
                        _run_git("commit", "-m",
                                 f"agent/{subtask.agent_name}: {subtask.description[:60]}",
                                 cwd=wt_path)

                # Merge each successful branch into current branch
                for subtask, branch_name in batch_branches:
                    if subtask.status != "done":
                        notify(f"Orchestrator: skipping merge of failed branch {branch_name}", level="warning")
                    else:
                        merge_result = _run_git("merge", "--no-ff", "-m",
                                                f"Merge agent/{subtask.agent_name}: {subtask.description[:60]}",
                                                branch_name)
                        if "CONFLICT" in merge_result or merge_result.startswith("Error"):
                            notify(f"Orchestrator: MERGE CONFLICT on {branch_name}: {merge_result[:200]}", level="error")
                            # Abort the merge and mark as failed
                            _run_git("merge", "--abort")
                            subtask.status = "failed"
                            subtask.result_summary = f"Merge conflict: {merge_result[:200]}"
                        else:
                            notify(f"Orchestrator: merged {branch_name} successfully")

                # Cleanup worktrees
                for subtask, branch_name in batch_branches:
                    try:
                        git_worktree_remove(branch_name, force=True)
                    except Exception:
                        pass

                agent_branches.extend(batch_branches)

        # Final summary
        done_count = sum(1 for t in dag.subtasks if t.status == "done")
        total = len(dag.subtasks)
        merged_count = sum(1 for st, _ in agent_branches if st.status == "done")
        if agent_branches:
            notify(f"Orchestrator: {done_count}/{total} sub-tasks complete, {merged_count} branches merged")
        else:
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

    # Signals that a task has multiple independent deliverables worth parallelizing
    MULTI_DELIVERABLE_SIGNALS = [
        re.compile(r'\b(multiple|several)\s+(files?|modules?|components?|services?|packages?)', re.I),
        re.compile(r'\b(three|four|five|six|[3-9])\s+(separate|independent|different|distinct)\s', re.I),
        re.compile(r'\bindependent(ly)?\b.*\b(modules?|files?|components?)\b', re.I),
        re.compile(r'\b(modules?|files?|components?)\b.*\bindependent(ly)?\b', re.I),
    ]

    @staticmethod
    def should_decompose(task: str) -> bool:
        """True only if task is complex AND has multi-deliverable signals."""
        if TaskComplexity.classify(task) != "complex":
            return False
        for pat in TaskComplexity.MULTI_DELIVERABLE_SIGNALS:
            if pat.search(task):
                return True
        return False


# ---------------------------------------------------------------------------
# Dualhead Deliberation: Grok ↔ External Reviewer
# ---------------------------------------------------------------------------

class ExternalReviewer:
    """Base class for CLI-based external plan reviewers (Claude, future Gemini)."""

    def __init__(self, name: str, cli_command: str):
        self.name = name
        self.cli_command = cli_command
        self.history: list[DeliberationRound] = []

    def build_review_prompt(self, plan_text: str, project_summary: str,
                            capabilities: str, prior_rounds: list[DeliberationRound]) -> str:
        """Build the full prompt sent to the external reviewer."""
        parts = [
            "You are reviewing a plan generated by GrokSwarm (an AI coding agent).",
            f"\n## Project Context\n{project_summary}",
            f"\n## GrokSwarm Capabilities\n{capabilities}",
        ]
        if prior_rounds:
            parts.append("\n## Prior Deliberation Rounds")
            for r in prior_rounds:
                parts.append(f"\n### Round {r.round_num}")
                parts.append(f"**Plan:**\n{r.grok_plan}")
                parts.append(f"**Your feedback:**\n{r.reviewer_feedback}")
        parts.append(f"\n## Current Plan to Review\n{plan_text}")
        parts.append(
            "\n## Instructions\n"
            "You and Grok are collaborating to produce the best plan.\n"
            "Review for: completeness, correctness, parallelism, edge cases, test coverage.\n"
            "If the plan is good enough to execute successfully, respond with: APPROVED\n"
            "You may add advisory notes AFTER the APPROVED line — Grok will see them.\n"
            "Only reject (respond without APPROVED) for issues that would cause the task to FAIL.\n"
            "Minor style or optimization suggestions should be noted AFTER APPROVED, not used as rejection reasons."
        )
        return "\n".join(parts)

    @staticmethod
    def parse_approval(response: str) -> bool:
        """Check if the reviewer approved the plan.
        CLI errors auto-approve so agents aren't blocked by tooling failures."""
        if response.startswith("[Claude review error:") or response.startswith("[Claude review timed out]") or response.startswith("[Claude CLI not found"):
            return True
        # Check first few non-blank lines for APPROVED (Claude sometimes adds preamble)
        lines_checked = 0
        for line in response.split("\n"):
            stripped = line.strip().upper()
            if not stripped:
                continue
            if "APPROVED" in stripped:
                return True
            lines_checked += 1
            if lines_checked >= 3:
                break
        return False

    def record_exchange(self, round_num: int, plan: str, feedback: str, approved: bool):
        """Record a deliberation round for context persistence."""
        self.history.append(DeliberationRound(
            round_num=round_num, grok_plan=plan,
            reviewer_feedback=feedback, approved=approved
        ))

    async def review(self, prompt: str, timeout: int = 120) -> str:
        """Send prompt to CLI and return response. Override in subclasses."""
        raise NotImplementedError


class ClaudeReviewer(ExternalReviewer):
    """Reviews plans via Claude Code CLI (`claude -p`)."""

    def __init__(self):
        super().__init__(name="Claude", cli_command="claude")

    # Env vars to strip so Claude CLI doesn't detect a nested session
    _ENV_STRIP = {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ACCESS_TOKEN"}

    async def review(self, prompt: str, timeout: int = 120) -> str:
        import subprocess
        cmd = [
            "claude", "-p",
            "--output-format", "json",
            "--no-session-persistence",
            "--max-turns", "1",
            "--max-budget-usd", "0.50",
            prompt,
        ]
        # Strip xAI keys + Claude nesting detection vars
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("XAI_", "GROK_")) and k not in self._ENV_STRIP}
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=timeout, env=env,
                    cwd=str(shared.get_project_dir()),
                ),
            )
            if result.returncode != 0:
                return f"[Claude review error: exit {result.returncode}] {result.stderr[:500]}"

            import json as _json
            try:
                data = _json.loads(result.stdout)
                text = data.get("result", "")
                if isinstance(text, list):
                    text = "\n".join(
                        b.get("text", "") for b in text
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                # Track cost
                cost = data.get("cost_usd") or data.get("costUsd") or 0
                if cost:
                    from grokswarm.agents import _cost_lock
                    with _cost_lock:
                        shared.state.project_cost_usd += cost
                        shared.state.global_cost_usd += cost
                return text
            except _json.JSONDecodeError:
                return result.stdout
        except subprocess.TimeoutExpired:
            return "[Claude review timed out]"
        except FileNotFoundError:
            return "[Claude CLI not found in PATH]"


class Deliberator:
    """Orchestrates Grok↔Reviewer deliberation rounds."""

    MAX_ROUNDS = 3

    CAPABILITIES = (
        "GrokSwarm is a multi-agent coding system with:\n"
        "- Tool loop: file read/write/edit, shell commands, git operations, tests\n"
        "- Orchestrator: decomposes tasks into parallel sub-tasks with DAG dependencies\n"
        "- Per-agent git worktrees for branch isolation in parallel execution\n"
        "- Guardrails: plan-then-execute, loop detection, cost budgets\n"
        "- Expert system: YAML-defined experts with tool whitelists and model preferences\n"
        "- Auto-lint, auto-checkpoint, auto-test after edits\n"
        "Each sub-task agent has up to 25 tool-call rounds and its own workspace."
    )

    def __init__(self, reviewer: ExternalReviewer):
        self.reviewer = reviewer

    @staticmethod
    def _format_dag_for_review(dag: TaskDAG) -> str:
        """Format a TaskDAG as human-readable plan text."""
        lines = [f"## Task: {dag.goal}\n", f"**{len(dag.subtasks)} sub-tasks:**\n"]
        for st in dag.subtasks:
            deps = f" (depends on: {', '.join(st.depends_on)})" if st.depends_on else " (independent)"
            lines.append(f"- **{st.id}** [{st.expert}]{deps}: {st.description}")
            if st.deliverables:
                lines.append(f"  Deliverables: {', '.join(st.deliverables)}")
        independent = sum(1 for st in dag.subtasks if not st.depends_on)
        lines.append(f"\n**Parallelism:** {independent} independent tasks can run simultaneously")
        return "\n".join(lines)

    def _build_project_summary(self) -> str:
        """Build a concise project summary for the reviewer."""
        from grokswarm.context import scan_project_context_cached, format_context_for_prompt
        ctx = scan_project_context_cached(shared.get_project_dir())
        full = format_context_for_prompt(ctx)
        if len(full) > 4000:
            full = full[:4000] + "\n... (truncated)"
        return full

    async def deliberate(self, task: str, dag: TaskDAG) -> TaskDAG:
        """Run deliberation rounds. Returns possibly-revised DAG."""
        project_summary = self._build_project_summary()
        plan_text = self._format_dag_for_review(dag)

        for round_num in range(1, self.MAX_ROUNDS + 1):
            notify(f"Deliberation round {round_num}/{self.MAX_ROUNDS}")
            shared._set_status(f"dualhead deliberating... round {round_num}/{self.MAX_ROUNDS}")
            shared.console.print(
                f"\n[bold green]\\[Grok → {self.reviewer.name}][/bold green] "
                f"Round {round_num} — sending plan for review..."
            )

            prompt = self.reviewer.build_review_prompt(
                plan_text, project_summary, self.CAPABILITIES, self.reviewer.history
            )

            feedback = await self.reviewer.review(prompt)

            approved = ExternalReviewer.parse_approval(feedback)
            self.reviewer.record_exchange(round_num, plan_text, feedback, approved)
            # Persist to global log so /delib can display it
            rnd = DeliberationRound(round_num=round_num, grok_plan=plan_text,
                                    reviewer_feedback=feedback, approved=approved)
            shared.state.deliberation_log.append(rnd)

            label = "APPROVED" if approved else "FEEDBACK"
            color = "green" if approved else "yellow"
            shared.console.print(
                f"[bold {color}]\\[{self.reviewer.name} → Grok][/bold {color}] "
                f"Round {round_num} — {label}"
            )
            display_fb = feedback[:800] + "..." if len(feedback) > 800 else feedback
            shared.console.print(f"[dim]{display_fb}[/dim]")

            if approved:
                shared._clear_status()
                notify(f"Deliberation: plan approved in round {round_num}")
                return dag

            shared._set_status(f"dualhead deliberating... revising plan")
            dag = await self._revise_plan(task, dag, feedback)
            plan_text = self._format_dag_for_review(dag)

        shared._clear_status()
        notify("Deliberation: max rounds reached, proceeding with latest plan")
        return dag

    async def _revise_plan(self, task: str, dag: TaskDAG, feedback: str) -> TaskDAG:
        """Ask Grok to revise the DAG based on reviewer feedback."""
        from grokswarm import llm
        from grokswarm.registry_helpers import list_experts

        experts = list_experts()
        current_plan = json.dumps([asdict(st) for st in dag.subtasks], indent=2)

        revision_prompt = (
            f"You are revising a task decomposition based on reviewer feedback.\n\n"
            f"Original task: {task}\n\n"
            f"Current plan:\n{current_plan}\n\n"
            f"Reviewer feedback:\n{feedback}\n\n"
            f"Available experts: {experts}\n\n"
            f"Output a revised JSON: {{\"subtasks\": [...]}}\n"
            f"Each subtask has: id, description, expert, depends_on, deliverables.\n"
            f"Apply the feedback while maintaining parallelism where possible."
        )
        messages = [
            {"role": "system", "content": "You are a task decomposition engine. Output valid JSON only."},
            {"role": "user", "content": revision_prompt},
        ]

        try:
            model = MODEL_ROUTING["reasoning"]
            chat = llm.create_chat(model, response_format="json_object")
            llm.populate_chat(chat, messages)
            response = await shared._api_call_with_retry(
                lambda: chat.sample(),
                label="Deliberator:revise"
            )
            text = llm.extract_text(response)
            data = json.loads(text)
            subtasks = [
                SubTask(
                    id=st["id"], description=st["description"],
                    expert=st.get("expert", experts[0] if experts else "assistant"),
                    depends_on=st.get("depends_on", []),
                    deliverables=st.get("deliverables", []),
                )
                for st in data.get("subtasks", [])
            ]
            if subtasks:
                return TaskDAG(goal=task, subtasks=subtasks)
        except Exception as e:
            notify(f"Deliberation: revision failed ({e}), keeping current plan", level="warning")

        return dag


# ---------------------------------------------------------------------------
# Session-Level Dualhead Deliberation (for /plan mode)
# ---------------------------------------------------------------------------

async def deliberate_on_session_plan(plan: list[dict], user_prompt: str, conversation: list[dict]) -> bool:
    """Send the session plan to Claude for review when dualhead_mode is on.

    Returns True if approved, False if rejected (caller should revert to planning).
    """
    if not plan:
        return True

    plan_text = "\n".join(f"{i+1}. {s.get('step', s)}" for i, s in enumerate(plan))
    formatted = (
        f"## Session Plan\n"
        f"## User prompt: {user_prompt[:500]}\n\n"
        f"**Plan ({len(plan)} steps):**\n{plan_text}"
    )

    reviewer = ClaudeReviewer()
    deliberator = Deliberator(reviewer)
    project_summary = deliberator._build_project_summary()

    prompt = reviewer.build_review_prompt(
        formatted, project_summary, Deliberator.CAPABILITIES, []
    )

    notify("Dualhead: sending session plan to Claude for review")
    shared._set_status("dualhead deliberating... reviewing session plan")
    shared.console.print(
        "\n[bold green]\\[Grok → Claude][/bold green] "
        "Reviewing session plan..."
    )

    feedback = await reviewer.review(prompt)
    approved = ExternalReviewer.parse_approval(feedback)

    rnd = DeliberationRound(round_num=1, grok_plan=formatted,
                            reviewer_feedback=feedback, approved=approved)
    shared.state.deliberation_log.append(rnd)
    shared._clear_status()

    label = "APPROVED" if approved else "FEEDBACK"
    color = "green" if approved else "yellow"
    shared.console.print(
        f"[bold {color}]\\[Claude → Grok][/bold {color}] {label}"
    )
    display_fb = feedback[:800] + "..." if len(feedback) > 800 else feedback
    shared.console.print(f"[dim]{display_fb}[/dim]")

    if not approved:
        conversation.append({
            "role": "user",
            "content": (
                "[DUALHEAD REVIEW] An external reviewer (Claude) has feedback on your plan:\n\n"
                f"{feedback}\n\n"
                "Revise your plan with update_plan to address this feedback, then proceed."
            ),
        })
        shared._log("session plan: dualhead review — sent back to planning")

    return approved


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

    def record_insight(self, category: str, insight: str,
                       files: list[str] | None = None, expert: str = ""):
        """Record a project insight (pattern, preference, architecture note)."""
        lessons = self._load()
        # Don't duplicate
        for existing in lessons:
            if existing.get("category") == category and existing.get("error_sig", "")[:60] == insight[:60]:
                existing["count"] = existing.get("count", 1) + 1
                existing["last_seen"] = datetime.now().isoformat()
                self._save(lessons)
                return
        lessons.insert(0, {
            "category": category,
            "error_sig": insight[:200],
            "fix": "",
            "files": (files or [])[:10],
            "expert": expert,
            "count": 1,
            "created": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
        })
        self._save(lessons)

    def record_completion(self, task_desc: str, files_modified: list[str],
                          tools_used: list[str], expert: str = ""):
        """Record a successful task completion for project intelligence."""
        # Only record if there's meaningful work
        if not files_modified:
            return
        summary = f"Completed: {task_desc[:100]}"
        if tools_used:
            summary += f" (tools: {', '.join(tools_used[:5])})"
        self.record_insight("completion", summary, files=files_modified, expert=expert)

    def format_for_prompt(self, lessons: list[dict]) -> str:
        """Format lessons into a system prompt injection."""
        if not lessons:
            return ""
        error_lessons = [l for l in lessons if l.get("fix")]
        insight_lessons = [l for l in lessons if not l.get("fix") and l.get("category")]

        lines = []
        if error_lessons:
            lines.append("[SYSTEM -- LESSONS FROM PREVIOUS SESSIONS]")
            lines.append("These issues were encountered before in this project:")
            for i, lesson in enumerate(error_lessons, 1):
                lines.append(f"  {i}. Error: {lesson['error_sig'][:100]}")
                lines.append(f"     Fix: {lesson['fix'][:200]}")
                if lesson.get("files"):
                    lines.append(f"     Files: {', '.join(lesson['files'][:5])}")
            lines.append("Use this knowledge to avoid repeating the same mistakes.")

        if insight_lessons:
            lines.append("")
            lines.append("[SYSTEM -- PROJECT INTELLIGENCE]")
            lines.append("Known patterns and recent completions:")
            for i, insight in enumerate(insight_lessons[:5], 1):
                cat = insight.get("category", "note")
                lines.append(f"  {i}. [{cat}] {insight['error_sig'][:150]}")
                if insight.get("files"):
                    lines.append(f"     Files: {', '.join(insight['files'][:5])}")

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
                 expert_data: dict, bus=None, is_sub_agent: bool = False):
        self.agent = agent
        self.display_name = display_name
        self.task_desc = task_desc
        self.expert_data = expert_data
        self.bus = bus
        self.is_sub_agent = is_sub_agent

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
        self._is_simple_task = False
        self._step_notified: set[str] = set()
        self._needs_deliberation = False  # set after plan transition when dualhead is on
        self._deliberation_round = 0
        self._deliberation_history = []   # [(plan_text, feedback, approved)]
        self._deliberation_escalated = False
        self.deliberation_bonus_rounds = 0  # legacy — kept for test compat
        self._deliberation_time_total = 0.0  # cumulative seconds spent in deliberation

        # Model routing — sub-agents use reasoning for planning (task already
        # well-defined by orchestrator decomposition, no need for expensive hardcore)
        if is_sub_agent:
            self.planning_model = MODEL_ROUTING["reasoning"]
        else:
            self.planning_model = ToolFilter.get_model_for_phase(expert_data, "planning")
        self.execution_model = ToolFilter.get_model_for_phase(expert_data, "executing")
        self.escalation_model = ToolFilter.get_model_for_escalation()

    # -- Setup (before loop) --

    def setup(self, conversation: list[dict]):
        """Initialize guardrails: complexity skip, lessons injection, memory retrieval."""
        # Complexity-based planning skip + model routing
        self._is_simple_task = TaskComplexity.should_skip_planning(self.task_desc)
        if self._is_simple_task:
            self.agent.phase = "executing"
            shared._log(f"agent {self.display_name}: simple task, skipping planning phase")

        # Inject lessons from previous sessions — use task description
        # and any file paths mentioned in it for relevance matching
        task_files = list(_extract_files_from_plan([{"step": self.task_desc}]))
        relevant_lessons = self.lessons_db.find_relevant(
            error_signature=self.task_desc[:200],
            files=task_files,
        )
        if relevant_lessons:
            lesson_text = self.lessons_db.format_for_prompt(relevant_lessons)
            conversation.append({"role": "user", "content": lesson_text})

        # Inject relevant memories from past agent runs
        from grokswarm.registry_helpers import find_relevant_memories
        expert_name = self.expert_data.get("name", "")
        memories = find_relevant_memories(expert_name, self.task_desc, max_results=2)
        if memories:
            mem_parts = []
            for mem in memories:
                # Truncate to keep context window reasonable
                snippet = mem["content"][:600]
                ts = mem["timestamp"][:19] if mem["timestamp"] else "?"
                mem_parts.append(f"--- {mem['key']} ({ts}) ---\n{snippet}")
            mem_text = "[PAST EXPERIENCE] Relevant results from previous agent runs:\n\n" + "\n\n".join(mem_parts)
            conversation.append({"role": "user", "content": mem_text})
            shared._log(f"agent {self.display_name}: injected {len(memories)} relevant memories")

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

    def select_model(self, round_num: int = 0) -> str:
        """Select the right model based on phase, task complexity, escalation, and round.

        Simple tasks use the fast model. Read-only early rounds use fast to save cost.
        Loop escalation overrides to hardcore.
        """
        if self.model_escalated:
            model = self.escalation_model
        elif self.agent.phase == "planning":
            model = self.planning_model
        elif self._is_simple_task and not self.made_file_mutations:
            # Simple tasks that haven't started mutating files: use fast model
            model = MODEL_ROUTING.get("fast", self.execution_model)
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
                _auto_print(f"COST RATE ALARM: spending {action[11:]} -- pausing agent to prevent runaway costs", level="error")
                self.agent.transition(AgentState.PAUSED)
                if self.bus:
                    self.bus.post(self.display_name, f"Rate alarm: spending {action[11:]}, agent paused", kind="status")
                return True

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
        # Plan deviation blocks file mutations to unapproved files
        deviation = PlanGate.check_plan_deviation(self.agent, tool_name, args)
        if deviation:
            shared._log(f"agent {self.display_name}: plan deviation blocked {tool_name} on {args.get('path', '?')}")
            return deviation
        return None

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
                # Flag for dualhead deliberation (async review happens in agent loop)
                if shared.state.dualhead_mode:
                    self._needs_deliberation = True

        return deviation

    async def deliberate_on_agent_plan(self, conversation: list[dict]):
        """Multi-round dualhead deliberation with escalation.

        Tracks rounds across calls. Flow:
          Rounds 1..N: Grok plans → Claude reviews (progressively lenient)
          Round N+1:   Hardcore Grok synthesizes → Claude reviews (lenient)
          After total max: auto-approve (Grok final say)
        """
        if not self._needs_deliberation:
            return
        self._needs_deliberation = False
        _delib_t0 = time.monotonic()
        try:
            plan_steps = self.agent.plan
            if not plan_steps:
                return

            self._deliberation_round += 1
            rnd = self._deliberation_round
            max_normal = shared.state.dualhead_max_rounds
            max_escalation = shared.state.dualhead_escalation_rounds
            total_max = max_normal + max_escalation

            formatted = self._format_plan_for_review()

            # Past total max → auto-approve (Grok final say)
            if rnd > total_max:
                self._auto_approve_with_history(conversation)
                return

            # Escalation phase: hardcore Grok synthesizes, then Claude reviews once more
            if rnd > max_normal and not self._deliberation_escalated:
                self._deliberation_escalated = True
                notify(f"Dualhead: escalating {self.display_name} — hardcore Grok synthesizing plan")
                shared.console.print(
                    f"\n[bold magenta]\\[Escalation][/bold magenta] "
                    f"Round {rnd}/{total_max}: Hardcore Grok synthesizing from {len(self._deliberation_history)} rounds of feedback..."
                )
                new_plan_text = await self._escalate_with_hardcore()
                if new_plan_text:
                    formatted = self._format_plan_for_review()  # re-format after plan update

            # Build round-aware review prompt and send to Claude
            reviewer = ClaudeReviewer()
            deliberator = Deliberator(reviewer)
            project_summary = deliberator._build_project_summary()

            prompt = self._build_collaborative_review_prompt(
                formatted, project_summary, rnd, max_normal, total_max
            )

            phase_label = "escalation" if rnd > max_normal else "normal"
            notify(f"Dualhead: round {rnd}/{total_max} ({phase_label}) — reviewing {self.display_name}")
            shared._set_status(f"dualhead round {rnd}/{total_max}... reviewing {self.display_name}")
            shared.console.print(
                f"\n[bold green]\\[Grok → Claude][/bold green] "
                f"Round {rnd}/{total_max} ({phase_label}) — reviewing {self.display_name}'s plan..."
            )

            feedback = await reviewer.review(prompt)
            approved = ExternalReviewer.parse_approval(feedback)

            # Record history
            self._deliberation_history.append((formatted, feedback, approved))
            delib_rnd = DeliberationRound(
                round_num=rnd, grok_plan=formatted,
                reviewer_feedback=feedback, approved=approved
            )
            shared.state.deliberation_log.append(delib_rnd)
            shared._clear_status()

            label = "APPROVED" if approved else "FEEDBACK"
            color = "green" if approved else "yellow"
            shared.console.print(
                f"[bold {color}]\\[Claude → Grok][/bold {color}] Round {rnd}/{total_max}: {label}"
            )
            display_fb = feedback[:800] + "..." if len(feedback) > 800 else feedback
            shared.console.print(f"[dim]{display_fb}[/dim]")

            if approved:
                # Inject advisory notes (text after APPROVED line) as non-blocking context
                notes = self._extract_advisory_notes(feedback)
                if notes:
                    conversation.append({
                        "role": "user",
                        "content": (
                            "[DUALHEAD NOTE] Claude approved your plan with advisory notes:\n\n"
                            f"{notes}\n\n"
                            "These are suggestions, not blockers. Proceed with execution."
                        ),
                    })
                notify(f"Dualhead: {self.display_name}'s plan approved by Claude (round {rnd})")
            elif rnd >= total_max:
                # Escalation round feedback → Grok final say, auto-approve
                self._auto_approve_with_history(conversation)
            else:
                # Normal feedback → inject and revert to planning
                remaining = total_max - rnd
                conversation.append({
                    "role": "user",
                    "content": (
                        f"[DUALHEAD REVIEW — Round {rnd}/{total_max}] "
                        f"Claude has feedback on your plan ({remaining} round(s) remaining before auto-approve):\n\n"
                        f"{feedback}\n\n"
                        "Revise your plan with update_plan to address this feedback, then proceed."
                    ),
                })
                self.agent.phase = "planning"
                self.agent.approved_plan = None
                shared._log(f"agent {self.display_name}: dualhead round {rnd} — sent back to planning")
        finally:
            _delib_elapsed = time.monotonic() - _delib_t0
            self._deliberation_time_total += _delib_elapsed
            shared._last_deliberation_time += _delib_elapsed

    def _format_plan_for_review(self) -> str:
        """Format the agent's current plan as markdown for review."""
        plan_steps = self.agent.plan or []
        plan_text = "\n".join(
            f"{i+1}. {s.get('step', s)}" for i, s in enumerate(plan_steps)
        )
        return (
            f"## Agent: {self.display_name}\n"
            f"## Task: {self.task_desc}\n\n"
            f"**Plan ({len(plan_steps)} steps):**\n{plan_text}"
        )

    def _build_collaborative_review_prompt(
        self, plan_text: str, project_summary: str,
        current_round: int, max_normal: int, total_max: int
    ) -> str:
        """Build a round-aware review prompt with history and escalation context."""
        parts = [
            "You are reviewing a plan generated by GrokSwarm (an AI coding agent).",
            f"\n## Project Context\n{project_summary}",
            f"\n## GrokSwarm Capabilities\n{Deliberator.CAPABILITIES}",
        ]

        # Include deliberation history
        if self._deliberation_history:
            parts.append("\n## Prior Deliberation Rounds")
            for i, (prev_plan, prev_feedback, prev_approved) in enumerate(self._deliberation_history, 1):
                status = "APPROVED" if prev_approved else "FEEDBACK"
                parts.append(f"\n### Round {i} ({status})")
                parts.append(f"**Plan:**\n{prev_plan}")
                parts.append(f"**Your feedback:**\n{prev_feedback}")

        parts.append(f"\n## Current Plan to Review (Round {current_round}/{total_max})\n{plan_text}")

        # Progressive leniency based on round
        if self._deliberation_escalated:
            parts.append(
                "\n## Instructions\n"
                "This plan was synthesized by a senior Grok model from all prior feedback.\n"
                "APPROVE unless the plan is fundamentally broken and would certainly fail.\n"
                "You may add advisory notes AFTER the APPROVED line.\n"
                "Respond with APPROVED unless there are critical showstopper issues."
            )
        elif current_round >= max_normal:
            parts.append(
                "\n## Instructions\n"
                f"This is the FINAL normal round ({current_round}/{max_normal}). "
                "Next round triggers hardcore escalation.\n"
                "You and Grok are collaborating to produce the best plan.\n"
                "Only reject for CRITICAL issues that would cause the task to FAIL.\n"
                "If the plan is workable, respond with: APPROVED\n"
                "Add advisory notes AFTER the APPROVED line if needed."
            )
        elif current_round > max_normal // 2:
            parts.append(
                "\n## Instructions\n"
                f"Round {current_round}/{max_normal} — approaching escalation threshold.\n"
                "You and Grok are collaborating to produce the best plan.\n"
                "Review for: completeness, correctness, parallelism, edge cases, test coverage.\n"
                "If the plan is good enough to execute successfully, respond with: APPROVED\n"
                "You may add advisory notes AFTER the APPROVED line.\n"
                "Only reject for issues that would cause the task to FAIL.\n"
                "Minor style or optimization suggestions should be noted AFTER APPROVED."
            )
        else:
            parts.append(
                "\n## Instructions\n"
                f"Round {current_round}/{max_normal}.\n"
                "You and Grok are collaborating to produce the best plan.\n"
                "Review for: completeness, correctness, parallelism, edge cases, test coverage.\n"
                "If the plan is good enough to execute successfully, respond with: APPROVED\n"
                "You may add advisory notes AFTER the APPROVED line — Grok will see them.\n"
                "Only reject (respond without APPROVED) for issues that would cause the task to FAIL.\n"
                "Minor style or optimization suggestions should be noted AFTER APPROVED, not used as rejection reasons."
            )

        return "\n".join(parts)

    async def _escalate_with_hardcore(self) -> str | None:
        """Call hardcore Grok to synthesize a final plan from all deliberation history.

        Updates self.agent.plan with the synthesized plan and returns the raw text.
        """
        from grokswarm import llm

        history_text = ""
        for i, (plan, feedback, approved) in enumerate(self._deliberation_history, 1):
            status = "APPROVED" if approved else "REJECTED"
            history_text += f"\n### Round {i} ({status})\n**Plan:**\n{plan}\n**Feedback:**\n{feedback}\n"

        prompt = (
            f"You are a senior AI planning expert. An agent working on the task below has gone through "
            f"{len(self._deliberation_history)} rounds of plan review with feedback from an external reviewer.\n\n"
            f"## Task\n{self.task_desc}\n\n"
            f"## Deliberation History\n{history_text}\n\n"
            f"## Your Job\n"
            f"Synthesize the best possible plan incorporating all valid feedback. "
            f"Output ONLY a numbered list of plan steps (1. ... 2. ... etc). "
            f"Be concrete, actionable, and thorough. Address the reviewer's concerns where valid, "
            f"but use your judgment — not all feedback needs to be followed."
        )

        messages = [
            {"role": "system", "content": "You are a senior AI planning expert. Output a clear, actionable plan."},
            {"role": "user", "content": prompt},
        ]

        try:
            model = MODEL_ROUTING["hardcore"]
            chat = llm.create_chat(model)
            llm.populate_chat(chat, messages)
            response = await shared._api_call_with_retry(
                lambda: chat.sample(),
                label=f"Dualhead:escalate({self.display_name})"
            )
            usage = llm.extract_usage(response)
            if usage["prompt_tokens"] or usage["completion_tokens"]:
                from grokswarm.agents import _record_usage
                _record_usage(model, usage["prompt_tokens"], usage["completion_tokens"],
                              usage["cached_tokens"])

            plan_text = (response.content or "").strip()
            if not plan_text:
                return None

            # Parse numbered steps and update agent plan
            new_steps = []
            for line in plan_text.split("\n"):
                line = line.strip()
                if re.match(r"^\d+\.\s", line):
                    step_text = re.sub(r"^\d+\.\s*", "", line)
                    new_steps.append({"step": step_text, "status": "pending"})
            if new_steps:
                self.agent.plan = new_steps
                shared.console.print(
                    f"[bold magenta]\\[Escalation][/bold magenta] "
                    f"Hardcore Grok produced {len(new_steps)}-step plan for {self.display_name}"
                )
            return plan_text
        except Exception as e:
            shared.console.print(
                f"[swarm.warning]Dualhead escalation failed: {type(e).__name__}: {str(e)[:100]}[/swarm.warning]"
            )
            shared._log(f"dualhead escalation error for {self.display_name}: {e}")
            return None

    def _auto_approve_with_history(self, conversation: list[dict]):
        """Auto-approve the plan (Grok final say) and inject all prior feedback as advisory."""
        all_feedback = []
        for i, (_, feedback, approved) in enumerate(self._deliberation_history, 1):
            if not approved:
                all_feedback.append(f"Round {i}: {feedback}")

        notes = "\n\n".join(all_feedback) if all_feedback else "No prior feedback."

        conversation.append({
            "role": "user",
            "content": (
                "[DUALHEAD FINAL — Grok Edit Master Head] "
                "Your plan has been auto-approved after maximum deliberation rounds.\n\n"
                "Prior reviewer feedback (advisory only, not blocking):\n\n"
                f"{notes}\n\n"
                "Proceed with execution. Consider the feedback where practical."
            ),
        })

        # Log approval
        rnd = DeliberationRound(
            round_num=self._deliberation_round,
            grok_plan=self._format_plan_for_review(),
            reviewer_feedback="[AUTO-APPROVED — Grok Edit Master Head]",
            approved=True
        )
        shared.state.deliberation_log.append(rnd)

        shared.console.print(
            f"[bold magenta]\\[Grok Edit Master Head][/bold magenta] "
            f"{self.display_name}'s plan auto-approved after {self._deliberation_round} rounds"
        )
        notify(f"Dualhead: {self.display_name} auto-approved (Grok final say)")
        shared._log(f"agent {self.display_name}: dualhead auto-approved after {self._deliberation_round} rounds")

    @staticmethod
    def _extract_advisory_notes(feedback: str) -> str:
        """Extract text after the APPROVED line as advisory notes."""
        lines = feedback.split("\n")
        found_approved = False
        note_lines = []
        for line in lines:
            if found_approved:
                note_lines.append(line)
            elif "APPROVED" in line.strip().upper():
                # Include any text on the same line after APPROVED
                after = line.strip().upper().split("APPROVED", 1)[-1].strip()
                if after and after not in (".", "!"):
                    # Get original case text after APPROVED
                    orig_after = line.strip().split("APPROVED", 1)[-1].strip() if "APPROVED" in line else ""
                    if orig_after:
                        note_lines.append(orig_after)
                found_approved = True
        result = "\n".join(note_lines).strip()
        return result

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
            done_count = sum(1 for s in self.agent.plan if s.get("status") == "done")
            total_count = len(self.agent.plan)
            for step in self.agent.plan:
                if step.get("status") == "done":
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

        # Cross-session learning: record lessons from loop recovery
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

        # Also learn from successful completions with test failures recovered
        elif self.made_file_mutations and ev_summary.get("last_test_status") == "PASS" and len(self.loop_detector.test_failures) > 0:
            files_involved = list(self.loop_detector.edit_targets.keys())
            try:
                self.lessons_db.record_lesson(
                    error_signature=self.loop_detector.test_failures[0],
                    fix_description=f"Task: {self.task_desc[:200]}. Fixed after {len(self.loop_detector.test_failures)} test failures.",
                    files_involved=files_involved,
                    expert=self.expert_data.get('name', ''),
                )
            except Exception:
                pass

        # Completion notification
        test_status = ev_summary.get("last_test_status", "never_run")
        models_info = ", ".join(f"{m}x{c}" for m, c in ev_summary.get("models_used", {}).items())
        _auto_print(f"[{self.display_name}] DONE -- {ev_summary.get('files_written', 0)} files changed, tests {test_status}, ${self.agent.cost_usd:.4f}, {rounds_used} rounds")

        return ev_summary, verification_issues


# Singleton cost guard for session-level cost tracking
_cost_guard = CostGuard()
