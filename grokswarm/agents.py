"""SwarmBus, spawning, messaging, run_supervisor, run_expert, cost tracking."""

import json
import sys
import yaml
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path

import subprocess

import grokswarm.shared as shared
from grokswarm.models import AgentState
from grokswarm.tools_registry import TOOL_DISPATCH, READ_ONLY_TOOLS, get_agent_tool_schemas
from grokswarm.registry_helpers import list_experts, save_memory
from grokswarm.engine import _execute_tool, _repair_json, _trim_conversation, _tool_detail, MAX_TOOL_RESULT_SIZE
from grokswarm.guardrails import (
    PlanGate, GoalVerifier, LoopDetector, EvidenceTracker,
    ToolFilter, Orchestrator, notify,
    TaskComplexity, LessonsDB, CostGuard, DynamicTools,
)

# Singleton cost guard for the session
_cost_guard = CostGuard()


class SwarmBus:
    """Lightweight SQLite message bus so swarm agents can see each other's work."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_dir = shared.PROJECT_DIR / ".grokswarm"
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
            "  total_tokens INTEGER NOT NULL,"
            "  cached_tokens INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        # Migrate: add cached_tokens column if missing (existing DBs)
        try:
            self.conn.execute("SELECT cached_tokens FROM metrics LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE metrics ADD COLUMN cached_tokens INTEGER NOT NULL DEFAULT 0")
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

    def log_usage(self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0):
        self.conn.execute(
            "INSERT INTO metrics (model, prompt_tokens, completion_tokens, total_tokens, cached_tokens) VALUES (?, ?, ?, ?, ?)",
            (model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, cached_tokens),
        )
        self.conn.commit()

    def get_metrics(self) -> dict:
        cur = self.conn.execute(
            "SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens), SUM(cached_tokens) FROM metrics"
        )
        row = cur.fetchone()
        return {
            "prompt_tokens": row[0] or 0,
            "completion_tokens": row[1] or 0,
            "total_tokens": row[2] or 0,
            "cached_tokens": row[3] or 0,
        }

    def check_abort(self) -> bool:
        cur = self.conn.execute("SELECT 1 FROM messages WHERE kind = 'abort' LIMIT 1")
        return cur.fetchone() is not None

    def close(self):
        self.conn.close()


# -- Persistent per-project cost accumulator --

def _costs_file() -> Path:
    return shared.PROJECT_DIR / ".grokswarm" / "costs.json"


def _load_project_costs():
    f = _costs_file()
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            shared.state.project_prompt_tokens = data.get("prompt_tokens", 0)
            shared.state.project_completion_tokens = data.get("completion_tokens", 0)
            shared.state.project_cached_tokens = data.get("cached_tokens", 0)
            shared.state.project_cost_usd = data.get("cost_usd", 0.0)
        except (json.JSONDecodeError, KeyError, OSError):
            pass


def _save_project_costs():
    f = _costs_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "prompt_tokens": shared.state.project_prompt_tokens,
        "completion_tokens": shared.state.project_completion_tokens,
        "cached_tokens": shared.state.project_cached_tokens,
        "cost_usd": round(shared.state.project_cost_usd, 6),
        "last_updated": datetime.now().isoformat(),
    }
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(f)


def _extract_cached_tokens(usage) -> int:
    """Extract cached_prompt_text_tokens from API response usage object."""
    if usage is None:
        return 0
    # xAI API returns cached_prompt_text_tokens
    val = getattr(usage, 'cached_prompt_text_tokens', None)
    if isinstance(val, int):
        return val
    return 0


def _record_usage(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0):
    get_bus().log_usage(model, prompt_tokens, completion_tokens, cached_tokens)
    inp_rate, cached_rate, out_rate = shared._get_pricing(model)
    shared.state.project_prompt_tokens += prompt_tokens
    shared.state.project_completion_tokens += completion_tokens
    shared.state.project_cached_tokens += cached_tokens
    # Cached tokens are charged at the lower cached rate
    non_cached = max(0, prompt_tokens - cached_tokens)
    shared.state.project_cost_usd += (
        (non_cached / 1_000_000.0) * inp_rate
        + (cached_tokens / 1_000_000.0) * cached_rate
        + (completion_tokens / 1_000_000.0) * out_rate
    )
    try:
        _save_project_costs()
    except OSError:
        pass


def get_bus() -> SwarmBus:
    db_path = str(shared.PROJECT_DIR / ".grokswarm" / "bus.db")
    if shared._bus_instance is None:
        shared._bus_instance = SwarmBus()
    else:
        try:
            cur_path = shared._bus_instance.conn.execute("PRAGMA database_list").fetchall()[0][2]
            if cur_path != db_path:
                shared._bus_instance.close()
                shared._bus_instance = SwarmBus()
        except Exception:
            shared._bus_instance = SwarmBus()
    return shared._bus_instance


async def run_supervisor(task: str):
    shared.console.print(f"[bold green]Supervisor analyzing task:[/bold green] {task}")
    existing = list_experts()
    system_prompt = f"""You are the Grok Swarm Supervisor. Break down the task and choose ONLY from these existing experts: {existing}.
Be direct. No hype. Output only the JSON.
Respond ONLY with valid JSON: {{"experts": ["assistant", "researcher"], "team_name": null or "string", "reason": "brief explanation"}}"""
    try:
        response = await shared._api_call_with_retry(
            lambda: shared.client.chat.completions.create(
                model=shared.MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": task}],
                response_format={"type": "json_object"},
            ),
            label="Supervisor"
        )
        if hasattr(response, 'usage') and response.usage:
            _record_usage(shared.MODEL, response.usage.prompt_tokens, response.usage.completion_tokens,
                          _extract_cached_tokens(response.usage))
        plan = json.loads(response.choices[0].message.content.strip())
        shared.console.print(f"[bold cyan]Plan:[/bold cyan] {plan}")
        return plan
    except (json.JSONDecodeError, ValueError, KeyError):
        shared.console.print("[swarm.warning]Supervisor returned invalid JSON, falling back to assistant.[/swarm.warning]")
        return {"experts": ["assistant"], "team_name": None, "reason": "fallback"}
    except Exception as e:
        shared.console.print(f"[swarm.error]Supervisor API error: {e}[/swarm.error]")
        return {"experts": ["assistant"], "team_name": None, "reason": f"API error fallback: {e}"}


EXPERT_DEFAULT_MAX_ROUNDS = 25


def _auto_checkpoint_before_agent(display_name: str):
    """If there are uncommitted changes, create a checkpoint commit so the user can revert agent work."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=shared.PROJECT_DIR, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return  # not a git repo or no changes
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True, text=True, cwd=shared.PROJECT_DIR, timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", f"auto-checkpoint before agent '{display_name}'"],
            capture_output=True, text=True, cwd=shared.PROJECT_DIR, timeout=15,
        )
        shared._log(f"auto-checkpoint created before agent '{display_name}'")
        shared.console.print(f"[swarm.dim]  git: auto-checkpoint before agent '{display_name}'[/swarm.dim]")
    except Exception:
        pass  # best-effort, don't block agent startup


def _detect_tech_stack() -> str:
    """Detect project tech stack from PROJECT_CONTEXT and key files. Returns a short summary string."""
    ctx = shared.PROJECT_CONTEXT
    if not ctx:
        return ""
    parts = []
    # Language stats
    lang_stats = ctx.get("language_stats", {})
    if lang_stats:
        top_langs = [ext.lstrip(".") for ext in list(lang_stats.keys())[:5]]
        parts.append(f"Languages: {', '.join(top_langs)}")
    # Detect from key files content
    key_files = ctx.get("key_files", {})
    frameworks = []
    py_version = ""
    for fname, content in key_files.items():
        c = content.lower() if content else ""
        if fname == "pyproject.toml" or fname == "setup.cfg":
            if "python_requires" in c:
                import re
                m = re.search(r'python_requires\s*=\s*["\']([^"\']+)', content)
                if m:
                    py_version = m.group(1)
            for fw in ["pytest", "textual", "rich", "fastapi", "flask", "django", "httpx",
                        "requests", "click", "typer", "pydantic", "sqlalchemy", "asyncio"]:
                if fw in c:
                    frameworks.append(fw)
        elif fname == "requirements.txt":
            for fw in ["pytest", "textual", "rich", "fastapi", "flask", "django", "httpx",
                        "requests", "click", "typer", "pydantic", "sqlalchemy"]:
                if fw in c:
                    frameworks.append(fw)
        elif fname == "package.json":
            for fw in ["react", "vue", "next", "express", "jest", "typescript", "tailwind"]:
                if fw in c:
                    frameworks.append(fw)
        elif fname == "Cargo.toml":
            for fw in ["tokio", "actix", "serde", "clap"]:
                if fw in c:
                    frameworks.append(fw)
    if py_version:
        parts.append(f"Python: {py_version}")
    if frameworks:
        parts.append(f"Frameworks/libs: {', '.join(sorted(set(frameworks)))}")
    # Detect test framework
    tree = ctx.get("tree", "")
    if "test_" in tree or "conftest" in tree:
        if "pytest" not in frameworks:
            parts.append("Testing: pytest (detected)")
    return "; ".join(parts) if parts else ""


def _build_completion_report(display_name: str, tool_actions: list[str], rounds_used: int,
                              max_rounds: int, full_output: str,
                              evidence_summary: dict | None = None,
                              verification_issues: list[str] | None = None) -> str:
    """Build a structured completion report for bus posting."""
    files_written = set()
    files_edited = set()
    tests_run = False
    test_result = ""
    for action in tool_actions:
        if action.startswith("write_file"):
            path = action.split(" \u2192 ", 1)[1] if " \u2192 " in action else ""
            if path:
                files_written.add(path.strip())
        elif action.startswith("edit_file"):
            path = action.split(" \u2192 ", 1)[1] if " \u2192 " in action else ""
            if path:
                files_edited.add(path.strip())
        elif action.startswith("run_tests"):
            tests_run = True

    lines = [f"=== Agent '{display_name}' Completion Report ==="]
    lines.append(f"Rounds: {rounds_used}/{max_rounds}")
    if files_written:
        lines.append(f"Files created: {', '.join(sorted(files_written))}")
    if files_edited:
        lines.append(f"Files modified: {', '.join(sorted(files_edited))}")
    if not files_written and not files_edited:
        lines.append("Files changed: none")
    lines.append(f"Tests run: {'yes' if tests_run else 'no'}")
    lines.append(f"Tools used: {len(tool_actions)}")
    # Evidence summary
    if evidence_summary:
        lines.append(f"Evidence: {evidence_summary.get('files_read', 0)} files read, "
                      f"{evidence_summary.get('test_runs', 0)} test runs, "
                      f"last test: {evidence_summary.get('last_test_status', 'never_run')}")
        models = evidence_summary.get("models_used", {})
        if models:
            models_str = ", ".join(f"{m}: {c} rounds" for m, c in models.items())
            lines.append(f"Models used: {models_str}")
    # Verification issues
    if verification_issues:
        lines.append(f"Verification issues: {'; '.join(verification_issues)}")
    # Git diff stat for changed files
    if files_written or files_edited:
        try:
            diff_result = subprocess.run(
                ["git", "diff", "--stat"], capture_output=True, text=True,
                cwd=shared.PROJECT_DIR, timeout=5,
            )
            if diff_result.returncode == 0 and diff_result.stdout.strip():
                lines.append(f"\nGit diff stat:\n{diff_result.stdout.strip()}")
        except Exception:
            pass
    # Include truncated output summary
    if full_output:
        summary = full_output.strip()
        if len(summary) > 1500:
            summary = summary[:1500] + "\n... (truncated)"
        lines.append(f"\nAgent output:\n{summary}")
    return "\n".join(lines)


def _log_tool_call(agent, tool_name: str, args: dict, result: str, round_num: int):
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
    agent.tool_call_log.append({
        "tool": tool_name,
        "args": args_summary,
        "result": result_preview,
        "round": round_num,
    })
    # Keep only last 10 entries
    if len(agent.tool_call_log) > 10:
        agent.tool_call_log = agent.tool_call_log[-10:]


def _planning_prompt(task_desc: str) -> str:
    """Return the planning section of the system prompt, or simplified version for simple tasks."""
    if TaskComplexity.should_skip_planning(task_desc):
        return """EXECUTION MODE:
- This is a simple task. All tools are available immediately.
- Just do it: read what you need, make the change, verify it works.
- Use update_plan to track your steps if you want, but it's optional."""
    return """PLANNING PHASE (ENFORCED):
- You start in PLANNING mode. Only read-only tools + update_plan are available.
- Read the relevant code, understand the problem, then call update_plan with your steps.
- Include which files you will modify in your step descriptions.
- After update_plan is called with a complete plan, you will transition to EXECUTION mode and write tools become available.
- Do NOT try to use write_file, edit_file, run_shell, or git_commit during planning.

PLANNING (MANDATORY):
- Your VERY FIRST action must be calling update_plan to outline your work steps.
- Each step should be a short, concrete action (e.g. "Read app.py to understand layout", "Fix Container layout param", "Run app to verify fix").
- As you complete each step, call update_plan again with the updated statuses. Mark the current step "in-progress" and completed steps "done".
- The user monitors your plan in real-time. Keep it accurate."""


async def run_expert(name: str, task_desc: str, bus: SwarmBus | None = None, agent_name: str | None = None):
    expert_file = shared.EXPERTS_DIR / f"{name.lower()}.yaml"
    if not expert_file.exists():
        shared.console.print(f"[red]Expert {name} not found.[/red]")
        return ""
    data = yaml.safe_load(expert_file.read_text())
    expert_model = data.get("model") or shared.CODE_MODEL or shared.MODEL
    expert_temperature = data.get("temperature")
    max_rounds = data.get("max_rounds", EXPERT_DEFAULT_MAX_ROUNDS)
    display_name = agent_name or name
    shared.console.print(f"[bold cyan]-> Running Expert:[/bold cyan] {data['name']} ({display_name}) -- {data['mindset']}")

    agent = shared.state.register_agent(display_name, data['name'], task_desc)

    if not agent.check_budget():
        agent.transition(AgentState.PAUSED)
        msg = f"Agent '{display_name}' paused: over budget (tokens={agent.tokens_used}/{agent.token_budget}, cost=${agent.cost_usd:.4f}/${agent.cost_budget_usd:.4f})"
        shared.console.print(f"[swarm.warning]{msg}[/swarm.warning]")
        if bus:
            bus.post(display_name, msg, recipient=agent.parent or "*", kind="status")
        return msg

    if agent.pause_requested:
        agent.transition(AgentState.PAUSED)
        msg = f"Agent '{display_name}' paused by user request."
        shared.console.print(f"[yellow]{msg}[/yellow]")
        return msg

    # Auto-checkpoint dirty state before agent starts modifying files
    _auto_checkpoint_before_agent(display_name)

    agent.transition(AgentState.THINKING)

    prior_context = ""
    if bus:
        summary = bus.summary()
        if summary:
            prior_context = f"\n\n--- Prior agent outputs ---\n{summary}\n---\n"

    # Inject project context so the agent knows the codebase
    project_context = ""
    if shared.PROJECT_CONTEXT and shared.PROJECT_CONTEXT.get("project_name"):
        from grokswarm.context import format_context_for_prompt
        try:
            project_context = "\n" + format_context_for_prompt(shared.PROJECT_CONTEXT)
        except (KeyError, TypeError):
            pass

    # Detect tech stack for contextual guidance
    tech_stack = _detect_tech_stack()
    tech_stack_line = f"\nProject tech stack: {tech_stack}" if tech_stack else ""

    system_prompt = f"""You are {data['name']}, an expert with permanent mindset:
{data['mindset']}

Core objectives: {data.get('objectives', ['Execute efficiently'])}
{tech_stack_line}
Rules:
- Stay strictly in character.
- Respond concisely and directly. No emojis. No hype. No filler.
- Do not say "Done!", "Fixed!", "Enjoy!", "Perfect!" or similar. State what changed.
- Focus on the task, not the AI brand.
- Focus only on helping the user.
- USE your tools to actually create files and make changes. Call write_file, edit_file, run_shell directly.
- Do not just describe what to do — actually do it immediately.
- After writing code, run it (run_shell or run_tests) to verify it works. Do not claim it works without running it.
- Use capture_tui_screenshot or run_app_capture to verify visual/UI output when relevant.
- Use web_search when you genuinely need current or unknown information. For standard programming knowledge, use your built-in knowledge.
- Work fast: create files, write code, test, done. Minimize rounds.
- BEFORE you finish, you MUST run run_tests (or run_shell with appropriate test command) if you modified any code files. Never claim success without verified test results.

{_planning_prompt(task_desc)}{prior_context}{project_context}"""

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_desc},
    ]
    full_output = ""
    tool_actions: list[str] = []
    made_file_mutations = False
    ran_tests = False
    verification_prompted = False

    # -- Guardrails setup --
    loop_detector = LoopDetector()
    evidence_tracker = EvidenceTracker()
    lessons_db = LessonsDB()
    loop_escalation_count = 0
    model_escalated = False  # True when loop detector escalated to hardcore model
    loop_error_at_escalation: str = ""  # error sig when loop was detected (for lesson recording)

    # Complexity-based planning skip
    if TaskComplexity.should_skip_planning(task_desc):
        agent.phase = "executing"
        shared._log(f"agent {display_name}: simple task, skipping planning phase")

    # Dynamic tool granting: merge expert's static tools with task-inferred ones
    allowed_tools = ToolFilter.get_tools_for_expert(data)
    allowed_tools = DynamicTools.merge_tools(allowed_tools, task_desc)
    agent_tools = get_agent_tool_schemas(allowed_tools=allowed_tools)

    # Model routing per phase (4-tier system)
    planning_model = ToolFilter.get_model_for_phase(data, "planning")
    execution_model = ToolFilter.get_model_for_phase(data, "executing")
    escalation_model = ToolFilter.get_model_for_escalation()

    # Inject lessons from previous sessions
    relevant_lessons = lessons_db.find_relevant(files=[])
    if relevant_lessons:
        lesson_text = lessons_db.format_for_prompt(relevant_lessons)
        conversation.append({"role": "user", "content": lesson_text})

    shared.state.agent_mode += 1
    try:
        rounds_used = 0
        for _round in range(max_rounds):
            rounds_used = _round + 1

            # Compact conversation if it's getting large (same as main chat loop)
            conversation = await _trim_conversation(conversation)

            # Mid-run user interaction: check bus for nudge messages
            if bus:
                nudges = bus.read(display_name, since_id=0)
                for nudge in nudges:
                    if nudge["kind"] == "nudge" and nudge["sender"] != display_name:
                        conversation.append({
                            "role": "user",
                            "content": f"[USER GUIDANCE] {nudge['body']}",
                        })
                        shared._log(f"agent {display_name}: received nudge from {nudge['sender']}")

            agent.transition(AgentState.THINKING)

            # -- Goal anchoring: inject at 40% and 70% of max_rounds --
            pct = (_round + 1) / max_rounds
            if _round > 0 and abs(pct - 0.4) < (1.0 / max_rounds):
                conversation.append({
                    "role": "user",
                    "content": f'[SYSTEM] Checkpoint -- original goal: "{task_desc[:200]}". Are you on track? If not, adjust your plan.',
                })

            # -- Reflection prompt at ~80% of max_rounds (gives agent 20% remaining to act) --
            reflection_round = int(max_rounds * 0.8)
            if _round == reflection_round and _round > 2:
                conversation.append({
                    "role": "user",
                    "content": GoalVerifier.build_reflection_prompt(task_desc),
                })

            # Warn agent on final round so it can wrap up
            if _round == max_rounds - 1:
                conversation.append({
                    "role": "user",
                    "content": "[SYSTEM] This is your FINAL round. Wrap up now: summarize what you accomplished and what remains unfinished.",
                })

            # -- Stale read warnings every 5 rounds --
            if _round > 0 and _round % 5 == 0:
                stale_warnings = evidence_tracker.check_stale_reads()
                if stale_warnings:
                    conversation.append({
                        "role": "user",
                        "content": "[SYSTEM] " + " ".join(stale_warnings),
                    })

            # Select model based on phase + escalation state
            if model_escalated:
                round_model = escalation_model  # loop detector escalated to hardcore
            elif agent.phase == "planning":
                round_model = planning_model    # hardcore for planning
            else:
                round_model = execution_model   # expert's preferred tier

            # Track which model is used this round
            evidence_tracker.record_model(round_model)
            agent.current_model = round_model

            _api_kwargs = dict(
                model=round_model,
                messages=conversation,
                tools=agent_tools,
                max_tokens=shared.MAX_TOKENS,
            )
            if expert_temperature is not None:
                _api_kwargs["temperature"] = float(expert_temperature)
            response = await shared._api_call_with_retry(
                lambda: shared.client.chat.completions.create(**_api_kwargs),
                label=f"Expert:{data['name']}({round_model})"
            )
            if hasattr(response, 'usage') and response.usage:
                pt = response.usage.prompt_tokens or 0
                ct = response.usage.completion_tokens or 0
                cached = _extract_cached_tokens(response.usage)
                _record_usage(round_model, pt, ct, cached)
                agent.add_usage(pt, ct, round_model, cached)
                shared.state.global_tokens_used += pt + ct
                _inp_r, _cached_r, _out_r = shared._get_pricing(round_model)
                non_cached = max(0, pt - cached)
                shared.state.global_cost_usd += (
                    (non_cached / 1_000_000.0) * _inp_r
                    + (cached / 1_000_000.0) * _cached_r
                    + (ct / 1_000_000.0) * _out_r
                )

            # Track cached tokens on the agent
            if hasattr(response, 'usage') and response.usage:
                agent.cached_tokens_total += _extract_cached_tokens(response.usage)

            # -- Cost guard: check session spending --
            cost_delta = agent.cost_usd  # approximate
            _cost_guard.record_cost(cost_delta / max(rounds_used, 1))
            cost_actions = _cost_guard.check(shared.state.global_cost_usd)
            for action in cost_actions:
                if action.startswith("warn:"):
                    notify(f"COST WARNING: session spending passed {action[5:]}", level="warning")
                elif action == "pause_all":
                    notify(f"COST LIMIT: session budget ${_cost_guard.session_budget_usd:.2f} exceeded -- pausing all agents", level="error")
                    agent.transition(AgentState.PAUSED)
                    if bus:
                        bus.post(display_name, f"Session budget exceeded (${shared.state.global_cost_usd:.2f}/${_cost_guard.session_budget_usd:.2f})", kind="status")
                    break
                elif action.startswith("rate_alarm:"):
                    notify(f"COST RATE ALARM: spending {action[11:]} -- consider pausing agents", level="warning")
            if agent.state == AgentState.PAUSED:
                break

            if not agent.check_budget():
                agent.transition(AgentState.PAUSED)
                if bus:
                    bus.post(display_name, f"Agent over budget after round {_round + 1}", recipient="*", kind="status")
                break

            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                content = msg.content or ""
                full_output += content
                conversation.append({"role": "assistant", "content": content})

                # Verification gate: if agent made file mutations but never ran tests,
                # inject a prompt and give it one more round (once only)
                if made_file_mutations and not ran_tests and not verification_prompted and _round < max_rounds - 1:
                    verification_prompted = True
                    conversation.append({
                        "role": "user",
                        "content": "[SYSTEM] You modified code files but have not run tests yet. "
                                   "Run run_tests now to verify your changes work before finishing.",
                    })
                    shared._log(f"agent {display_name}: verification gate triggered (files changed, no tests run)")
                    continue  # give it another round

                break

            agent.transition(AgentState.WORKING)
            conversation.append({
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            if msg.content:
                full_output += msg.content + "\n"

            shared._log(f"agent {display_name}: round {_round + 1}/{max_rounds} - {len(msg.tool_calls)} tools")

            # Parse all tool calls first
            parsed_tools: list[tuple] = []  # (tc_dict, tool_name, args)
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    try:
                        args = json.loads(_repair_json(tc.function.arguments))
                    except json.JSONDecodeError:
                        conversation.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": "Error: invalid JSON arguments",
                        })
                        continue

                if tool_name == "update_plan":
                    args["_agent_name"] = display_name
                elif tool_name == "spawn_agent":
                    args["_parent"] = display_name

                # -- PlanGate: block tools during planning phase --
                gate_error = PlanGate.check_tool_allowed(agent, tool_name)
                if gate_error:
                    conversation.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": gate_error,
                    })
                    shared._log(f"agent {display_name}: PlanGate blocked {tool_name}")
                    continue

                tc_dict = {"id": tc.id, "name": tool_name}
                parsed_tools.append((tc_dict, tool_name, args))

                detail = _tool_detail(tool_name, args)
                shared._log(f"agent {display_name}: {tool_name}{detail}")
                tool_actions.append(f"{tool_name}{detail}")

                # Track file mutations and test runs
                if tool_name in ("write_file", "edit_file"):
                    made_file_mutations = True
                elif tool_name == "run_tests":
                    ran_tests = True

            # Execute tools — parallel for read-only, sequential otherwise
            all_read_only = all(n in READ_ONLY_TOOLS for _, n, _ in parsed_tools)
            can_parallelize = all_read_only and len(parsed_tools) > 1

            if can_parallelize:
                shared._log(f"agent {display_name}: parallel executing {len(parsed_tools)} read-only tools")

                async def _run_one(tc_d, t_name, t_args):
                    agent.current_tool = t_name
                    try:
                        result_str = await _execute_tool(t_name, t_args)
                    except Exception as e:
                        result_str = f"Error: {e}"
                    if len(result_str) > MAX_TOOL_RESULT_SIZE:
                        result_str = result_str[:MAX_TOOL_RESULT_SIZE] + "\n... (truncated)"
                    return tc_d, t_name, t_args, result_str

                results = await asyncio.gather(*[_run_one(tc_d, tn, ta) for tc_d, tn, ta in parsed_tools])
                for tc_d, t_name, t_args, result_str in results:
                    # -- Plan deviation warning --
                    deviation = PlanGate.check_plan_deviation(agent, t_name, t_args)
                    if deviation:
                        result_str = deviation + "\n" + result_str

                    # -- Record for loop detection, evidence tracking, and live log --
                    loop_detector.record_tool_call(t_name, t_args, result_str)
                    evidence_tracker.record_tool(rounds_used, t_name, t_args, result_str)
                    _log_tool_call(agent, t_name, t_args, result_str, rounds_used)

                    conversation.append({
                        "role": "tool", "tool_call_id": tc_d["id"],
                        "content": result_str,
                    })
            else:
                for tc_d, tool_name, args in parsed_tools:
                    agent.current_tool = tool_name
                    result_str = await _execute_tool(tool_name, args)
                    if len(result_str) > MAX_TOOL_RESULT_SIZE:
                        result_str = result_str[:MAX_TOOL_RESULT_SIZE] + "\n... (truncated)"

                    # -- Plan deviation warning --
                    deviation = PlanGate.check_plan_deviation(agent, tool_name, args)
                    if deviation:
                        result_str = deviation + "\n" + result_str

                    # -- Record for loop detection, evidence tracking, and live log --
                    loop_detector.record_tool_call(tool_name, args, result_str)
                    evidence_tracker.record_tool(rounds_used, tool_name, args, result_str)
                    _log_tool_call(agent, tool_name, args, result_str, rounds_used)

                    conversation.append({
                        "role": "tool", "tool_call_id": tc_d["id"],
                        "content": result_str,
                    })

                    # -- PlanGate: auto-transition after update_plan with ready plan --
                    if tool_name == "update_plan" and agent.phase == "planning":
                        if PlanGate.check_plan_ready(agent):
                            # In trust/agent mode: auto-transition
                            if shared.state.trust_mode or shared.state.agent_mode > 1:
                                PlanGate.transition_to_executing(agent)
                                notify(f"[{display_name}] Plan approved -- executing ({len(agent.plan)} steps)")
                                shared._log(f"agent {display_name}: auto-transitioned to executing phase")
                            else:
                                # Post plan for user review and auto-approve
                                # The user can still /reject to revert and send feedback
                                plan_text = "\n".join(f"  {i+1}. {s['step']}" for i, s in enumerate(agent.plan))
                                if bus:
                                    bus.post(display_name, f"Plan ready:\n{plan_text}", kind="plan")
                                notify(f"[{display_name}] Plan: {len(agent.plan)} steps -- /reject {display_name} <feedback> to revise")
                                PlanGate.transition_to_executing(agent)
                                shared._log(f"agent {display_name}: plan posted, auto-transitioned to executing")

            agent.current_tool = None

            # -- Loop detection after each round --
            loop_msg = loop_detector.check_loop()
            if loop_msg:
                loop_escalation_count += 1
                shared._log(f"agent {display_name}: loop detected (escalation #{loop_escalation_count})")
                notify(f"WARNING: {display_name} stuck in loop -- use /tell or /peek", level="warning")
                shared.console.print(f"[bold yellow]WARNING: {display_name} loop detected (escalation #{loop_escalation_count})[/bold yellow]")

                if loop_escalation_count >= 3:
                    # Pause agent after 3 loop escalations
                    agent.transition(AgentState.PAUSED)
                    pause_msg = f"Agent {display_name} paused: stuck in loop after 3 escalations. Use /tell {display_name} to provide guidance or /resume {display_name}."
                    if bus:
                        bus.post(display_name, pause_msg, kind="status")
                    notify(pause_msg, level="error")
                    shared.console.print(f"[bold red]{pause_msg}[/bold red]")
                    break
                elif loop_escalation_count == 1:
                    # First escalation: upgrade model to hardcore for better reasoning
                    model_escalated = True
                    # Record the error for potential lesson learning
                    if loop_detector.test_failures:
                        loop_error_at_escalation = loop_detector.test_failures[-1]
                    elif loop_detector.edit_targets:
                        loop_error_at_escalation = f"repeated edits to {list(loop_detector.edit_targets.keys())[-1]}"
                    # Check lessons DB for relevant prior solutions
                    relevant = lessons_db.find_relevant(
                        error_signature=loop_error_at_escalation,
                        files=list(loop_detector.edit_targets.keys()),
                    )
                    lesson_hint = ""
                    if relevant:
                        lesson_hint = "\n\n" + lessons_db.format_for_prompt(relevant)
                    shared._log(f"agent {display_name}: escalating to hardcore model after loop detection")
                    notify(f"[{display_name}] Escalating to hardcore model after loop detection", level="warning")
                    conversation.append({
                        "role": "user",
                        "content": f"[SYSTEM] {loop_msg}\n\n[MODEL ESCALATED] You are now running on a more powerful reasoning model. Use this opportunity to think more carefully about the problem.{lesson_hint}",
                    })
                else:
                    # Second escalation: warn harder
                    conversation.append({
                        "role": "user",
                        "content": f"[SYSTEM] {loop_msg}\n\n[FINAL WARNING] One more loop and you will be paused. Ask for help via send_message if needed.",
                    })

            # -- Milestone notifications: plan step completion --
            if agent.plan:
                done_count = sum(1 for s in agent.plan if s["status"] == "done")
                total_count = len(agent.plan)
                # Find recently completed steps
                for step in agent.plan:
                    if step["status"] == "done":
                        step_key = f"_notified_{step['step'][:30]}"
                        if not hasattr(agent, step_key):
                            setattr(agent, step_key, True)
                            notify(f"[{display_name}] completed: {step['step'][:60]} ({done_count}/{total_count})")

        # -- Post-completion: GoalVerifier validation --
        verification_result = GoalVerifier.validate_completion(agent, tool_actions, full_output)
        verification_issues = verification_result.get("issues", [])

        if verification_issues and rounds_used < max_rounds:
            # Give agent 2 more rounds to address issues
            issue_text = "; ".join(verification_issues)
            conversation.append({
                "role": "user",
                "content": f"[SYSTEM] Completion verification found issues: {issue_text}. Address these now.",
            })
            shared._log(f"agent {display_name}: verification issues, giving extra rounds: {issue_text}")
            for _extra_round in range(min(2, max_rounds - rounds_used)):
                rounds_used += 1
                response = await shared._api_call_with_retry(
                    lambda: shared.client.chat.completions.create(
                        model=execution_model, messages=conversation,
                        tools=agent_tools, max_tokens=shared.MAX_TOKENS,
                    ),
                    label=f"Expert:{data['name']}:verification"
                )
                if hasattr(response, 'usage') and response.usage:
                    pt = response.usage.prompt_tokens or 0
                    ct = response.usage.completion_tokens or 0
                    cached = _extract_cached_tokens(response.usage)
                    _record_usage(execution_model, pt, ct, cached)
                    agent.add_usage(pt, ct, execution_model, cached)
                choice = response.choices[0]
                msg = choice.message
                if msg.content:
                    full_output += msg.content + "\n"
                    conversation.append({"role": "assistant", "content": msg.content})
                if msg.tool_calls:
                    conversation.append({
                        "role": "assistant", "content": msg.content or None,
                        "tool_calls": [
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                            for tc in msg.tool_calls
                        ],
                    })
                    for tc in msg.tool_calls:
                        try:
                            tc_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            tc_args = {}
                        if tc.function.name == "update_plan":
                            tc_args["_agent_name"] = display_name
                        result_str = await _execute_tool(tc.function.name, tc_args)
                        if len(result_str) > MAX_TOOL_RESULT_SIZE:
                            result_str = result_str[:MAX_TOOL_RESULT_SIZE] + "\n... (truncated)"
                        conversation.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                        tool_actions.append(f"{tc.function.name}")
                        if tc.function.name == "run_tests":
                            ran_tests = True
                else:
                    break
            # Re-validate
            verification_result = GoalVerifier.validate_completion(agent, tool_actions, full_output)
            verification_issues = verification_result.get("issues", [])

        # -- Post-completion smoke test --
        # Only run if mutations were made, tests weren't run, and we're not inside pytest
        if made_file_mutations and not ran_tests and "pytest" not in sys.modules:
            shared._log(f"agent {display_name}: post-completion smoke test (files mutated, no tests)")
            try:
                from grokswarm.tools_test import run_tests as _run_tests_fn
                test_output = _run_tests_fn(None, None)
                if "[FAIL]" in test_output:
                    verification_issues.append("[REGRESSION WARNING] Post-completion test run failed")
                    notify(f"[{display_name}] REGRESSION WARNING: post-completion tests failed", level="warning")
            except Exception:
                pass

        # Evidence summary
        ev_summary = evidence_tracker.get_evidence_summary()

        # Structured completion report
        report = _build_completion_report(
            display_name, tool_actions, rounds_used, max_rounds, full_output,
            evidence_summary=ev_summary, verification_issues=verification_issues if verification_issues else None,
        )
        save_memory(f"expert_{display_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", report)
        if bus:
            bus.post(display_name, report, kind="result")
        agent.transition(AgentState.DONE)

        # -- Cross-session learning: record lesson if we recovered from a loop --
        if loop_escalation_count > 0 and ev_summary.get("last_test_status") == "PASS":
            # Agent hit a loop but eventually succeeded — record what worked
            fix_desc = full_output[-500:] if full_output else "Unknown fix"
            files_involved = list(loop_detector.edit_targets.keys())
            try:
                lessons_db.record_lesson(
                    error_signature=loop_error_at_escalation,
                    fix_description=fix_desc,
                    files_involved=files_involved,
                    expert=data['name'],
                )
                shared._log(f"agent {display_name}: recorded lesson for '{loop_error_at_escalation[:50]}'")
            except Exception:
                pass  # best-effort

        # Completion notification
        test_status = ev_summary.get("last_test_status", "never_run")
        models_info = ", ".join(f"{m}x{c}" for m, c in ev_summary.get("models_used", {}).items())
        notify(f"[{display_name}] DONE -- {ev_summary.get('files_written', 0)} files changed, tests {test_status}, ${agent.cost_usd:.4f}, {rounds_used} rounds, models: {models_info or 'none'}")

        return full_output
    except Exception as e:
        agent.transition(AgentState.ERROR)
        shared.console.print(f"[swarm.error]Expert {data['name']} ({display_name}) API error: {e}[/swarm.error]")
        return f"Error: {e}"
    finally:
        shared.state.agent_mode -= 1


# -- Spawning & Messaging Implementation --
async def _spawn_agent_impl(expert: str, task: str, name: str | None = None,
                            token_budget: int = 0, cost_budget: float = 0.0,
                            parent: str | None = None) -> str:
    expert_file = shared.EXPERTS_DIR / f"{expert.lower()}.yaml"
    if not expert_file.exists():
        return f"Error: expert profile '{expert}' not found. Available: {list_experts()}"

    if name is None:
        shared._agent_counter += 1
        name = f"{expert}_{shared._agent_counter}"

    if name in shared.state.agents:
        return f"Error: agent '{name}' already exists. Choose a different name."

    agent = shared.state.register_agent(name, expert, task,
                                  token_budget=token_budget,
                                  cost_budget_usd=cost_budget,
                                  parent=parent)

    async def _agent_task():
        try:
            result = await run_expert(expert, task, bus=get_bus(), agent_name=name)
            get_bus().post(name, result or "Task completed (no output).", recipient="*", kind="result")
        except Exception as e:
            agent = shared.state.get_agent(name)
            if agent:
                agent.transition(AgentState.ERROR)
            get_bus().post(name, f"Agent error: {e}", recipient="*", kind="error")

    try:
        bg_task = asyncio.create_task(_agent_task())
        shared._background_tasks[name] = bg_task
        return f"Agent '{name}' spawned with expert profile '{expert}'. Task: {task[:100]}..."
    except RuntimeError:
        await run_expert(expert, task, bus=get_bus(), agent_name=name)
        return f"Agent '{name}' ran synchronously (no event loop for background task)."


def _send_message_impl(sender: str, to: str, body: str, kind: str = "request") -> str:
    bus = get_bus()
    bus.post(sender, body, recipient=to, kind=kind)
    return f"Message sent to '{to}' (kind={kind})."


def _check_messages_impl(agent_name: str = "*", since_id: int = 0) -> str:
    bus = get_bus()
    msgs = bus.read(agent_name, since_id=since_id)
    if not msgs:
        return "No new messages."
    lines = []
    for m in msgs:
        lines.append(f"[{m['id']}] {m['ts']} {m['sender']}\u2192{m['recipient']} ({m['kind']}): {m['body'][:500]}")
    return "\n".join(lines)


def _list_agents_impl() -> str:
    if not shared.state.agents:
        return "No active agents."
    lines = [f"Global: tokens={shared.state.global_tokens_used}, cost=${shared.state.global_cost_usd:.4f}"]
    for name, agent in shared.state.agents.items():
        bg_status = ""
        if name in shared._background_tasks:
            task = shared._background_tasks[name]
            bg_status = " [background: " + ("done" if task.done() else "running") + "]"
        budget_str = ""
        if agent.token_budget > 0:
            budget_str += f" budget={agent.tokens_used}/{agent.token_budget}tok"
        if agent.cost_budget_usd > 0:
            budget_str += f" ${agent.cost_usd:.4f}/${agent.cost_budget_usd:.4f}"
        lines.append(f"  {name} ({agent.expert}): state={agent.state.value}, tokens={agent.tokens_used}, cost=${agent.cost_usd:.4f}{budget_str}{bg_status}")
    return "Active agents:\n" + "\n".join(lines)


# Register spawning/messaging in tool dispatch
TOOL_DISPATCH["spawn_agent"] = lambda args: _spawn_agent_impl(
    args["expert"], args["task"], args.get("name"),
    token_budget=args.get("token_budget", 0),
    cost_budget=args.get("cost_budget", 0.0),
    parent=args.get("_parent")
)
TOOL_DISPATCH["send_message"] = lambda args: _send_message_impl("user", args["to"], args["body"], args.get("kind", "request"))
TOOL_DISPATCH["check_messages"] = lambda args: _check_messages_impl(args.get("agent_name", "*"), args.get("since_id", 0))
TOOL_DISPATCH["list_agents"] = lambda args: _list_agents_impl()
