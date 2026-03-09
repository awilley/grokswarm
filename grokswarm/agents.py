"""SwarmBus, spawning, messaging, run_supervisor, run_expert, cost tracking."""

import json
import os
import sys
import time
import yaml
import sqlite3
import asyncio
import threading
from datetime import datetime
from pathlib import Path

import subprocess

# Lock for thread-safe cost/token updates when multiple agents run concurrently
_cost_lock = threading.Lock()

import grokswarm.shared as shared
from grokswarm.models import AgentState
from grokswarm.tools_registry import TOOL_DISPATCH, READ_ONLY_TOOLS, get_agent_tool_schemas
from grokswarm.registry_helpers import list_experts, save_memory
from grokswarm.engine import _execute_tool, _repair_json, _trim_conversation, _tool_detail, MAX_TOOL_RESULT_SIZE
from grokswarm.guardrails import (
    PlanGate, GoalVerifier, LoopDetector, EvidenceTracker,
    ToolFilter, Orchestrator, notify, _auto_print,
    TaskComplexity, LessonsDB, CostGuard, DynamicTools,
    GuardrailPipeline,
)


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
    non_cached = max(0, prompt_tokens - cached_tokens)
    cost = (
        (non_cached / 1_000_000.0) * inp_rate
        + (cached_tokens / 1_000_000.0) * cached_rate
        + (completion_tokens / 1_000_000.0) * out_rate
    )
    with _cost_lock:
        shared.state.project_prompt_tokens += prompt_tokens
        shared.state.project_completion_tokens += completion_tokens
        shared.state.project_cached_tokens += cached_tokens
        shared.state.project_cost_usd += cost
        shared.state.global_tokens_used += prompt_tokens + completion_tokens
        shared.state.global_cost_usd += cost
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
        from grokswarm import llm
        chat = llm.create_chat(shared.MODEL, response_format="json_object")
        llm.populate_chat(chat, [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}])
        response = await shared._api_call_with_retry(
            lambda: chat.sample(),
            label="Supervisor"
        )
        usage = llm.extract_usage(response)
        if usage["prompt_tokens"] or usage["completion_tokens"]:
            _record_usage(shared.MODEL, usage["prompt_tokens"], usage["completion_tokens"],
                          usage["cached_tokens"])
        plan = json.loads((response.content or "").strip())
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
            ["git", "add", "-u"],  # Only tracked files — avoids committing .env/secrets
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
                cwd=shared.get_project_dir(), timeout=5,
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


def _validate_expert_yaml(data: dict, filepath: str) -> str | None:
    """Validate expert YAML structure. Returns error message or None if valid."""
    if not isinstance(data, dict):
        return f"Expert file {filepath}: expected a YAML mapping, got {type(data).__name__}"
    missing = [f for f in ("name", "mindset") if f not in data]
    if missing:
        return f"Expert file {filepath}: missing required field(s): {', '.join(missing)}"
    if "temperature" in data and not isinstance(data["temperature"], (int, float)):
        return f"Expert file {filepath}: 'temperature' must be a number, got {type(data['temperature']).__name__}"
    if "max_rounds" in data and not isinstance(data["max_rounds"], int):
        return f"Expert file {filepath}: 'max_rounds' must be an integer, got {type(data['max_rounds']).__name__}"
    return None


async def run_expert(name: str, task_desc: str, bus: SwarmBus | None = None,
                     agent_name: str | None = None, workspace_dir: Path | None = None,
                     is_sub_agent: bool = False):
    expert_file = shared.EXPERTS_DIR / f"{name.lower()}.yaml"
    if not expert_file.exists():
        shared.console.print(f"[red]Expert {name} not found.[/red]")
        return ""
    data = yaml.safe_load(expert_file.read_text())
    err = _validate_expert_yaml(data, str(expert_file))
    if err:
        shared.console.print(f"[red]{err}[/red]")
        return ""
    expert_temperature = data.get("temperature")
    max_rounds = data.get("max_rounds", EXPERT_DEFAULT_MAX_ROUNDS)
    display_name = agent_name or name

    # Set workspace override for branch-isolated agents
    if workspace_dir:
        shared._workspace_override.set(workspace_dir)
        shared.console.print(f"[bold cyan]-> Running Expert:[/bold cyan] {data['name']} ({display_name}) in worktree {workspace_dir.name} -- {data['mindset']}")
    else:
        shared.console.print(f"[bold cyan]-> Running Expert:[/bold cyan] {data['name']} ({display_name}) -- {data['mindset']}")

    agent = shared.state.register_agent(display_name, data['name'], task_desc)
    agent.workspace = workspace_dir

    if not agent.check_budget():
        agent.transition(AgentState.PAUSED)
        msg = f"Agent '{display_name}' paused: over budget"
        shared.console.print(f"[swarm.warning]{msg}[/swarm.warning]")
        if bus:
            bus.post(display_name, msg, recipient=agent.parent or "*", kind="status")
        return msg

    if agent.pause_requested:
        agent.transition(AgentState.PAUSED)
        return f"Agent '{display_name}' paused by user request."

    _auto_checkpoint_before_agent(display_name)
    agent.transition(AgentState.THINKING)
    _agent_start_time = time.monotonic()

    # -- Build system prompt --
    prior_context = ""
    if bus:
        summary = bus.summary()
        if summary:
            prior_context = f"\n\n--- Prior agent outputs ---\n{summary}\n---\n"

    project_context = ""
    if shared.PROJECT_CONTEXT and shared.PROJECT_CONTEXT.get("project_name"):
        from grokswarm.context import format_context_for_prompt
        try:
            project_context = "\n" + format_context_for_prompt(shared.PROJECT_CONTEXT)
        except (KeyError, TypeError):
            pass

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
- Focus only on helping the user.
- USE your tools to actually create files and make changes. Call write_file, edit_file, run_shell directly.
- Do not just describe what to do — actually do it immediately.
- After writing code, run it (run_shell or run_tests) to verify it works. Do not claim it works without running it.
- Use capture_tui_screenshot or run_app_capture to verify visual/UI output when relevant.
- Use web_search when you genuinely need current or unknown information.
- Work fast: create files, write code, test, done. Minimize rounds.
- BEFORE you finish, you MUST run run_tests if you modified any code files.
- If a subtask is clearly outside your expertise (e.g. you are a coder asked to do deep research), use spawn_agent to delegate to the right expert, then wait_for_agent to get their results before proceeding.

{_planning_prompt(task_desc)}{prior_context}{project_context}"""

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_desc},
    ]
    full_output = ""
    tool_actions: list[str] = []

    # -- Initialize guardrail pipeline (all guardrail state lives here) --
    gp = GuardrailPipeline(agent, display_name, task_desc, data, bus,
                           is_sub_agent=is_sub_agent)
    gp.setup(conversation)
    agent_tools = gp.get_tool_schemas()

    shared.state.agent_mode += 1
    try:
        rounds_used = 0
        for _round in range(max_rounds):
            rounds_used = _round + 1

            conversation = await _trim_conversation(conversation)

            # Check bus for user nudges
            if bus:
                for nudge in bus.read(display_name, since_id=0):
                    if nudge["kind"] == "nudge" and nudge["sender"] != display_name:
                        conversation.append({"role": "user", "content": f"[USER GUIDANCE] {nudge['body']}"})

            agent.transition(AgentState.THINKING)

            # Guardrail: inject round-appropriate system messages
            gp.on_round_start(_round, max_rounds, conversation)

            # Select model (handles phase routing + escalation)
            round_model = gp.select_model(_round)

            # -- API call --
            from grokswarm import llm
            _xai_tools = llm.convert_tools(agent_tools)
            _chat_kwargs: dict = dict(model=round_model, tools=_xai_tools, max_tokens=shared.MAX_TOKENS)
            if expert_temperature is not None:
                _chat_kwargs["temperature"] = float(expert_temperature)
            _chat = llm.create_chat(**_chat_kwargs)
            llm.populate_chat(_chat, conversation)
            response = await shared._api_call_with_retry(
                lambda: _chat.sample(),
                label=f"Expert:{data['name']}({round_model})"
            )

            # -- Record usage --
            _usage = llm.extract_usage(response)
            pt = _usage["prompt_tokens"]
            ct = _usage["completion_tokens"]
            cached = _usage["cached_tokens"]
            if pt or ct:
                _record_usage(round_model, pt, ct, cached)
                agent.add_usage(pt, ct, round_model, cached)
                agent.cached_tokens_total += cached

            # Guardrail: cost limits
            if gp.check_cost_limits(rounds_used):
                break

            tool_calls = llm.response_tool_calls(response)

            # -- No tool calls: agent is done (or needs verification gate) --
            if not tool_calls:
                content = response.content or ""
                full_output += content
                conversation.append({"role": "assistant", "content": content})
                if gp.check_verification_gate(_round, max_rounds, conversation):
                    continue
                break

            # -- Process tool calls --
            agent.transition(AgentState.WORKING)
            conversation.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [llm.tool_call_to_dict(tc) for tc in tool_calls],
            })
            if response.content:
                full_output += response.content + "\n"

            shared._log(f"agent {display_name}: round {_round + 1}/{max_rounds} - {len(tool_calls)} tools")

            # Parse tool calls
            parsed_tools: list[tuple] = []
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    try:
                        args = json.loads(_repair_json(tc.function.arguments))
                    except json.JSONDecodeError:
                        conversation.append({"role": "tool", "tool_call_id": tc.id, "content": "Error: invalid JSON arguments"})
                        continue

                if tool_name == "update_plan":
                    args["_agent_name"] = display_name
                elif tool_name == "spawn_agent":
                    args["_parent"] = display_name

                # Guardrail: PlanGate check
                gate_error = gp.check_tool(tool_name, args)
                if gate_error:
                    conversation.append({"role": "tool", "tool_call_id": tc.id, "content": gate_error})
                    continue

                parsed_tools.append(({"id": tc.id, "name": tool_name}, tool_name, args))
                detail = _tool_detail(tool_name, args)
                shared._log(f"agent {display_name}: {tool_name}{detail}")
                tool_actions.append(f"{tool_name}{detail}")

            # Execute tools — parallel for read-only, sequential otherwise
            all_read_only = all(n in READ_ONLY_TOOLS for _, n, _ in parsed_tools)
            if all_read_only and len(parsed_tools) > 1:
                async def _run_one(tc_d, t_name, t_args):
                    agent.current_tool = t_name
                    try:
                        r = await _execute_tool(t_name, t_args)
                    except Exception as e:
                        r = f"Error: {e}"
                    if len(r) > MAX_TOOL_RESULT_SIZE:
                        r = r[:MAX_TOOL_RESULT_SIZE] + "\n... (truncated)"
                    return tc_d, t_name, t_args, r

                results = await asyncio.gather(*[_run_one(d, n, a) for d, n, a in parsed_tools])
                for tc_d, t_name, t_args, result_str in results:
                    deviation = gp.on_tool_result(t_name, t_args, result_str, rounds_used)
                    if deviation:
                        result_str = deviation + "\n" + result_str
                    conversation.append({"role": "tool", "tool_call_id": tc_d["id"], "content": result_str})
            else:
                for tc_d, tool_name, args in parsed_tools:
                    agent.current_tool = tool_name
                    result_str = await _execute_tool(tool_name, args)
                    if len(result_str) > MAX_TOOL_RESULT_SIZE:
                        result_str = result_str[:MAX_TOOL_RESULT_SIZE] + "\n... (truncated)"
                    deviation = gp.on_tool_result(tool_name, args, result_str, rounds_used)
                    if deviation:
                        result_str = deviation + "\n" + result_str
                    conversation.append({"role": "tool", "tool_call_id": tc_d["id"], "content": result_str})

            agent.current_tool = None

            # Guardrail: loop detection + milestone notifications
            if gp.on_round_end(_round, conversation):
                break

        # -- Post-completion: verification + evidence + report --
        ev_summary, verification_issues = gp.on_completion(tool_actions, full_output, rounds_used, max_rounds)

        if verification_issues and rounds_used < max_rounds:
            issue_text = "; ".join(verification_issues)
            conversation.append({"role": "user", "content": f"[SYSTEM] Completion verification found issues: {issue_text}. Address these now."})
            shared._log(f"agent {display_name}: verification issues, giving extra rounds: {issue_text}")
            execution_model = gp.execution_model
            _v_xai_tools = llm.convert_tools(agent_tools)
            for _extra in range(min(2, max_rounds - rounds_used)):
                rounds_used += 1
                _v_chat = llm.create_chat(execution_model, tools=_v_xai_tools, max_tokens=shared.MAX_TOKENS)
                llm.populate_chat(_v_chat, conversation)
                response = await shared._api_call_with_retry(
                    lambda: _v_chat.sample(),
                    label=f"Expert:{data['name']}:verification"
                )
                _v_usage = llm.extract_usage(response)
                if _v_usage["prompt_tokens"] or _v_usage["completion_tokens"]:
                    _record_usage(execution_model, _v_usage["prompt_tokens"], _v_usage["completion_tokens"],
                                  _v_usage["cached_tokens"])
                    agent.add_usage(_v_usage["prompt_tokens"], _v_usage["completion_tokens"],
                                    execution_model, _v_usage["cached_tokens"])
                v_tool_calls = llm.response_tool_calls(response)
                if response.content:
                    full_output += response.content + "\n"
                    conversation.append({"role": "assistant", "content": response.content})
                if v_tool_calls:
                    conversation.append({
                        "role": "assistant", "content": response.content or None,
                        "tool_calls": [llm.tool_call_to_dict(tc) for tc in v_tool_calls],
                    })
                    for tc in v_tool_calls:
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
                        tool_actions.append(tc.function.name)
                        if tc.function.name == "run_tests":
                            gp.ran_tests = True
                        gp.on_tool_result(tc.function.name, tc_args, result_str, rounds_used)
                    if gp.on_round_end(rounds_used, conversation):
                        break
                else:
                    break
            ev_summary, verification_issues = gp.on_completion(tool_actions, full_output, rounds_used, max_rounds)

        # Build and post completion report
        report = _build_completion_report(
            display_name, tool_actions, rounds_used, max_rounds, full_output,
            evidence_summary=ev_summary, verification_issues=verification_issues if verification_issues else None)
        save_memory(f"expert_{display_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", report)
        if bus:
            bus.post(display_name, report, kind="result")
        agent.transition(AgentState.DONE)
        # Terminal bell if agent ran >30s in interactive mode
        if (shared.state.agent_mode <= 1
                and time.monotonic() - _agent_start_time > 30):
            try:
                sys.__stdout__.write("\a")
                sys.__stdout__.flush()
            except Exception:
                pass

        # Record completion in project intelligence
        try:
            from grokswarm.guardrails import LessonsDB
            db = LessonsDB()
            files_mod = list({a["args"].split(",")[0].strip().strip("'\"")
                             for a in tool_actions
                             if a["tool"] in ("edit_file", "write_file") and a.get("args")})[:10]
            tools_used = list({a["tool"] for a in tool_actions})[:10]
            db.record_completion(task_description, files_mod, tools_used, expert=display_name)
        except Exception:
            pass

        return full_output
    except Exception as e:
        agent.transition(AgentState.ERROR)
        shared.console.print(f"[swarm.error]Expert {data['name']} ({display_name}) API error: {e}[/swarm.error]")
        try:
            from grokswarm.bugs import log_exception
            log_exception(e, context_label=f"Expert:{data['name']}({display_name})")
        except Exception:
            pass
        return f"Error: {e}"
    finally:
        shared.state.agent_mode -= 1


_CLAUDE_ENV_STRIP = {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ACCESS_TOKEN"}


async def run_claude_expert(task_desc: str, bus: SwarmBus | None = None,
                            agent_name: str | None = None,
                            workspace_dir: Path | None = None,
                            expert_name: str = "claude",
                            is_sub_agent: bool = False,
                            timeout: int = 300,
                            max_budget_usd: float = 2.0):
    """Execute an agent task through Claude Code CLI instead of the internal tool loop."""
    display_name = agent_name or expert_name

    # Set workspace override for branch-isolated agents
    if workspace_dir:
        shared._workspace_override.set(workspace_dir)
        shared.console.print(f"[bold magenta]-> Running Claude Expert:[/bold magenta] {display_name} in worktree {workspace_dir.name}")
    else:
        shared.console.print(f"[bold magenta]-> Running Claude Expert:[/bold magenta] {display_name}")

    agent = shared.state.register_agent(display_name, expert_name, task_desc)
    agent.workspace = workspace_dir
    shared.state.agent_mode += 1

    _auto_checkpoint_before_agent(display_name)
    agent.transition(AgentState.THINKING)
    _agent_start_time = time.monotonic()

    try:
        cwd = str(workspace_dir) if workspace_dir else str(shared.PROJECT_DIR)
        cmd = [
            "claude", "-p",
            "--output-format", "json",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--max-turns", "25",
            f"--max-budget-usd", str(max_budget_usd),
            task_desc,
        ]
        env = {k: v for k, v in os.environ.items() if k not in _CLAUDE_ENV_STRIP}

        agent.transition(AgentState.WORKING)
        loop = asyncio.get_event_loop()
        proc_result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=timeout, env=env, cwd=cwd),
        )

        if proc_result.returncode != 0:
            err = proc_result.stderr.strip() or proc_result.stdout.strip() or "unknown"
            agent.transition(AgentState.ERROR)
            report = f"Claude CLI error (exit {proc_result.returncode}): {err}"
            if bus:
                bus.post(display_name, report, kind="result")
            return report

        # Parse JSON output
        try:
            data = json.loads(proc_result.stdout)
        except json.JSONDecodeError:
            # Plain text fallback
            result_text = proc_result.stdout
            data = {}
        else:
            result_text = data.get("result", "")
            if isinstance(result_text, list):
                result_text = "\n".join(
                    block.get("text", "") for block in result_text
                    if isinstance(block, dict) and block.get("type") == "text"
                )

        # Track cost
        cost = data.get("cost_usd") or data.get("costUsd") or 0
        turns = data.get("num_turns") or data.get("numTurns") or 0
        if cost:
            with _cost_lock:
                shared.state.project_cost_usd += cost
                shared.state.global_cost_usd += cost
            agent.cost_usd = cost

        elapsed = time.monotonic() - _agent_start_time
        report = (
            f"[Claude Expert: {display_name}] Completed in {elapsed:.1f}s, "
            f"{turns} turn(s), ${cost:.4f}\n\n{result_text}"
        )

        # Save memory & post result
        save_memory(f"claude_{display_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", report[:2000])
        if bus:
            bus.post(display_name, report, kind="result")
        agent.transition(AgentState.DONE)

        # Terminal bell if agent ran >30s in interactive mode
        if shared.state.agent_mode <= 1 and elapsed > 30:
            try:
                sys.__stdout__.write("\a")
                sys.__stdout__.flush()
            except Exception:
                pass

        return result_text

    except subprocess.TimeoutExpired:
        agent.transition(AgentState.ERROR)
        report = f"Claude CLI timed out after {timeout}s"
        if bus:
            bus.post(display_name, report, kind="result")
        return report
    except Exception as e:
        agent.transition(AgentState.ERROR)
        shared.console.print(f"[swarm.error]Claude Expert {display_name} error: {e}[/swarm.error]")
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
                from grokswarm.bugs import log_exception
                log_exception(e, context_label=f"BackgroundAgent:{name}")
            except Exception:
                pass

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


async def _wait_for_agent_impl(name: str, timeout_sec: int = 300) -> str:
    """Block until a spawned agent completes and return its output."""
    task = shared._background_tasks.get(name)
    if task is None:
        # Agent might have already finished and been cleaned up
        agent = shared.state.get_agent(name)
        if agent is None:
            return f"No agent named '{name}' found. Use list_agents to see available agents."
        # Already done — read its messages
    else:
        try:
            await asyncio.wait_for(task, timeout=timeout_sec)
        except asyncio.TimeoutError:
            return f"Agent '{name}' did not finish within {timeout_sec}s. It is still running. Check back later with check_messages."
        except Exception as e:
            return f"Agent '{name}' encountered an error: {e}"

    # Collect results from bus
    bus = get_bus()
    msgs = bus.read("*")
    agent_msgs = [m for m in msgs if m["sender"] == name and m["kind"] in ("result", "error")]
    if agent_msgs:
        return "\n".join(m["body"][:2000] for m in agent_msgs[-3:])

    # Fallback: check agent state
    agent = shared.state.get_agent(name)
    if agent:
        return f"Agent '{name}' finished (state={agent.state.value}) but posted no result messages."
    return f"Agent '{name}' completed."


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
TOOL_DISPATCH["wait_for_agent"] = lambda args: _wait_for_agent_impl(args["name"], args.get("timeout", 300))


# ---------------------------------------------------------------------------
# Subprocess-based agent execution (multi-process parallelism)
# ---------------------------------------------------------------------------

def spawn_expert_subprocess(
    expert: str,
    task_desc: str,
    *,
    name: str | None = None,
    timeout: int = 600,
) -> subprocess.Popen:
    """Spawn an expert in a separate Python process.

    Uses the same SwarmBus (SQLite) for coordination, so results
    are visible to other agents and the REPL dashboard.

    Returns the Popen object for monitoring.
    """
    display_name = name or f"worker_{expert}"
    cmd = [
        sys.executable, "-m", "grokswarm.commands",
        "expert", expert, task_desc,
    ]
    env = dict(__import__("os").environ)
    env["GROKSWARM_AGENT_NAME"] = display_name
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(shared.PROJECT_DIR),
    )
    shared._log(f"spawned subprocess worker: {display_name} (pid={proc.pid})")
    bus = get_bus()
    bus.post(display_name, f"Subprocess worker started for: {task_desc[:100]}", kind="status")
    return proc
