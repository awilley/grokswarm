"""SwarmBus, spawning, messaging, run_supervisor, run_expert, cost tracking."""

import json
import yaml
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path

import subprocess

import grokswarm.shared as shared
from grokswarm.models import AgentState
from grokswarm.tools_registry import TOOL_DISPATCH, get_agent_tool_schemas
from grokswarm.registry_helpers import list_experts, save_memory
from grokswarm.engine import _execute_tool, _repair_json, _trim_conversation, MAX_TOOL_RESULT_SIZE


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
            "  total_tokens INTEGER NOT NULL"
            ")"
        )
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

    def log_usage(self, model: str, prompt_tokens: int, completion_tokens: int):
        self.conn.execute(
            "INSERT INTO metrics (model, prompt_tokens, completion_tokens, total_tokens) VALUES (?, ?, ?, ?)",
            (model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
        )
        self.conn.commit()

    def get_metrics(self) -> dict:
        cur = self.conn.execute(
            "SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens) FROM metrics"
        )
        row = cur.fetchone()
        return {
            "prompt_tokens": row[0] or 0,
            "completion_tokens": row[1] or 0,
            "total_tokens": row[2] or 0,
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
            shared.state.project_cost_usd = data.get("cost_usd", 0.0)
        except (json.JSONDecodeError, KeyError, OSError):
            pass


def _save_project_costs():
    f = _costs_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "prompt_tokens": shared.state.project_prompt_tokens,
        "completion_tokens": shared.state.project_completion_tokens,
        "cost_usd": round(shared.state.project_cost_usd, 6),
        "last_updated": datetime.now().isoformat(),
    }
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(f)


def _record_usage(model: str, prompt_tokens: int, completion_tokens: int):
    get_bus().log_usage(model, prompt_tokens, completion_tokens)
    inp_rate, out_rate = shared._get_pricing(model)
    shared.state.project_prompt_tokens += prompt_tokens
    shared.state.project_completion_tokens += completion_tokens
    shared.state.project_cost_usd += (prompt_tokens / 1_000_000.0) * inp_rate + (completion_tokens / 1_000_000.0) * out_rate
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
            _record_usage(shared.MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)
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
    if shared.PROJECT_CONTEXT:
        from grokswarm.context import format_context_for_prompt
        project_context = "\n" + format_context_for_prompt(shared.PROJECT_CONTEXT)

    system_prompt = f"""You are {data['name']}, an expert with permanent mindset:
{data['mindset']}

Core objectives: {data.get('objectives', ['Execute efficiently'])}

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
- Work fast: create files, write code, test, done. Minimize rounds.

PLANNING (MANDATORY):
- Your VERY FIRST action must be calling update_plan to outline your work steps.
- Each step should be a short, concrete action (e.g. "Read app.py to understand layout", "Fix Container layout param", "Run app to verify fix").
- As you complete each step, call update_plan again with the updated statuses. Mark the current step "in-progress" and completed steps "done".
- The user monitors your plan in real-time. Keep it accurate.{prior_context}{project_context}"""

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_desc},
    ]
    full_output = ""
    tool_actions: list[str] = []

    # Build tool schemas dynamically so MCP tools are included
    agent_tools = get_agent_tool_schemas()

    shared.state.agent_mode += 1
    try:
        for _round in range(max_rounds):
            # Compact conversation if it's getting large (same as main chat loop)
            conversation = await _trim_conversation(conversation)

            agent.transition(AgentState.THINKING)

            # Warn agent on final round so it can wrap up
            if _round == max_rounds - 1:
                conversation.append({
                    "role": "user",
                    "content": "[SYSTEM] This is your FINAL round. Wrap up now: summarize what you accomplished and what remains unfinished.",
                })

            _api_kwargs = dict(
                model=expert_model,
                messages=conversation,
                tools=agent_tools,
                max_tokens=shared.MAX_TOKENS,
            )
            if expert_temperature is not None:
                _api_kwargs["temperature"] = float(expert_temperature)
            response = await shared._api_call_with_retry(
                lambda: shared.client.chat.completions.create(**_api_kwargs),
                label=f"Expert:{data['name']}"
            )
            if hasattr(response, 'usage') and response.usage:
                pt = response.usage.prompt_tokens or 0
                ct = response.usage.completion_tokens or 0
                _record_usage(expert_model, pt, ct)
                agent.add_usage(pt, ct)
                shared.state.global_tokens_used += pt + ct
                _inp_r, _out_r = shared._get_pricing(expert_model)
                shared.state.global_cost_usd += (pt / 1_000_000.0) * _inp_r + (ct / 1_000_000.0) * _out_r

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
                            "content": f"Error: invalid JSON arguments",
                        })
                        continue

                agent.current_tool = tool_name
                detail = ""
                if tool_name in ("write_file", "read_file", "edit_file"):
                    detail = f" \u2192 {args.get('path', '?')}"
                elif tool_name == "run_shell":
                    detail = f" \u2192 {args.get('command', '?')[:60]}"
                shared._log(f"agent {display_name}: {tool_name}{detail}")
                tool_actions.append(f"{tool_name}{detail}")

                if tool_name == "update_plan":
                    args["_agent_name"] = display_name
                elif tool_name == "spawn_agent":
                    args["_parent"] = display_name

                result_str = await _execute_tool(tool_name, args)
                if len(result_str) > MAX_TOOL_RESULT_SIZE:
                    result_str = result_str[:MAX_TOOL_RESULT_SIZE] + "\n... (truncated)"
                conversation.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": result_str,
                })

            agent.current_tool = None

        save_memory(f"expert_{display_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", full_output)
        if bus:
            bus.post(display_name, full_output[:2000] if full_output else "(no text output)")
            if tool_actions:
                bus.post(display_name, f"Tools executed: {', '.join(tool_actions[:20])}", kind="status")
        agent.transition(AgentState.DONE)
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
