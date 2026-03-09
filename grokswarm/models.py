"""Pure data models — no grokswarm imports."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AgentState(Enum):
    """Lifecycle states for each agent in the swarm."""
    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    PAUSED = "paused"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentInfo:
    """Runtime metadata for a single agent instance."""
    name: str
    expert: str
    state: AgentState = AgentState.IDLE
    task: str = ""
    current_tool: str | None = None
    tokens_used: int = 0
    token_budget: int = 0
    cost_usd: float = 0.0
    cost_budget_usd: float = 0.0
    parent: str | None = None
    pause_requested: bool = False
    plan: list[dict] = field(default_factory=list)
    # Guardrails: plan-then-execute
    phase: str = "planning"  # "planning" | "executing" | "verifying"
    approved_plan: list[dict] | None = None  # frozen copy of approved plan
    plan_files_allowed: set[str] = field(default_factory=set)  # files agent declared it would touch
    # Live visibility: rolling log of recent tool calls
    tool_call_log: list[dict] = field(default_factory=list)  # [{tool, args_summary, result_preview, round}]
    current_model: str = ""  # model being used this round
    cached_tokens_total: int = 0  # total cached tokens for this agent
    workspace: Path | None = None  # worktree path for branch-isolated agents
    branch: str = ""  # git branch name for this agent's worktree

    def transition(self, new_state: AgentState):
        self.state = new_state

    def check_budget(self) -> bool:
        if self.token_budget > 0 and self.tokens_used >= self.token_budget:
            return False
        if self.cost_budget_usd > 0 and self.cost_usd >= self.cost_budget_usd:
            return False
        return True

    def add_usage(self, prompt_tokens: int, completion_tokens: int, model: str | None = None, cached_tokens: int = 0):
        from grokswarm.shared import _get_pricing, MODEL
        inp_rate, cached_rate, out_rate = _get_pricing(model or MODEL)
        self.tokens_used += prompt_tokens + completion_tokens
        non_cached = max(0, prompt_tokens - cached_tokens)
        self.cost_usd += (
            (non_cached / 1_000_000.0) * inp_rate
            + (cached_tokens / 1_000_000.0) * cached_rate
            + (completion_tokens / 1_000_000.0) * out_rate
        )


@dataclass
class SwarmState:
    """All mutable session state in one place."""
    trust_mode: bool = False
    request_auto_approve: bool = False
    read_only: bool = False
    self_improve_active: bool = False
    verbose_mode: bool = False
    vi_mode: bool = False
    agent_mode: int = 0
    edit_history: list = field(default_factory=list)
    pending_write_count: int = 0
    last_edited_file: str | None = None
    test_fix_state: dict = field(default_factory=lambda: {"cmd": None, "attempts": 0})
    agents: dict[str, AgentInfo] = field(default_factory=dict)
    global_token_budget: int = 0
    global_tokens_used: int = 0
    global_cost_budget_usd: float = 0.0
    global_cost_usd: float = 0.0
    session_cost_budget_usd: float = 0.0  # 0 = no limit; set via /budget
    project_prompt_tokens: int = 0
    project_completion_tokens: int = 0
    project_cached_tokens: int = 0
    project_cost_usd: float = 0.0

    def reset_project_state(self):
        self.edit_history.clear()
        self.test_fix_state["cmd"] = None
        self.test_fix_state["attempts"] = 0
        self.pending_write_count = 0
        self.agents.clear()
        self.global_tokens_used = 0
        self.global_cost_usd = 0.0
        self.request_auto_approve = False
        self.vi_mode = False

    def register_agent(self, name: str, expert: str, task: str = "",
                       token_budget: int = 0, cost_budget_usd: float = 0.0,
                       parent: str | None = None) -> AgentInfo:
        agent = AgentInfo(name=name, expert=expert, task=task,
                          token_budget=token_budget, cost_budget_usd=cost_budget_usd,
                          parent=parent)
        self.agents[name] = agent
        return agent

    def get_agent(self, name: str) -> AgentInfo | None:
        return self.agents.get(name)

    def clear_swarm(self):
        from grokswarm.shared import _background_tasks
        self.agents.clear()
        self.global_tokens_used = 0
        self.global_cost_usd = 0.0
        self.vi_mode = False
        for task_name, task in _background_tasks.items():
            if not task.done():
                task.cancel()
        _background_tasks.clear()


@dataclass
class SubTask:
    """A single sub-task in an orchestrated task DAG."""
    id: str
    description: str
    expert: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | done | failed
    deliverables: list[str] = field(default_factory=list)
    result_summary: str = ""
    agent_name: str = ""


@dataclass
class TaskDAG:
    """Directed acyclic graph of sub-tasks for orchestrated execution."""
    goal: str
    subtasks: list[SubTask] = field(default_factory=list)
    current_phase: int = 0

    def ready_tasks(self) -> list[SubTask]:
        """Tasks whose dependencies are all 'done'."""
        done_ids = {t.id for t in self.subtasks if t.status == "done"}
        return [t for t in self.subtasks
                if t.status == "pending" and all(d in done_ids for d in t.depends_on)]

    def is_complete(self) -> bool:
        return all(t.status in ("done", "skipped") for t in self.subtasks)

    def failed_tasks(self) -> list[SubTask]:
        return [t for t in self.subtasks if t.status == "failed"]
