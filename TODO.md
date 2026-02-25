# GrokSwarm — Task Tracker

**Last updated:** February 21, 2026
**Current version:** v0.30.0

---

## Next Generation: Hierarchical, Asynchronous Swarm (v0.27.0+)

This is the roadmap to transform GrokSwarm from a flat, sequential tool into a dynamic, parallel, hierarchical agent swarm with deep observability and real-time human interaction.

### Phase 1: Async Core & State Machine (v0.27.0) ✅
- [x] **Refactor Core Loop:** Rewrite `chat()` and `_execute_tool()` to use `asyncio` so multiple agents can run concurrently.
- [x] **Agent State Machine:** Implement states (`IDLE`, `THINKING`, `WORKING`, `PAUSED`, `WAITING_ON_DEPENDENCY`) for agents.
- [x] **CLI Multiplexing:** Update the `prompt_toolkit` CLI to remain responsive while agents work in the background.

### Phase 2: Spawning & Messaging Tools (v0.28.0) ✅
- [x] **`spawn_agent` Tool:** Give the LLM the ability to spawn sub-agents with specific roles, instructions, and budgets.
- [x] **`send_message` Tool:** Allow agents to send direct messages to other agents for negotiation and coordination.
- [x] **Async SwarmBus:** Upgrade the SQLite `SwarmBus` to support async pub/sub so agents can listen for messages addressed to them.

### Phase 3: Resource Management & Master Control (v0.29.0) ✅
- [x] **Budget Tracking:** Implement a `ResourceManager` to track token/cost budgets globally and per-agent.
- [x] **Out-of-Funds State:** Agents hit a `PAUSED_OUT_OF_FUNDS` state when they exceed their budget and must negotiate for more.
- [x] **Real-Time Interruption:** Implement `/pause <agent>` and `/resume <agent>` commands to control agents mid-task.

### Phase 4: The Panopticon TUI Dashboard (v0.30.0) ✅
- [x] **Rich TUI Framework:** Upgrade `/dashboard` using advanced `Rich` live display with Tree view.
- [x] **Live Tree View:** Visualize the agent hierarchy (e.g., `CEO -> [BackendLead, FrontendLead]`).
- [x] **Live State & Feed:** Show real-time agent states, budget usage, and a live feed of inter-agent `SwarmBus` messages.

---

## Rejected

| Item | Why |
|---|---|
| S7 — Safety Monitor Agent | Over-engineered; denylist covers it |
| S8 — edit_file default=False | Would cripple normal workflow |
| U6 — Fuzzy-match edit_file | Silent corruption risk |
| A1 — Refactor into package | Single-file is a feature |
| P5 — Selective tool schemas | Not worth the complexity |
| R1 — invoke_cli meta-tool | Redundant — tools already accessible |
| R5 — Persistent Coordinator | Background daemon, no current need |
| R6 — Voice mode | Out of scope |
| R7 — Plugin marketplace | Out of scope |

---

## Completed Releases

- [x] v0.30.0 — Phase 4 Panopticon Dashboard: Rich Tree agent hierarchy view, live state indicators, budget tracking display, status/error message feed. 176 tests pass.
- [x] v0.29.0 — Phase 3 Resource Management: per-agent token/cost budgets (AgentInfo.check_budget/add_usage), global budget tracking, /pause and /resume slash commands, PAUSED state for over-budget agents. 176 tests pass.
- [x] v0.28.0 — Phase 2 Spawning & Messaging: spawn_agent tool (background asyncio.Task), send_message tool (SwarmBus direct messaging), check_messages and list_agents tools, agent_name support in run_expert. 176 tests pass.
- [x] v0.27.0 — Phase 1 Async Core: full asyncio refactor (AsyncOpenAI, async _execute_tool/_stream_with_tools/_compact_conversation/analyze_image, asyncio.gather parallel tools, prompt_async CLI), AgentState enum + AgentInfo dataclass state machine. 149 tests pass.
- [x] v0.26.0 — A8 Token Usage & Cost Metrics panel in dashboard + `/metrics` command. 149 tests pass.
- [x] v0.25.0 — L3 SwarmState refactor, L4 docs/naming harmonization, A4 SQLite SwarmBus coordination, A5 live `dashboard` command. 149 tests pass.
- [x] v0.24.0 — A3 dynamic tool registration (skills auto-register as callable LLM tools on load + create), L2 /doctor chromium binary check. 142 tests pass.
- [x] v0.23.0 — B12 undo-delete for new files, L1 /project tab-completion (dir completer + recent list), A6 isolated temp-dir test validation for /self-improve promotion. 138 tests pass.
- [x] v0.22.1 — B1–B11 bug fixes from FEEDBACK_EVAL Tier 1 + Tier 2 (global declarations, undo pre-state, cross-project state leaks, SSRF cloud metadata, cache consistency, symbol cap alignment, mtime sampling, readonly run_tests, self-improve test gate, hoisted constants). 133 tests pass.
- [x] v0.22.0 — U1/G1 project switcher, Perf-P1 context cache, U2 /doctor, A2 playwright atexit fix, S5 read-only sessions, Perf-P4 symbol trim. All P2 items complete. 135 tests.
- [x] v0.21.0 — C9 context auto-refresh, S4 secret redaction, G4 trust mode, C6 file size guardrail, U4 .grokswarm.yml config, Perf-P2 ripgrep fallback, G3 multi-level undo. 7 P2 items. 125 tests.
- [x] v0.20.1 — Fixed S2 symlink escape (walk component chain), self-improve shell guard. 111 tests.
- [x] v0.20.0 — A7 JSON repair, C4 compaction boundary, P3 token-aware compaction, U3 analyze_image, G2 Playwright install prompt. All P1 items complete. 107 tests.
- [x] v0.19.0 — S2 symlink guard, S3 SSRF block, S6 edit preview, C2 AnnAssign, C5 /edit→/write, C7 AST warnings, C8 expert prompt
- [x] v0.18.0 — S1 dangerous-command denylist, C1 IGNORE_DIRS fnmatch, `/self-improve` shadow copy
- [x] v0.17.0 — xAI server-side `web_search` + `x_search` tools
- [x] v0.16.0 — Dead code cleanup, `--max-tokens`, `/undo`
- [x] v0.15.0 — AST code intelligence, `find_symbol`, `find_references`
- [x] v0.14.0 — Regex grep, `--model` flag, token usage display
- [x] v0.13.0 — Multi-edit arrays, git stash/init, checkpoint reminders
- [x] v0.12.0 — Test-fix cycle
- [x] v0.11.0 — Parallel tool execution, round counter
- [x] v0.10.0 — Error recovery, auto-lint, API retry
- [x] v0.9.0 — Surgical editing, compaction, testing, deep context
- [x] v0.8.0 — Playwright browser tools
- [x] v0.7.0 — Git integration
