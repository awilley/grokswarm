# GrokSwarm: Next-Gen Hierarchical Swarm Architecture

**Date:** February 21, 2026

## 1. Current State of GrokSwarm

Right now, GrokSwarm's architecture is **flat and sequential**:
*   **The Flow:** You run a task, the `Supervisor` creates a plan (a list of experts), and then the system runs each expert **one by one** in a blocking loop.
*   **Communication:** Experts leave messages on the SQLite `SwarmBus` for the next expert to read.
*   **Hierarchy:** None. The Supervisor just picks a list of experts. Experts cannot spawn other experts.
*   **Concurrency:** None. Only one agent is "thinking" or acting at a time.
*   **Interaction:** You chat with the active agent, but you cannot interrupt an agent mid-task or switch contexts to a background agent (because there are no background agents).

## 2. Analysis of Your Proposed Architecture

Your vision describes a **Dynamic, Asynchronous, Hierarchical Swarm**. This is the holy grail of agentic systems and aligns perfectly with the learnings we just documented from Overstory and NanoClaw.

Here is a breakdown of your requested features and what it takes to build them:

### A. The "CEO" Agent (Project Orchestrator)
*   **Concept:** Every project has a single head agent (CEO) that you chat with. It acts as the router, planner, and resource allocator.
*   **Implementation:** We designate a specific expert profile (e.g., `ceo.md`) that is always the entry point for a project. The CEO doesn't write code; it writes specs, spawns leads, and reviews outcomes.

### B. Dynamic Hierarchical Structure & Negotiation
*   **Concept:** The CEO can spin up resources (sub-agents) with any structure, and negotiate with direct reports.
*   **Implementation:** 
    *   We need to give agents a new tool: `spawn_agent(role, instructions, budget)`.
    *   We need a `send_message(recipient_id, message)` tool for inter-agent negotiation.
    *   **Crucial Shift:** Agents must run asynchronously. We need to move GrokSwarm's core loop from synchronous Python to `asyncio` (or background threads) so multiple agents can run in parallel.

### C. Resource & Cost Limits (Master Control)
*   **Concept:** Set reasonable limits and track costs globally and per-agent.
*   **Implementation:** 
    *   The `SwarmBus` (or a new `ResourceManager`) acts as the bank.
    *   When the CEO spawns an agent, it allocates a token/cost budget.
    *   Before every LLM API call, the agent checks its budget. If it hits the limit, it enters a `PAUSED_OUT_OF_FUNDS` state and sends a message to its manager (or the user) requesting more budget.

### D. Deep Observability (The Panopticon)
*   **Concept:** See what every agent is doing, the hierarchy, and their comms.
*   **Implementation:** 
    *   We upgrade the `/dashboard` command to use a rich TUI framework (like `Textual` or an advanced `Rich` live display).
    *   It will show a live tree view: `CEO -> [BackendLead -> [DBWorker, APIWorker], FrontendLead]`.
    *   It will show live states: `THINKING`, `EXECUTING_TOOL`, `PAUSED`, `WAITING_ON_DEPENDENCY`.
    *   A live feed panel will show inter-agent `SwarmBus` messages.

### E. Real-Time Interaction & Interruption
*   **Concept:** Chat with any agent at any moment, wait for their thought to complete, or force a pause.
*   **Implementation:**
    *   The main CLI thread remains responsive to user input while agents run in the background.
    *   We introduce a `/chat <agent_id>` command to switch your terminal focus to a specific agent.
    *   Agents will have a `pause_event` flag. If you type `/pause <agent_id>`, the system sets the flag. After the agent finishes its current API call or tool execution, it yields control to you, allowing you to inject new instructions before it continues.

---

## 3. How to Build It (The Roadmap)

To get there, we need to execute a major architectural refactor. 

**Phase 1: Async Core & State Machine**
*   Rewrite the `chat()` and `_execute_tool()` loops to use `asyncio`.
*   Implement the Agent State Machine (`IDLE`, `THINKING`, `WORKING`, `PAUSED`).

**Phase 2: The Spawning & Messaging Tools**
*   Add `spawn_agent` and `send_message` tools.
*   Upgrade `SwarmBus` to support async pub/sub so agents can "listen" for messages addressed to them.

**Phase 3: Resource Management & Dashboard**
*   Implement the budget tracking system.
*   Build the live TUI dashboard to visualize the agent tree and states.

**Phase 4: CLI Multiplexing**
*   Update the `prompt_toolkit` CLI to allow switching contexts (`/chat <agent_id>`) and interrupting background tasks.

---

## 4. A Good First Project to Test This

To test a hierarchical, parallel swarm, we need a project that naturally requires decomposition, parallel work, and coordination between distinct domains.

### The Test Project: "Build a Full-Stack Markdown Blog Engine"

**Why it's perfect:** It has distinct components (Frontend UI, Backend API, Database/Storage) that can be worked on in parallel, but require strict contract negotiation (API schemas).

**The Test Scenario:**
1.  **Initialization:** You start GrokSwarm and tell the CEO: *"Build a markdown blog engine with a FastAPI backend and a React frontend. Budget: $2.00."*
2.  **Decomposition:** The CEO analyzes the request and uses `spawn_agent` to create two Leads: `BackendLead` and `FrontendLead`, giving them each a $0.50 budget.
3.  **Parallel Spawning:** 
    *   `BackendLead` spawns a `DatabaseWorker` and an `ApiWorker`.
    *   `FrontendLead` spawns a `ComponentWorker` and a `StylingWorker`.
4.  **Observability Check:** You open the `/dashboard` and see the tree of 7 agents. You see `DatabaseWorker` is `EXECUTING_TOOL` (writing SQL models) while `FrontendLead` is `WAITING_ON_DEPENDENCY` (waiting for the API schema).
5.  **Negotiation:** You watch the live feed as `FrontendLead` sends a message to `BackendLead`: *"I need the JSON schema for the blog post object."* `BackendLead` replies with the schema.
6.  **Interruption Test:** You notice `StylingWorker` is about to write plain CSS. You type `/pause StylingWorker`, wait for it to yield, and type: *"Make sure to use Tailwind CSS instead."* It acknowledges and resumes.
7.  **Resource Limit Test:** `ApiWorker` gets stuck in a test-fix loop and hits its $0.20 budget. It pauses and alerts `BackendLead`. `BackendLead` analyzes the failure, fixes the architectural flaw, and allocates $0.10 more to `ApiWorker` to finish.
8.  **Completion:** The CEO verifies the final integration tests pass and reports back to you.