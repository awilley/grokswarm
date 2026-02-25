# Swarm Architecture: Learnings from NanoClaw & Overstory

**Date:** February 21, 2026

This document captures architectural insights, patterns, and features from two other AI agent swarm projects—**NanoClaw** and **Overstory**—and compares them to **GrokSwarm**. The goal is to identify practices we can learn, adopt, and integrate into GrokSwarm to make it a more robust, scalable, and capable local-first AI team.

---

## 1. NanoClaw Insights

NanoClaw is a Node.js-based orchestrator that connects WhatsApp to Claude Agent SDKs running in isolated containers.

### What they do well (Architecture & UX):
*   **UX: "Skills over Features" (Zero Config):** NanoClaw avoids configuration files entirely. Instead of building a bloated monolith with toggles for every use case, users run Claude Code to apply a skill (e.g., `/add-telegram`) which modifies their fork's code directly.
    *   *GrokSwarm Comparison:* GrokSwarm has `.grokswarm.yml` and various CLI flags.
    *   *Opportunity:* We could adopt a model where users "install" new tools or experts by having the LLM write the code directly into their GrokSwarm installation, keeping the core minimal.
*   **Architecture: 3-Level Conflict Resolution:** When applying skills, NanoClaw uses a strict escalation path: Git (programmatic merge) -> Claude (AI resolution) -> User (human decision). It uses `git rerere` to cache resolutions so users don't have to resolve the same conflicts twice.
    *   *Opportunity:* If GrokSwarm adopts parallel worktrees, we will need a robust merge strategy. Using `git rerere` combined with LLM conflict resolution is a brilliant pattern.
*   **Architecture: Safe Operations (Backup/Restore):** Before any skill application, NanoClaw copies all affected files to a backup directory. If tests fail after the merge, it restores the backup.
    *   *Opportunity:* GrokSwarm's `/undo` relies on git checkouts. A pre-flight backup directory for complex multi-file edits would provide an extra layer of safety, especially for users who forget to commit.
*   **Containerized Agent Isolation:** NanoClaw routes messages to agents running inside isolated Linux VMs/containers. Each group has an isolated filesystem. 
    *   *GrokSwarm Comparison:* GrokSwarm currently runs either entirely locally or inside a single shared Docker container. 
    *   *Opportunity:* For true safety when agents execute arbitrary shell commands or write code, GrokSwarm should consider spawning ephemeral, isolated containers per task or per agent, rather than relying solely on path validation and denylists.
*   **Markdown-Driven Skills:** NanoClaw defines its skills (e.g., `add-gmail`, `debug`) using Markdown files (`SKILL.md`). 
    *   *GrokSwarm Comparison:* GrokSwarm currently uses YAML for Expert definitions and Python functions for tools.
    *   *Opportunity:* LLMs natively understand Markdown much better than YAML. Transitioning our Expert and Skill registries to Markdown documents would likely improve the LLM's comprehension of its own capabilities and instructions.

---

## 2. Overstory Insights

Overstory is a Bun/TypeScript project-agnostic swarm system that turns a single session into a multi-agent team using tmux, SQLite, and git worktrees.

### What they do well (Architecture & UX):
*   **UX: Rich Observability CLI:** Overstory has a massive suite of observability commands (`dashboard`, `trace`, `feed`, `replay`, `costs`, `inspect`). It treats the swarm like a fleet of servers.
    *   *GrokSwarm Comparison:* GrokSwarm has a basic `/dashboard` and `/metrics`.
    *   *Opportunity:* We should expand our CLI to include a real-time event stream (`/feed`), chronological replay (`/replay`), and deep per-agent inspection (`/inspect <agent>`).
*   **UX: Inter-Agent Mail System:** Agents communicate via a typed SQLite mail protocol (`worker_done`, `escalation`, `question`) with broadcast addresses (`@builders`, `@all`).
    *   *GrokSwarm Comparison:* GrokSwarm's `SwarmBus` is currently a simple message queue.
    *   *Opportunity:* We should adopt a typed protocol for `SwarmBus` messages, allowing agents to send structured status updates, questions, and results to specific roles or groups.
*   **Architecture: Watchdog & Fleet Health:** Overstory uses a tiered health monitoring system (mechanical daemon -> AI triage -> monitor agent) to ensure agents don't get stuck in infinite loops or crash silently.
    *   *Opportunity:* GrokSwarm's test-fix loop can sometimes spin out of control. A background watchdog that monitors agent progress and interrupts stalled or looping agents would be highly valuable.
*   **Git Worktrees for Parallel Isolation:** Overstory uses `git worktree` to give each worker agent its own isolated directory and branch. Agents commit to their worktree, and the orchestrator merges them back.
    *   *GrokSwarm Comparison:* GrokSwarm operates directly on the main working directory, relying on a custom undo stack (`/undo`) and sequential execution.
    *   *Opportunity:* Adopting `git worktree` would allow GrokSwarm to run multiple agents in parallel on the same codebase without them overwriting each other's files. It provides a native, robust undo/isolation mechanism.
*   **Hierarchical Delegation:** Overstory enforces a strict hierarchy: Coordinator -> Team Lead -> Specialist Workers, with configurable depth limits to prevent runaway spawning.
    *   *GrokSwarm Comparison:* GrokSwarm has a "Supervisor" that routes tasks, but it's relatively flat.
    *   *Opportunity:* Formalizing a hierarchy where a Lead agent breaks down a spec into sub-tasks and spawns Worker agents in isolated worktrees would massively scale GrokSwarm's ability to handle complex features.
*   **Base + Overlay Prompting:** Overstory separates the "HOW" (reusable base agent definitions like `builder.md`) from the "WHAT" (task-specific overlays generated per task, containing file scope, branch name, and task ID).
    *   *GrokSwarm Comparison:* GrokSwarm injects everything into a single massive system prompt.
    *   *Opportunity:* We should separate our Expert personas (the "HOW") from the specific user request/context (the "WHAT"). This makes experts highly reusable across different projects.
*   **Structured Expertise (Mulch):** Overstory uses a tool called `mulch` to record specific project conventions, patterns, and failures. Agents are required to query this before starting and record new learnings before finishing.
    *   *GrokSwarm Comparison:* GrokSwarm dumps unstructured JSON files into a `memory/` folder.
    *   *Opportunity:* Implement a structured, queryable expertise database (perhaps using our new SQLite `SwarmBus`) to store specific conventions, decisions, and failure patterns.
*   **Strict Quality Gates:** Overstory agents cannot report a task as complete until they pass strict gates: `bun test`, `biome check`, and `tsc`.
    *   *GrokSwarm Comparison:* GrokSwarm has a great test-fix loop, but it's somewhat optional depending on the prompt.
    *   *Opportunity:* Enforce hard quality gates before an agent's worktree branch can be merged back into the main branch.

---

## 3. Validation of Current GrokSwarm Features

Reviewing these projects also validates several architectural decisions we've recently made in GrokSwarm:
*   **SQLite for IPC:** Overstory uses `bun:sqlite` in WAL mode for its mail, events, and metrics. GrokSwarm just implemented `SwarmBus` using SQLite for multi-agent coordination (v0.25.0). This confirms SQLite is the right choice for lightweight, concurrent agent messaging.
*   **CLI-First UX:** Both Overstory and GrokSwarm heavily index on a rich CLI experience (dashboards, trace commands, metrics) rather than building complex web UIs.

---

## 4. Actionable Recommendations for GrokSwarm

### Short-Term (v0.27.x)
1.  **Markdown Registries:** Migrate `experts/*.yaml` to `experts/*.md`. Use Markdown headers for Role, Capabilities, and Constraints.
2.  **Base + Overlay Prompts:** Refactor the system prompt generation to clearly separate the Expert Base Definition from the Task Overlay (current project context, specific request).
3.  **Typed SwarmBus Protocol:** Upgrade `SwarmBus` to support typed messages (`worker_done`, `escalation`, `question`) and broadcast addresses (`@all`, `@builders`).

### Medium-Term (v0.28.x - v0.29.x)
4.  **Git Worktree Isolation:** Implement a new tool/workflow where the Supervisor can spawn a sub-agent in a `git worktree`. This will require a new merge/review step but will unlock safe parallel coding.
5.  **Structured Expertise:** Replace the `memory/*.json` dumps with a structured SQLite table (e.g., `expertise.db`) categorized by `convention`, `pattern`, `failure`, and `decision`.
6.  **Rich Observability CLI:** Add `/feed`, `/replay`, and `/inspect <agent>` commands to give users deep visibility into what the swarm is doing.

### Long-Term (v0.30+)
7.  **Ephemeral Containers:** Explore using Docker SDK for Python to spin up lightweight, ephemeral containers for worker agents to execute shell commands and tests safely, rather than running them on the host OS.
8.  **Watchdog Daemon:** Implement a background monitor that tracks agent health, interrupts infinite loops, and escalates failures to the user.
9.  **Zero-Config Skills:** Transition from a monolithic codebase to a "skills" model where users can ask the LLM to inject new capabilities directly into their fork, rather than relying on config files.