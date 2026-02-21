# Grok Swarm — Project Goals & Roadmap

**Last updated:** February 20, 2026

## Vision
Build the **best CLI in the world** for Grok — a persistent, local-first, multi-agent AI workhorse that runs on my Windows 11 machine all day and night.

It should feel like a real digital team that:
- Works silently in the background
- Remembers everything (skills, mindsets, context, history)
- Can actually **do real work** on my files and system
- Self-improves over time
- Is delightful and fast to use (better than Claude CLI)

## Core Requirements (Must-Haves)
- Local-first (everything runs on my PC, no cloud dependency except the Grok API)
- Multiple specialized Grok agents (experts) that can work independently or collaborate
- Persistent Skill Registry (human-editable, self-extending)
- Persistent Expert Registry (mindsets, objectives, KPIs)
- Multi-agent Supervisor with dynamic self-organization and saved teams
- Persistent memory & command history (survives restarts)
- Native filesystem & safe shell tools with approval gates
- Beautiful, fast interactive mode with streaming responses and tab completion
- Short, convenient CLI (grokswarm)
- Keep a terminal open when active (no forced daemon for now)

## Current Status (as of Feb 20, 2026)
- ✅ Short grokswarm CLI with clean interactive mode
- ✅ Live streaming responses with markdown rendering
- ✅ Persistent command history (↑/↓ arrows via prompt_toolkit)
- ✅ Tab completion for slash commands, experts, and file paths
- ✅ Rich slash commands (/list, /read, /edit, /run, /search, /swarm, /experts, /context, /clear, /test)
- ✅ Safe filesystem + shell tools with sandbox (path validation via `is_relative_to`)
- ✅ Skill & Expert Registries + multi-agent supervisor
- ✅ Saved teams, rich tables, memory
- ✅ Docker-based, easy restart
- ✅ Claude CLI-inspired UI (minimal ❯ prompt, themed output, spinner, Panel welcome)
- ✅ Conversation context (multi-turn within session, auto-trimmed to stay within limits)
- ✅ Proper error handling (specific exceptions, no bare except)
- ✅ System prompt for personality/consistency
- ✅ **Automatic project/folder context awareness** — scans cwd for file tree + key files, injects into system prompt, /context command to view/refresh
- ✅ **Named persistent sessions** — `grokswarm --session research` saves/resumes conversations, auto-saves on each response, /session list/save/load/delete commands
- ✅ **Tool-calling agent** — LLM autonomously invokes filesystem/shell tools via OpenAI function calling (list_directory, read_file, write_file, search_files, run_shell), multi-round tool loop with streaming, approval gates on writes/shell
- ✅ **Self-extension** — LLM can propose new experts and skills via `create_expert` / `create_skill` tool calls with rich approval panels, `list_registry` tool to inspect available agents, proactive proposal when specialists are missing
- ✅ **Git integration (v0.7.0)** — LLM-callable git tools (status, diff, log, commit, checkout, branch) with approval gates on destructive ops, `/git` slash command, tab completion for git subcommands
- ✅ **Playwright browser tools (v0.8.0)** — LLM-callable browser tools (fetch_page, screenshot_page, extract_links) with lazy-init headless Chromium, `/browse` slash command, approval gate on screenshots
- ✅ **Surgical file editing (v0.9.0)** — `edit_file` tool for precise search/replace edits with diff preview and approval gate, LLM instructed to always prefer edit_file over write_file for modifications
- ✅ **Enhanced code editing precision (v0.9.0)** — `read_file` supports line ranges (start_line/end_line) for efficient partial reads with line numbers
- ✅ **Deeper git awareness (v0.9.0)** — `git_show_file` to view files at any commit/branch, `git_blame` for line-by-line history, code structure map (classes/functions) injected into context
- ✅ **Smart context awareness (v0.9.0)** — language detection & stats, code structure extraction (Python/JS/TS/Go/Rust/Java/Ruby), increased context limits (8KB/file, depth 4, 150 files), shown in /context view
- ✅ **Conversation compaction (v0.9.0)** — LLM-powered summarization of old messages instead of hard truncation, preserves context from earlier conversations while saving tokens
- ✅ **Testing integration (v0.9.0)** — `run_tests` tool with auto-detection of test frameworks (pytest, jest, go test, cargo test, mocha, unittest), `/test` slash command, pass/fail indicators, 120s timeout
- ✅ **Error recovery loop (v0.10.0)** — auto-lint after every `edit_file`/`write_file` (Python `py_compile`, Node `--check`, TS `tsc --noEmit`), lint errors injected into tool response so LLM self-corrects immediately, system prompt instructs LLM to never leave files in broken state, **auto-retry with backoff on all API calls** (3 attempts, 2/5/10s backoff, skips auth errors), resilient multi-agent swarms
- ✅ **Advanced agentic loop (v0.11.0)** — parallel execution of read-only tools via ThreadPoolExecutor (up to 4 workers), round counter showing progress through 10-round tool loop, per-tool wall-clock timing, graceful Ctrl+C interruption that returns partial results, centralized `_execute_tool` helper
- ✅ **Structural test-fix iteration (v0.12.0)** — when `run_tests` fails, the system enters a test-fix cycle: auto-reruns the same test command after every successful `edit_file` (up to 3 attempts), tagged results (`[AUTO-TEST FAILURE]`, `[AUTO-RETEST PASSED]`, `[AUTO-RETEST FAILED]`) force the LLM to analyze and fix until tests pass, system prompt documents the cycle so LLM cooperates
- ✅ **Multi-edit + Git completion (v0.13.0)** — `edit_file` now supports an `edits` array for multiple search/replace ops in one tool call (atomic, all-or-nothing validation), `git_stash` tool (list/push/pop/drop with approval gates), `git_init` tool, auto-checkpoint reminders after 5+ consecutive file mutations without a commit, 10 git tools total
- ✅ **Regex grep + model flexibility + UX (v0.14.0)** — `grep_files` now supports regex patterns (`is_regex`) and context lines (`context_lines`, like grep -C). CLI flags `--model`/`-m`, `--base-url`, `--api-key` for provider flexibility (works with any OpenAI-compatible API). Token usage display after each response. Tool-aware conversation compaction preserves file read/write context. Unicode tree connectors. Bug fixes: JSX/TSX structure patterns, git_stash/git_init display, `/grep` quoted patterns, run_shell 120s timeout.
- ✅ **AST code intelligence + repo-map (v0.15.0)** — `find_symbol` tool (go-to-definition via Python AST, regex for 7 other languages), `find_references` tool (import analysis + word-boundary search), import dependency graph injected into project context, deep symbol index (classes with methods, functions with args), zero new dependencies (uses built-in `ast` module), 29 tools total
- ✅ **Dead code cleanup + max_tokens + /undo (v0.16.0)** — removed obsolete `_scan_code_structure` + `STRUCTURE_PATTERNS` (~40 lines dead code). Added `MAX_TOKENS = 16384` default + `--max-tokens` CLI flag to prevent runaway responses. New `/undo` command for quick file revert via `git checkout`, tracks last edited file automatically. 19 slash commands total.
- ✅ **xAI server-side search (v0.17.0)** — new `web_search` and `x_search` tools powered by the xAI Responses API (`/v1/responses`). Real-time web and X/Twitter search with citations, no Playwright needed. New `/web` and `/x` slash commands. 31 tools, 21 slash commands. System prompt directs LLM to prefer search tools over Playwright for finding information.

- ✅ **Quick P1 fixes (v0.19.0)** — S2 symlink guard in `_safe_path`, S3 SSRF block for `fetch_page`/`screenshot_page`/`extract_links` (blocks localhost/RFC1918/file://), S6 edit preview expanded to 8 lines, C2 `AnnAssign` in AST symbol index, C5 renamed `/edit` to `/write` (was misnamed), C7 AST parse failure warnings, C8 expert prompt de-branded. 95 tests passing.

- ✅ **All P1 complete (v0.20.0)** — A7 `_repair_json()` strips markdown fences + trailing commas from LLM tool arguments, C4 compaction boundary fix (orphaned tool messages pulled into old), P3 token-aware compaction (`_estimate_tokens()` at ~4 chars/token, `COMPACTION_TOKEN_LIMIT = 100,000`), U3 `analyze_image` vision tool (base64 → Grok multimodal API, png/jpg/gif/webp, 20 MB limit), G2 Playwright first-run install prompt. 29 tools, 107 tests passing.

- ✅ **Bug fixes (v0.20.1)** — Fixed S2 symlink escape guard (now walks each path component instead of checking after resolve). Self-improve guard extended to block `run_shell` commands touching main.py (not just edit_file/write_file). 111 tests.

- ✅ **P2 high-leverage batch (v0.21.0)** — C9 incremental context auto-refresh after edits (re-parses AST, rebuilds system prompt), S4 secret redaction in session saves (regex for API keys/JWTs/PEM), G4 trust mode `/trust` toggle (auto-approves non-dangerous ops), C6 file size guardrail in `read_file` (warns >1 MB), U4 `.grokswarm.yml` project config (model/base_url/ignore_dirs), Perf-P2 ripgrep fallback in `grep_files`, G3 multi-level undo stack (20 entries). 22 slash commands, 125 tests.

- ✅ **P2 completion (v0.22.0)** — U1/G1 `/project` switcher with recent-5 cache and numeric shortcuts, Perf-P1 smart context cache (`~/.grokswarm/cache` with mtime invalidation), U2 `/doctor` environment health check, A2 Playwright atexit 3s timeout wrapper, S5 `/readonly` session toggle (blocks all mutating tools), Perf-P4 symbol index trimmed to 15 defs/file. All P2 items complete. 25 slash commands, 135 tests.

- ✅ **Tier A completion (v0.23.0)** — B12 undo for newly created files (delete-on-undo sentinel), L1 `/project` directory tab-completion + recent project suggestions, A6 isolated temp-dir test validation for `/self-improve` promotion. 138 tests.

- ✅ **Tier B completion (v0.24.0)** — A3 dynamic skill-tool registration (`skill_{name}` auto-loaded + hot-registered on create), L2 `/doctor` checks chromium browser binary readiness (not just Playwright import). 142 tests.

- ✅ **Tier C + P3 completion (v0.25.0)** — L3 `SwarmState` dataclass migration for mutable session globals, L4 command/help naming harmonization, A4 SQLite `SwarmBus` coordination for swarm/team runs, A5 live `dashboard` command. 149 tests.
- ✅ **Dashboard Metrics (v0.26.0)** — A8 Token Usage & Cost Metrics panel in dashboard + `/metrics` command. 149 tests.

- ✅ **Safety + self-improvement (v0.18.0)** — dangerous-command denylist for `run_shell` (15 patterns: `rm -rf`, `sudo`, `curl|bash`, `mkfs`, `git push --force`, etc). Fixed `IGNORE_DIRS` glob matching (fnmatch for `*.egg-info`). New `/self-improve` command: safe self-editing via shadow copy with mechanical guard blocking live `main.py` edits + py_compile verification before promotion. 21 slash commands.

## Known Issues / Tech Debt
_(cleaned up in v0.8.1)_
- ~~`requirements.txt` still lists langchain + chromadb~~ — removed, down to 7 deps
- ~~`/context refresh` resets conversation history~~ — now updates system prompt in-place
- ~~`search_files` only matches file names~~ — added `grep_files` content search tool + `/grep` command
- ~~Docker stale chroma_db volume + unused `data` dir~~ — removed
- Docker workflow (`grokswarm.cmd`) still runs via Docker; all dev happens locally on Windows now

## Note on web and x search
~~xAI's built-in web_search and x_search tools run server-side and would be more reliable than Playwright for web queries. They require switching from the Chat Completions API to the xAI Responses API or the xai_sdk package — a worthwhile architectural shift when you're ready.~~
**DONE in v0.17.0.** `web_search` and `x_search` now call the xAI Responses API directly via `httpx`. The main chat loop stays on Chat Completions with function calling; search tools make a separate Responses API call internally. Playwright is still used for direct page control (fetch_page, screenshot, links).

## Next Priorities (in order)
1. **v0.26.x — Dogfooding hardening**
	- Run repeated `/self-improve` cycles on real tasks and collect failure patterns
	- Add targeted regression tests for any new edge-cases discovered
2. **v0.27.0 — Multi-agent quality tuning**
	- Improve supervisor decomposition prompts and bus message formatting
	- Add optional per-expert message filters (recipient/kind policies)
3. **v0.28.0 — UX polish**
	- Expand dashboard panels (active tool rounds, test-fix state)
4. See [TODO.md](TODO.md) for the full task tracker and [FEEDBACK_EVAL.md](FEEDBACK_EVAL.md) for the detailed analysis

## Non-Goals (for now)
- Running completely headless / 24/7 daemon without a terminal open
- Using external memory services (Mem0, Supermemory, etc.)

## Success Criteria
When I type grokswarm and start typing, it should feel like I have a full team of expert AIs sitting next to me — fast, capable, and always ready.

---

**This file will be updated as we progress.**

