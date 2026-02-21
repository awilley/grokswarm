# GrokSwarm — Task Tracker

**Last updated:** February 20, 2026
**Current version:** v0.25.0
**Source:** [FEEDBACK_EVAL.md](FEEDBACK_EVAL.md) — consolidated from reviews by Grok, GPT, Gemini, Copilot

---

## v0.22.1 Bug Fixes (from FEEDBACK_EVAL Tier 1 + Tier 2)

### Tier 1 — Critical / Security (all DONE)

- [x] **B1** — Add `global _trust_mode, _session_read_only` in `chat()` — DONE v0.22.1
  - `/trust` and `/readonly` previously crashed with `UnboundLocalError` (GPT)
- [x] **B2** — Capture pre-edit file content BEFORE tool handler runs — DONE v0.22.1
  - `/undo` previously restored post-edit state (no-op). Now captures true pre-state (GPT)
- [x] **B3+B4** — Clear `_edit_history` and reset `_test_fix_state` in `_switch_project()` — DONE v0.22.1
  - Prevents undo stack and test-fix loop leaking across projects (Gemini)
- [x] **B5** — Add `169.254.x.x` to `_SSRF_BLOCKED` regex — DONE v0.22.1
  - Blocks cloud metadata IP (AWS/GCP/Azure credential theft vector) (Gemini)

### Tier 2 — Consistency / Correctness (all DONE)

- [x] **B6** — `/context refresh` now saves to cache after scanning — DONE v0.22.1
  - Previously bypassed cache entirely; next startup loaded stale data (GPT)
- [x] **B7** — Align `_incremental_context_refresh` symbol cap from 40 → 15 — DONE v0.22.1
  - Matches `_build_deep_symbol_index` cap; prevents prompt drift after edits (GPT)
- [x] **B8** — Raise `_project_mtime()` max_files from 100 → 10,000 — DONE v0.22.1
  - Cache invalidation now covers virtually all real projects (GPT + Gemini)
- [x] **B9** — Remove `run_tests` from read-only blocked tools — DONE v0.22.1
  - Tests are non-mutating; now allowed in `/readonly` mode (GPT)
- [x] **B10** — Add pytest gate to `/self-improve` promotion — DONE v0.22.1
  - Runs `pytest test_grokswarm.py -x -q` before allowing shadow → main.py copy (Grok)
- [x] **B11** — Hoist `MUTATING_TOOLS` to module-level `_READONLY_BLOCKED_TOOLS` — DONE v0.22.1
  - No longer rebuilt on every `_execute_tool` call (GPT)

---

## Tier 3 — Polish (all DONE)

- [x] **L3** — Migrate mutable session globals to `SwarmState` dataclass — DONE v0.25.0
- [x] **L4** — Docs/naming harmonization across `SLASH_COMMANDS` + `/help` text — DONE v0.25.0
- [x] **A8** — Token Usage & Cost Metrics panel in dashboard + `/metrics` command — DONE v0.26.0

## v0.24.0 Features (all DONE)

- [x] **A3** — Dynamic tool registration — skills saved via `create_skill` are auto-registered as callable LLM tools (`skill_{name}`) — DONE v0.24.0
- [x] **L2** — `/doctor` checks for chromium browser binary (not just `import playwright`) — DONE v0.24.0

## v0.23.0 Features (all DONE)

- [x] **B12** — `/undo` deletes newly created files (stores `(path, None)` sentinel) — DONE v0.23.0
- [x] **L1** — Tab-complete `/project` paths (directory completer + recent projects + subcmds) — DONE v0.23.0
- [x] **A6** — Isolated test validation for `/self-improve` (copies shadow + tests to temp dir) — DONE v0.23.0

## P3 — Defer

- [x] **A4** — SQLite coordination bus for multi-agent messaging — DONE v0.25.0
- [x] **A5** — Live TUI dashboard (`grokswarm dashboard`) — DONE v0.25.0

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
