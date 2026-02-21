# Full Review — main.py (v0.22.0)

Date: 2026-02-20

## Executive Summary

`main.py` is feature-rich and thoughtfully evolved: strong safety posture (SSRF + dangerous shell denylist + self-improve guard), good operational ergonomics (parallel read-only tools, auto-lint, auto-retest, context compaction), and practical UX additions (`/trust`, `/project`, `/doctor`, `@grokswarm/` self-knowledge).

The codebase is productive, but there are **2 high-priority correctness bugs** introduced in the latest feature wave, and a handful of **medium-priority consistency/perf issues**.

---

## What’s Working Well

1. **Safety layering is solid**
	- `_check_ssrf`, `_is_dangerous_command`, `_safe_path`, and self-improve shell/file guards form a good defense-in-depth baseline.
	- Approval gates are present for risky operations and dangerous commands require stricter confirmation.

2. **Agent loop robustness is strong**
	- `_api_call_with_retry`, stream retries, tool-call parsing fallback (`_repair_json`), result truncation, and KeyboardInterrupt recovery are well thought through.

3. **Developer UX is notably strong**
	- `edit_file` preview + multi-edit flow is practical.
	- Auto-lint + auto-retest loop catches regressions early.
	- Tool timing and parallel read-only execution improve feedback and throughput.

4. **Cross-project ergonomics improved meaningfully**
	- External project support + project context caching + recent-project tracking + visible project path are all good additions.

---

## High-Priority Issues (Fix First)

### 1) `/trust`, `/readonly` mutate globals without `global` declaration in `chat()`

**Problem**
- Inside `chat()`, these branches assign to `_trust_mode` and `_session_read_only`:
  - `elif cmd == "trust": _trust_mode = not _trust_mode`
  - `elif cmd == "readonly": _session_read_only = not _session_read_only`
- Because assignment occurs in function scope without `global`, Python treats these as locals and the read on RHS causes `UnboundLocalError` at runtime.

**Impact**
- `/trust` and `/readonly` commands can crash/fail on first use.

**Fix**
- Add at top of `chat()`:
  - `global _trust_mode, _session_read_only`
  - (or route through helper functions that encapsulate state mutation)

---

### 2) `edit_file` undo snapshot captures **post-edit** state, not pre-edit state

**Problem**
- `_execute_tool` appends `_edit_history` **after** running the tool handler.
- For `edit_file`, this stores already-modified content.

**Impact**
- `/undo` can restore to the same content (no-op) after an edit.

**Fix**
- Capture file content before applying `edit_file`/`write_file` (pre-state), then push snapshot only if mutation succeeds.

---

## Medium-Priority Issues

### 3) Context refresh inconsistency: `/context refresh` bypasses cache

**Problem**
- Startup uses `scan_project_context_cached()`, but `/context refresh` still calls `scan_project_context()`.

**Impact**
- Inconsistent perf behavior and potentially slower refresh on large projects.

**Fix**
- Use `scan_project_context_cached(PROJECT_DIR)` for refresh too.

---

### 4) Symbol trim mismatch between full scan and incremental refresh

**Problem**
- `_build_deep_symbol_index` now caps at 15 symbols/file, but `_incremental_context_refresh` still caps at 40.

**Impact**
- Prompt shape drifts after edits; context becomes inconsistent over time.

**Fix**
- Align incremental cap to 15 and include same truncation marker logic.

---

### 5) Cache invalidation can miss changes due to mtime sampling cap

**Problem**
- `_project_mtime()` checks only first 100 walked files.

**Impact**
- File changes outside sampled set may not invalidate cache immediately.

**Fix options**
- Increase sample cap substantially, or
- Include directory mtimes/manifest hash (better), or
- Track mtimes for files included in context artifacts.

---

### 6) Read-only guard blocks `run_tests`

**Problem**
- `run_tests` is included in `MUTATING_TOOLS` under `_session_read_only`.

**Impact**
- Surprising behavior: “read-only” mode also blocks non-mutating validation.

**Recommendation**
- Consider allowing `run_tests` in read-only mode; keep shell and write/edit/git-mutation blocked.

---

## Low-Priority / Polish

1. `/project` is functional but could benefit from tab-completion over recent projects.
2. `MUTATING_TOOLS` set is rebuilt per call inside `_execute_tool`; move to module constant.
3. `_run_doctor()` import check for `playwright` is fine, but could include browser binary availability check too.
4. Slight naming drift in docs/comments (`grok init` vs `/doctor`) should be harmonized.

---

## Suggested Next Patch (small, high ROI)

1. Fix `global _trust_mode, _session_read_only` in `chat()`.
2. Fix undo pre-state capture for `edit_file` and `write_file`.
3. Make `/context refresh` use cached scanner.
4. Align `_incremental_context_refresh` symbol cap to 15.

These four changes would remove the largest correctness risks while keeping behavior consistent with recent design goals.

---

## Overall Grade

**A- (very strong practical tool, with a couple correctness regressions to clean up).**

Once the two high-priority fixes are in, this is production-ready for daily use in the current architecture style.
