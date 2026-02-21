# Feedback Evaluation — All Three Reviews (v0.22.0)

**Date:** February 20, 2026
**Codebase:** GrokSwarm v0.22.0 (~3499 lines, main.py)
**Reviewers:** Grok (self-review), GPT (ChatGPT), Gemini, Copilot (this document)
**Revision 3:** Complete rewrite for v0.22.0 — all prior P0/P1/P2 items shipped; evaluating new findings only

---

## What Changed Since the v0.17.0 Eval

The previous eval (Revision 2) was written against v0.17.0 (~2750 lines). Since then, **all P0, P1, and P2 items from Revision 2 have been implemented**:

- **Security (shipped):** S1 dangerous-command denylist, S2 symlink escape blocking, S3 SSRF guard, S4 secret redaction, S5 read-only toggle, S6 expanded edit preview
- **Correctness (shipped):** C1 fnmatch for IGNORE_DIRS, C2 AnnAssign support, C4 compaction tool-call boundary fix, C5 `/write` naming fix, C6 file-size warning, C7 AST parse warnings, C8 expert prompt fix, C9 incremental context refresh
- **Performance (shipped):** P1 context cache with mtime, P2 ripgrep fallback, P3 token-aware compaction, P4 symbol index trim to 15/file
- **UX (shipped):** U1/G1 project switcher, U2 /doctor, U3 analyze_image, U4 .grokswarm.yml config, G2 Playwright auto-install, G3 multi-level undo, G4 trust mode, G5 auto-test cap (3 attempts)
- **Architecture (shipped):** A2 Playwright atexit timeout, A7 JSON repair
- **Phase -1 (shipped):** 0.A test_grokswarm.py (135 tests), 0.B /self-improve shadow copy

This eval is therefore a **fresh assessment of v0.22.0**, not an update of the old list.

---

## Source Credibility Summary (v0.22.0 Reviews)

| Reviewer | Tone | Signal-to-Noise | Accuracy | Key Strength |
|---|---|---|---|---|
| **Grok** | Measured, architectural | **Medium-High** — focused on structural gaps and roadmap status, not specific code bugs | Accurate on what it covers, but missed every runtime bug | Best at roadmap tracking and architectural direction |
| **GPT** | Analytical, bug-hunter | **High** — found 2 runtime-crashing bugs and 4 real consistency issues | Every high/medium finding verified against source | Best at finding actual code-level correctness bugs |
| **Gemini** | Security-focused, practical | **Very High** — found the most dangerous cross-project state contamination bugs that no other reviewer caught | All findings verified; includes concrete fix code | Best at state management and security edge cases |

---

## Verified Bug List — Cross-Referenced Against Source

I read all 3499 lines of main.py and independently verified every claim. Here is the consolidated, deduplicated list of real issues.

### CRITICAL (runtime crashes or data corruption)

| # | Bug | Found By | Verified | Details |
|---|---|---|---|---|
| **B1** | **`/trust` and `/readonly` crash with `UnboundLocalError`** | GPT | **YES** | In `chat()`, lines ~3139 and ~3145 assign to `_trust_mode` and `_session_read_only` without `global` declarations. Python treats them as locals throughout the function, so `not _trust_mode` on the RHS raises `UnboundLocalError`. Note: `global _self_improve_active` IS declared at line ~3181 for `/self-improve`, proving the developer knows the pattern — just missed it for these two. |
| **B2** | **`/undo` restores post-edit content (no-op)** | GPT | **YES** | `_execute_tool` appends to `_edit_history` AFTER running the tool handler (line ~2565). By then, `edit_file`/`write_file` has already modified the file. `p.read_text()` captures the NEW content. `/undo` then "restores" the same content that's already there. |
| **B3** | **Undo stack leaks across projects** | Gemini | **YES** | `_switch_project()` (line ~2176) does NOT clear `_edit_history`. If you edit files in Project A, switch to Project B, then `/undo`, it pops a Project A edit and writes that content into whatever path resolves in Project B. Could silently corrupt files. |
| **B4** | **Test-fix loop leaks across projects** | Gemini | **YES** | `_switch_project()` does NOT reset `_test_fix_state`. After switching projects, the next edit triggers auto-retest using the OLD project's test command in the NEW project's directory. |

### HIGH (security gaps or incorrect behavior)

| # | Bug | Found By | Verified | Details |
|---|---|---|---|---|
| **B5** | **SSRF guard misses cloud metadata IP `169.254.169.254`** | Gemini | **YES** | `_SSRF_BLOCKED` regex blocks localhost, 127.x, 10.x, 172.16-31.x, 192.168.x but not `169.254.x.x` (link-local). On AWS/GCP/Azure, a prompt injection could fetch `http://169.254.169.254/latest/meta-data/` to steal instance credentials. |
| **B6** | **`/context refresh` bypasses cache** | GPT | **YES** | Line ~3025: `scan_project_context(PROJECT_DIR)` called directly, skipping `scan_project_context_cached()`. The fresh scan is also NOT saved to cache, so the next startup reloads stale cache data. |
| **B7** | **Symbol cap mismatch: 15 (full scan) vs 40 (incremental)** | GPT | **YES** | `_build_deep_symbol_index` caps at 15 symbols/file (line ~449). `_incremental_context_refresh` caps at 40 (line ~598). After a few edits, the prompt grows beyond what a clean startup would produce. Context shape becomes inconsistent. |

### MEDIUM (suboptimal but not breaking)

| # | Bug | Found By | Verified | Details |
|---|---|---|---|---|
| **B8** | **Cache invalidation only samples 100 files** | GPT, Gemini | **YES** | `_project_mtime()` passes `max_files=100` to `_iter_project_files`. Changes beyond file #100 won't invalidate the cache. Gemini correctly argues that since `_iter_project_files` already prunes ignored dirs, bumping this to 100,000 is cheap. |
| **B9** | **`run_tests` blocked in read-only mode** | GPT | **YES** | `MUTATING_TOOLS` set in `_execute_tool` includes `run_tests`. Tests are non-mutating reads. Blocking them in "exploration" mode is surprising. |
| **B10** | **`/self-improve` promotion skips full test suite** | Grok | **YES** | Promotion gate only runs `py_compile` (line ~3209). If `test_grokswarm.py` exists and passes before self-improve but the shadow introduces regressions, promotion still succeeds as long as syntax is valid. |
| **B11** | **`MUTATING_TOOLS` set rebuilt every tool call** | GPT | **YES** | The set literal is inside `_execute_tool()`, reconstructed on every invocation. Should be a module-level constant. Minor perf overhead. |
| **B12** | **`/undo` doesn't handle newly created files** | Gemini | **Partial** | This is mostly a consequence of B2. Even when `write_file` creates a new file, the undo snapshot captures the new content (useless). True deletion-on-undo for new files would need a `(path, None)` sentinel. Low priority — accepted limitation. |

### LOW (polish, cosmetic)

| # | Issue | Found By | Details |
|---|---|---|---|
| **L1** | `/project` lacks tab-completion for paths | GPT | `SwarmCompleter.PATH_COMMANDS` doesn't include `project`. Easy fix. |
| **L2** | `/doctor` could check browser binary availability | GPT | Currently checks `import playwright` but not if chromium is installed. |
| **L3** | Global state creep — 5+ module-level mutables | Grok | `_trust_mode`, `_session_read_only`, `_self_improve_active`, `_edit_history`, `_test_fix_state`, `_pending_write_count`. Moving to a `SwarmState` class would prevent future `global` bugs. |
| **L4** | Naming drift in docs vs code | GPT | Minor inconsistencies between help text descriptions and actual behavior. |

---

## Reviewer Agreement Matrix

Shows which bugs each reviewer independently identified:

| Bug | GPT | Gemini | Grok | Copilot |
|---|---|---|---|---|
| B1 (global UnboundLocalError) | **Found** | — | — | Confirmed |
| B2 (undo captures post-edit) | **Found** | — | — | Confirmed |
| B3 (undo leaks cross-project) | — | **Found** | — | Confirmed |
| B4 (test-fix leaks cross-project) | — | **Found** | — | Confirmed |
| B5 (SSRF cloud metadata) | — | **Found** | — | Confirmed |
| B6 (context refresh bypasses cache) | **Found** | — | — | Confirmed |
| B7 (symbol cap mismatch 15/40) | **Found** | — | — | Confirmed |
| B8 (mtime samples 100 files) | **Found** | **Found** | — | Confirmed |
| B9 (run_tests blocked readonly) | **Found** | — | — | Confirmed |
| B10 (self-improve skips tests) | — | — | **Found** | Confirmed |
| B11 (MUTATING_TOOLS rebuilt) | **Found** | — | — | Confirmed |
| B12 (undo new files) | — | **Found** | — | Confirmed (consequence of B2) |

**GPT found the most issues (7).** Gemini found the most dangerous issues (B3, B4, B5). Grok focused on architecture rather than code-level bugs, catching only B10.

---

## My (Copilot) Independent Assessment

### What's Genuinely Excellent

1. **Safety layering is production-grade.** `_check_ssrf`, `_is_dangerous_command`, `_safe_path` with symlink walk, self-improve mechanical guards, trust mode with dangerous-op carve-outs, secret redaction. This is not one safety check — it's defense in depth.

2. **The agent loop is robust.** Retry with exponential backoff, stream interruption recovery, JSON repair fallback, tool-call ID tracking through compaction, parallel read-only execution, result truncation. These are the details that make the difference between a demo and a daily-driver.

3. **Auto-test-fix cycle is uniquely strong.** The automatic lint → retest → re-edit flow after `edit_file` is something most AI coding tools don't have. The 3-attempt cap (G5) prevents infinite loops. This is the single best feature for practical development.

4. **Single-file architecture is correct for this stage.** At 3499 lines, a single file is still navigable. The `@grokswarm/` self-knowledge prefix proves the model can read and reason about its own source in one shot. Refactoring into packages would break this.

### Where the Reviews Disagree — My Verdict

| Disagreement | GPT | Gemini | Grok | My Verdict |
|---|---|---|---|---|
| **Refactor into package?** | Didn't suggest | Didn't suggest | Suggest state class | **Not yet.** State class (L3) is the right intermediate step. Full package split is premature below ~5000 lines. |
| **`run_tests` in readonly?** | Block it | Not mentioned | Not mentioned | **Allow it.** Tests are read-only operations. The "readonly" mental model means "don't change my files", not "don't run anything". |
| **Cache mtime sample size** | Raise cautiously | Remove cap entirely | Not mentioned | **Raise to 10,000.** Gemini's "100,000" is correct that it's cheap, but 10,000 covers virtually all real projects while providing a safety bound against pathological trees. |
| **Self-improve promotion gate** | Not mentioned | Not mentioned | Run full test suite | **Agree with Grok.** If `test_grokswarm.py` exists, the promotion block should run `pytest` before allowing copy. 5 lines of code, prevents regressions. |

### What Nobody Caught

1. **`_compact_conversation` summary can lose the user's original task.** If the conversation summary doesn't preserve the original request verbatim, the model may forget what it was asked to do after compaction. The current implementation asks the LLM to summarize "key decisions, files modified, tasks completed" but doesn't explicitly preserve the original user intent. This matters most for long multi-step tasks.

2. **No timeout on `_iter_project_files`.** A project with a very deep ignored-but-not-gitignored directory tree (e.g., a mounted file system) could cause `scan_project_context` to hang. The `max_files` cap mitigates this for file count, but deeply nested empty directories would still be walked.

3. **`run_expert` swallows the expert's output into memory but doesn't return it to the calling conversation.** The `/swarm` command's `run_expert` calls produce printed output and save to memory, but the main conversation never receives the expert's analysis as context for follow-up. The swarm result is effectively fire-and-forget.

---

## Prioritized Fix Plan (v0.22.1)

### Tier 1 — Fix Now (runtime crashes + security)

| # | Fix | Source | Est. Lines |
|---|---|---|---|
| B1 | Add `global _trust_mode, _session_read_only` in `chat()` | GPT | 1 |
| B2 | Capture file content BEFORE tool handler runs in `_execute_tool` | GPT | ~8 |
| B3+B4 | Clear `_edit_history` and reset `_test_fix_state` in `_switch_project()` | Gemini | 3 |
| B5 | Add `\|169\.254\.\d+\.\d+` to `_SSRF_BLOCKED` regex | Gemini | 1 |

**Total: ~13 lines, eliminates all CRITICAL and security bugs.**

### Tier 2 — Fix Soon (consistency + correctness)

| # | Fix | Source | Est. Lines |
|---|---|---|---|
| B6 | Make `/context refresh` call `scan_project_context_cached()` (force-invalidate first) or scan fresh + save to cache | GPT | 3 |
| B7 | Align `_incremental_context_refresh` symbol cap from 40 → 15 | GPT | 1 |
| B8 | Raise `_project_mtime` max_files from 100 → 10,000 | GPT+Gemini | 1 |
| B9 | Remove `run_tests` from `MUTATING_TOOLS` set | GPT | 1 |
| B10 | Add pytest gate to `/self-improve` promotion (run tests if test file exists) | Grok | 5 |
| B11 | Hoist `MUTATING_TOOLS` to module-level constant | GPT | 3 |

**Total: ~14 lines.**

### Tier 3 — Nice to Have (defer to v0.23.0+)

| # | Item | Source |
|---|---|---|
| L1 | Tab-complete `/project` paths | GPT |
| L2 | `/doctor` check browser binary | GPT |
| L3 | Migrate globals to `SwarmState` class | Grok |
| L4 | Docs/naming harmonization | GPT |
| B12 | `/undo` deletion for new files | Gemini |

### Rejected

| Item | Source | Why |
|---|---|---|
| Refactor into package | (Nobody suggested for v0.22.0) | Single-file is a feature at this scale |
| SQLite coordination bus | Grok (from prior eval) | Still P4 — no multi-agent concurrency yet |
| TUI dashboard | Grok (from prior eval) | Rich console output is sufficient |
| Fuzzy-match edit_file | (From prior eval) | Silent corruption risk remains |

---

## Overall Grade

| Aspect | Grade | Notes |
|---|---|---|
| Safety | **A** | SSRF, symlink walk, dangerous-cmd denylist, self-improve guard, secret redaction. Only gap: cloud metadata IP (B5). |
| Correctness | **B+** | 4 real bugs (B1-B4), all <5 lines to fix. The global declaration bug is embarrassing but trivial. |
| Performance | **A-** | Context caching, ripgrep fallback, parallel read-only tools, token-aware compaction. Cache mtime sampling (B8) is the only gap. |
| UX / Ergonomics | **A** | 25 slash commands, tab completion, trust mode, project switching, session persistence, auto-test-fix. Very polished for a personal tool. |
| Architecture | **A-** | Clean single-file design at 3499 lines. Global state creep (L3) is the main concern but not blocking. |
| **Overall** | **A-** | Ship Tier 1 fixes (~13 lines) and this is a solid A. The codebase has evolved from a prototype to a production-quality daily-driver with genuine safety depth. |

---

## Summary

**Gemini delivered the highest-impact review** — the cross-project state contamination bugs (B3, B4) are the most dangerous findings because they can silently corrupt files without any error message. The SSRF cloud metadata gap (B5) is the most security-relevant addition.

**GPT delivered the highest-volume review** — 7 verified findings including the 2 runtime-crashing global declaration bugs. Methodical and thorough.

**Grok delivered the best architectural perspective** — correctly identified that self-improve promotion needs a test gate, and that global state should migrate to a class. Less useful for code-level bug hunting.

**Bottom line:** The v0.22.0 codebase is strong. 13 lines of fixes (Tier 1) eliminate all critical bugs. 14 more lines (Tier 2) clean up consistency issues. After that, only polish remains.
