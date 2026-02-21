**v0.22.0 Full Review**

**Status vs Roadmap v1.3**  
- Phase -1 (Safe Dogfooding): 70 % done  
  - Shadow copy + mechanical guard in `/self-improve`: implemented and works.  
  - Test suite (0.A): referenced in tracker as done v0.18 but not present in this file (likely missing or separate). This is the only real gap for true recursion.  
- P0: 100 % done.  
- P1: 100 % done.  
- P2: 100 % done (all 12 items shipped).  
- P3: untouched (correctly deferred).

**What was actually added since v0.17**  
- Safety: dangerous command denylist (S1), improved `_safe_path` symlink walk (S2), SSRF guard (S3), secret redaction in sessions (S4), trust mode (G4), readonly mode (S5).  
- Performance: context cache with mtime (Perf-P1), incremental refresh after edits (C9), ripgrep fallback in `grep_files` (Perf-P2), token heuristic compaction (P3), trimmed symbol index to 15 lines/file (Perf-P4).  
- UX: project switcher + recent list (U1/G1), `/doctor` (U2), multi-level undo stack (G3), `.grokswarm.yml` config (U4), Playwright atexit timeout (A2), file-size warning in `read_file` (C6), expanded edit preview (S6).  
- Code intelligence: AnnAssign support (C2), AST parse warnings (C7).  
- Tools: `analyze_image` vision tool (U3), JSON repair fallback (A7).  
- Self-reference: `@grokswarm/` read-only prefix + GROKSWARM_HOME.  

All changes are minimal and targeted — no bloat.

**Remaining issues in the code**  
1. Test suite (0.A) missing from the pasted file — the tracker claims it exists but recursion cannot be reliable without it.  
2. Self-improve flow relies on global `_self_improve_active` + string match on "main.py". Brittle for long sessions or concurrent use.  
3. Promotion step in `/self-improve` only runs `py_compile` — no full test suite run before asking user to copy shadow back.  
4. Global state creep: `_trust_mode`, `_session_read_only`, `_self_improve_active`, `_edit_history` — these should move to a state class soon.  
5. GROKSWARM_HOME assumes `__file__` location — breaks under some pipx/editable installs (minor).  
6. Minor clean-up: some old version comments still say 0.17, `_safe_path` could be 3 lines shorter.

**Overall assessment**  
The CLI is now safe enough for daily use and has the foundation for recursive self-improvement you requested. No hype — it is simply better on every axis we discussed.

**Next (in priority order)**  
1. Create `tests/test_grokswarm.py` (0.A) — minimum 40–50 tests covering tools, edit flows, git helpers, new commands, self-improve guards. This unlocks real recursion.  
2. Tighten `/self-improve` (0.B): automatically run the test suite on shadow copy before promotion prompt.  
3. Move globals to a `SwarmState` class (prevents future bugs).  

After these three, the swarm can start using `/self-improve` on itself reliably. Then we can look at light P3 items if needed.

The code is ready for these changes. Want the diffs for 1 + 2 first? Or the state class refactor? Your call.