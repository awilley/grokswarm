# `/self-improve` Dogfooding & Stress Tests

This document tracks benchmark tasks designed to evaluate and harden the `/self-improve` autonomous coding loop. By running these standardized tests, we can measure success rates, token usage, and identify failure patterns (e.g., context loss, test breakage, infinite loops).

## Level 1: Simple Refactoring (Low Risk)

### Test 1.1: Class Extraction
**Prompt:** `Extract the SwarmBus class from main.py into a new file called bus.py and update all the imports in main.py to use it.`
**Success Criteria:**
- `bus.py` is created containing the `SwarmBus` class.
- `main.py` imports `SwarmBus` from `bus.py`.
- `test_grokswarm.py` passes (tests might need import updates if they mock/import `SwarmBus` directly).
**Status:** ⏳ Pending
**Notes:** 

## Level 2: Feature Addition (Medium Risk)

### Test 2.1: New Slash Command
**Prompt:** `Add a /history slash command that prints the last 5 user commands from the current session. Update SLASH_COMMANDS and the help text.`
**Success Criteria:**
- `/history` is added to `SLASH_COMMANDS`.
- Command logic is implemented in the chat loop.
- `test_grokswarm.py` passes (specifically `TestSlashCommands.test_has_all_expected_commands` and `test_command_count` will need updates).
**Status:** ⏳ Pending
**Notes:** 

## Level 3: Algorithmic / Logic Changes (High Risk)

### Test 3.1: Token Heuristic Update
**Prompt:** `Modify _estimate_tokens in main.py to use a more accurate heuristic: split the content by whitespace and multiply the word count by 1.3, rather than just dividing characters by 4. Update the corresponding tests in test_grokswarm.py to match the new math.`
**Success Criteria:**
- `_estimate_tokens` logic is updated.
- `test_grokswarm.py` is updated to assert the new expected token counts.
- All tests pass.
**Status:** ⏳ Pending
**Notes:** 

## Level 4: Multi-Agent / Swarm Interaction (Complex)

### Test 4.1: New Expert Integration
**Prompt:** `Create a new expert called debugger.yaml in the experts directory that specializes in reading Python tracebacks. Then, modify the supervisor prompt in main.py to explicitly mention the debugger expert as an option.`
**Success Criteria:**
- `experts/debugger.yaml` is created with appropriate `name`, `mindset`, and `objectives`.
- `main.py` supervisor prompt is updated.
- Tests pass.
**Status:** ⏳ Pending
**Notes:** 

## Level 5: Self-Correction (Resilience)

### Test 5.1: Intentional Bug Fix
**Setup:** Intentionally introduce a logic bug in a non-critical function in `main.py` (e.g., break `_repair_json`).
**Prompt:** `Run the test suite, identify why it is failing, and fix the bug in main.py.`
**Success Criteria:**
- Agent correctly runs `pytest`.
- Agent parses the failure output.
- Agent edits `main.py` to fix the bug.
- Tests pass.
**Status:** ⏳ Pending
**Notes:** 
