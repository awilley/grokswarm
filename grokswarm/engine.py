"""_execute_tool, _stream_with_tools, compaction, API retry, JSON repair."""

import re
import json
import time
import asyncio
from pathlib import Path

from rich.markdown import Markdown

import grokswarm.shared as shared
from grokswarm.context import _safe_path, _incremental_context_refresh
from grokswarm.tools_registry import (
    TOOL_SCHEMAS, TOOL_DISPATCH, READ_ONLY_TOOLS,
    _FILE_MUTATION_TOOLS, _READONLY_BLOCKED_TOOLS,
)
from grokswarm.tools_test import _lint_file, _run_tests_raw, _detect_test_framework, TEST_COMMANDS, MAX_TEST_FIX_ATTEMPTS

MAX_TOOL_ROUNDS = 10
MAX_TOOL_RESULT_SIZE = 12000
MAX_STREAM_RETRIES = 2

MAX_CONVERSATION_MESSAGES = 40
COMPACTION_THRESHOLD = 50
COMPACTION_KEEP_RECENT = 20
COMPACTION_TOKEN_LIMIT = 100_000

# Base set of tools safe to run in a background thread (no user approval needed).
# Trust mode / agent mode adds write tools dynamically in _execute_tool.
_THREADABLE_TOOLS_BASE = frozenset({
    "list_directory", "read_file", "search_files", "grep_files",
    "git_status", "git_diff", "git_log", "git_show_file", "git_blame",
    "fetch_page", "extract_links",
    "find_symbol", "find_references",
    "web_search", "x_search",
    "list_registry", "list_agents", "check_messages",
})

_THREADABLE_TOOLS_TRUSTED = _THREADABLE_TOOLS_BASE | frozenset({
    "run_shell", "run_tests", "write_file", "edit_file",
    "git_commit", "git_checkout", "git_branch", "git_stash", "git_init",
    "screenshot_page", "create_expert", "create_skill",
})


def _estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += len(content) // 4
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                total += len(tc.get("function", {}).get("arguments", "")) // 4
    return total


def _repair_json(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        s = "\n".join(lines).strip()
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


async def _compact_conversation(conversation: list) -> list:
    system = [m for m in conversation if m["role"] == "system"]
    others = [m for m in conversation if m["role"] != "system"]
    if len(others) <= COMPACTION_KEEP_RECENT:
        return conversation
    old_messages = others[:-COMPACTION_KEEP_RECENT]
    recent_messages = others[-COMPACTION_KEEP_RECENT:]
    while recent_messages and recent_messages[0]["role"] == "tool":
        old_messages.append(recent_messages.pop(0))
    while (recent_messages and recent_messages[0]["role"] == "assistant"
           and recent_messages[0].get("tool_calls")
           and len(recent_messages) > 1
           and recent_messages[1]["role"] == "tool"):
        old_messages.append(recent_messages.pop(0))
        while recent_messages and recent_messages[0]["role"] == "tool":
            old_messages.append(recent_messages.pop(0))
    old_text_parts = []
    for m in old_messages:
        role = m["role"]
        content = m.get("content", "")
        if role == "tool":
            snippet = content[:150].replace("\n", " ") if content else "(empty)"
            old_text_parts.append(f"  tool result: {snippet}")
            continue
        if content:
            old_text_parts.append(f"{role}: {content[:200]}")
        if m.get("tool_calls"):
            tools_used = [tc["function"]["name"] for tc in m["tool_calls"]]
            old_text_parts.append(f"  (used tools: {', '.join(tools_used)})")
    if not old_text_parts:
        return conversation
    old_text = "\n".join(old_text_parts)
    if len(old_text) > 4000:
        old_text = old_text[:4000] + "\n... (truncated)"
    try:
        from grokswarm import llm
        chat = llm.create_chat(shared.MODEL, max_tokens=500)
        llm.populate_chat(chat, [
            {"role": "system", "content": "Summarize this conversation history into a concise paragraph. Capture key decisions, files modified, tasks completed, and important context. Be factual and brief."},
            {"role": "user", "content": old_text}
        ])
        summary_response = await shared._api_call_with_retry(
            lambda: chat.sample(),
            label="Compaction"
        )
        usage = llm.extract_usage(summary_response)
        if usage["prompt_tokens"] or usage["completion_tokens"]:
            from grokswarm.agents import _record_usage
            _record_usage(shared.MODEL, usage["prompt_tokens"], usage["completion_tokens"],
                          usage["cached_tokens"])
        summary = (summary_response.content or "").strip()
    except Exception:
        return system + others[-MAX_CONVERSATION_MESSAGES:]
    summary_msg = {"role": "user", "content": f"[CONVERSATION SUMMARY -- earlier messages compacted]\n{summary}"}
    ack_msg = {"role": "assistant", "content": "Understood. I have the context from our earlier conversation. Let's continue."}
    return system + [summary_msg, ack_msg] + recent_messages


async def _trim_conversation(conversation: list) -> list:
    system = [m for m in conversation if m["role"] == "system"]
    others = [m for m in conversation if m["role"] != "system"]
    est_tokens = _estimate_tokens(conversation)
    if len(others) > COMPACTION_THRESHOLD or est_tokens > COMPACTION_TOKEN_LIMIT:
        shared.console.print(f"[swarm.dim]  ~ compacting conversation history (~{est_tokens:,} tokens, {len(others)} msgs)...[/swarm.dim]")
        return await _compact_conversation(conversation)
    if len(others) > MAX_CONVERSATION_MESSAGES:
        others = others[-MAX_CONVERSATION_MESSAGES:]
        while others and others[0]["role"] == "tool":
            others.pop(0)
    return system + others


async def _suspend_prompt():
    """Suspend prompt_toolkit so approval prompts can safely use the terminal.

    Call once before a batch of tools that may need user approval.
    Pair with _resume_prompt() after the batch completes.
    No-op if no REPL is running or already suspended.
    """
    app = shared._toolbar_app_ref
    if not app or not getattr(app, "is_running", False):
        return
    if shared._is_prompt_suspended:
        return  # already suspended

    async with shared._suspend_lock:
        shared._toolbar_status = ""
        shared._toolbar_suspended = True
        shared._is_prompt_suspended = True

        try:
            shared._saved_prompt_text = app.current_buffer.text
            app.exit(result="__MAGIC_SUSPEND__")
        except Exception:
            pass

        await shared._prompt_suspend_event.wait()
        shared._prompt_suspend_event.clear()

        # Clear terminal line so approval prompts start on a fresh row
        import sys
        sys.__stdout__.write("\r\033[K")
        sys.__stdout__.flush()


def _resume_prompt():
    """Resume prompt_toolkit after a batch of tools completes.

    No-op if not currently suspended.
    """
    if not shared._is_prompt_suspended:
        return
    shared._is_prompt_suspended = False
    shared._toolbar_suspended = False
    shared._prompt_resume_event.set()


async def _suspend_prompt_and_run(func):
    """Suspend prompt_toolkit, run *func* in a thread, then resume.

    Legacy wrapper -- used when a single tool needs approval outside
    of the batch suspension path.  If prompt is already suspended
    (batch mode), just runs in a thread without re-suspending.
    """
    app = shared._toolbar_app_ref
    if not app or not getattr(app, "is_running", False):
        return await asyncio.to_thread(func)

    if shared._is_prompt_suspended:
        # Already suspended by batch -- just run the function
        return await asyncio.to_thread(func)

    # One-off suspend/resume (shouldn't normally happen with batch mode)
    await _suspend_prompt()
    try:
        return await asyncio.to_thread(func)
    finally:
        _resume_prompt()


def _handle_test_failure(name: str, args: dict, result_str: str) -> str:
    """Track test failures and set up auto-retest state. Returns appended result_str."""
    if name == "run_tests" and result_str.startswith("[FAIL]"):
        cmd = args.get("command") or ""
        if not cmd:
            fw = _detect_test_framework()
            if fw:
                cmd = TEST_COMMANDS[fw]["cmd"]
        pattern = args.get("pattern")
        if pattern and cmd:
            cmd += f" -k {pattern}" if "pytest" in cmd else f" {pattern}"
        if cmd:
            shared.state.test_fix_state["cmd"] = cmd
            shared.state.test_fix_state["attempts"] = 0
            result_str += f"\n\n[AUTO-TEST FAILURE] Tests failed. You MUST: 1) analyze the failure output above, 2) fix the code with edit_file, 3) the system will auto-rerun tests after your edit to verify the fix. Do NOT proceed to other tasks until tests pass."
    elif name == "run_tests" and result_str.startswith("[PASS]"):
        if shared.state.test_fix_state["cmd"]:
            if shared.state.verbose_mode:
                shared.console.print(f"  [swarm.accent]\u2714 test-fix cycle complete[/swarm.accent]")
            shared._log("test-fix cycle complete")
        shared.state.test_fix_state["cmd"] = None
        shared.state.test_fix_state["attempts"] = 0
    return result_str


async def _maybe_auto_retest(name: str, lint_clean: bool, result_str: str) -> str:
    """If an edit passed lint and tests are in fix-cycle, auto-rerun tests. Returns appended result_str."""
    if not (name in ("edit_file", "write_file")
            and lint_clean
            and shared.state.test_fix_state["cmd"]
            and shared.state.test_fix_state["attempts"] < MAX_TEST_FIX_ATTEMPTS):
        return result_str

    shared.state.test_fix_state["attempts"] += 1
    test_cmd = shared.state.test_fix_state["cmd"]
    attempt = shared.state.test_fix_state["attempts"]
    if shared.state.verbose_mode:
        shared.console.print(f"  [swarm.accent]\u21bb auto-retest ({attempt}/{MAX_TEST_FIX_ATTEMPTS}):[/swarm.accent] [dim]{test_cmd}[/dim]")
    shared._log(f"auto-retest ({attempt}/{MAX_TEST_FIX_ATTEMPTS}): {test_cmd}")
    test_output = await asyncio.to_thread(_run_tests_raw, test_cmd, 60)
    if test_output.startswith("[PASS]"):
        if shared.state.verbose_mode:
            shared.console.print(f"  [swarm.accent]\u2714 tests pass \u2014 fix verified![/swarm.accent]")
        shared._log("tests pass - fix verified")
        shared.state.test_fix_state["cmd"] = None
        shared.state.test_fix_state["attempts"] = 0
        result_str += f"\n\n[AUTO-RETEST PASSED] Tests now pass after your edit. Fix verified. You may proceed."
    else:
        fail_summary = test_output[:3000] if len(test_output) > 3000 else test_output
        result_str += f"\n\n[AUTO-RETEST FAILED] (attempt {attempt}/{MAX_TEST_FIX_ATTEMPTS}) Tests still failing after edit:\n{fail_summary}\nAnalyze the error and fix with another edit_file call."
        if attempt >= MAX_TEST_FIX_ATTEMPTS:
            if shared.state.verbose_mode:
                shared.console.print(f"  [swarm.warning]\u26a0 auto-retest limit reached \u2014 continuing without auto-retest[/swarm.warning]")
            shared._log("auto-retest limit reached")
            result_str += "\n\nAuto-retest limit reached. Use run_tests manually to continue testing."
    return result_str


async def _execute_tool(name: str, args: dict, timed: bool = False):
    t0 = time.perf_counter()

    if shared.state.read_only and name in _READONLY_BLOCKED_TOOLS:
        result_str = f"BLOCKED: Session is in read-only mode. Use /readonly to toggle. Tool '{name}' is not allowed."
        if timed:
            return result_str, time.perf_counter() - t0
        return result_str

    if shared.state.self_improve_active and name in ("edit_file", "write_file"):
        target = args.get("path", "")
        if target and Path(target).name == "main.py" and not str(target).startswith(".grokswarm"):
            result_str = "BLOCKED: During /self-improve, you must only edit the shadow copy at .grokswarm/shadow/main.py, not main.py directly."
            if timed:
                return result_str, time.perf_counter() - t0
            return result_str

    if shared.state.self_improve_active and name == "run_shell":
        shell_cmd = args.get("command", "")
        if re.search(r'(?<!shadow/)\bmain\.py\b', shell_cmd) and '.grokswarm' not in shell_cmd:
            result_str = "BLOCKED: During /self-improve, shell commands must not touch main.py directly. Edit .grokswarm/shadow/main.py instead."
            if timed:
                return result_str, time.perf_counter() - t0
            return result_str

    _pre_edit_content: str | None = None
    _pre_edit_existed: bool = False
    if name in _FILE_MUTATION_TOOLS:
        edit_path = args.get("path", "")
        try:
            p = _safe_path(edit_path)
            if p and p.is_file():
                _pre_edit_content = p.read_text(encoding="utf-8", errors="ignore")
                _pre_edit_existed = True
        except Exception:
            pass

    if shared.state.trust_mode or shared.state.agent_mode > 0:
        threadable = _THREADABLE_TOOLS_TRUSTED
    else:
        threadable = _THREADABLE_TOOLS_BASE
    is_threadable = name in threadable or name.startswith("mcp_")

    handler = TOOL_DISPATCH.get(name)
    if handler:
        try:
            if is_threadable:
                result = await asyncio.to_thread(handler, args)
            elif shared._is_prompt_suspended:
                # Prompt already suspended by batch -- run directly in thread
                result = await asyncio.to_thread(handler, args)
            else:
                # One-off suspension (shouldn't happen with batch mode)
                shared._clear_status()
                result = await _suspend_prompt_and_run(lambda: handler(args))
                if asyncio.iscoroutine(result):
                    result = await result
        except Exception as e:
            result = f"Error: {e}"
    else:
        result = f"Unknown tool: {name}"

    result_str = shared._sanitize_surrogates(str(result))

    if len(result_str) > MAX_TOOL_RESULT_SIZE:
        result_str = result_str[:MAX_TOOL_RESULT_SIZE] + f"\n... (truncated from {len(str(result)):,} chars to {MAX_TOOL_RESULT_SIZE:,})"

    lint_clean = False
    if name in ("edit_file", "write_file") and not result_str.startswith(("Error", "Cancelled", "Access denied", "Edit cancelled")):
        edited_path = _safe_path(args.get("path", ""))
        if edited_path and edited_path.exists():
            lint_err = await asyncio.to_thread(_lint_file, edited_path)
            if lint_err:
                if shared.state.verbose_mode:
                    shared.console.print(f"  [swarm.warning]\u26a0 lint error:[/swarm.warning] [dim]{lint_err[:120]}[/dim]")
                shared._log(f"lint error: {lint_err[:120]}")
                result_str += f"\n\n[AUTO-LINT ERROR] Syntax check failed after edit:\n{lint_err}\nYou MUST fix this immediately using edit_file before proceeding."
            else:
                if shared.state.verbose_mode:
                    shared.console.print(f"  [swarm.accent]\u2714 lint clean[/swarm.accent]")
                shared._log("lint clean")
                lint_clean = True

    if name in _FILE_MUTATION_TOOLS and not result_str.startswith(("Error", "Cancelled", "Access denied", "Edit cancelled")):
        edit_path = args.get("path", "")
        if _pre_edit_existed:
            shared.state.edit_history.append((edit_path, _pre_edit_content))
            if len(shared.state.edit_history) > shared.MAX_EDIT_HISTORY:
                shared.state.edit_history.pop(0)
        elif not _pre_edit_existed:
            shared.state.edit_history.append((edit_path, None))
            if len(shared.state.edit_history) > shared.MAX_EDIT_HISTORY:
                shared.state.edit_history.pop(0)
        shared.state.last_edited_file = edit_path
        _incremental_context_refresh(edit_path)
        shared.state.pending_write_count += 1
        if shared.state.pending_write_count >= shared.AUTO_CHECKPOINT_THRESHOLD:
            if shared.state.verbose_mode:
                shared.console.print(f"  [swarm.warning]\u26a0 {shared.state.pending_write_count} file mutations without a commit \u2014 consider git_commit for a checkpoint[/swarm.warning]")
            shared._log(f"{shared.state.pending_write_count} file mutations without commit")
            result_str += f"\n\n[AUTO-CHECKPOINT] You have made {shared.state.pending_write_count} file edits without committing. Consider using git_commit to create a checkpoint before continuing."
    elif name == "git_commit" and not result_str.startswith("Error"):
        shared.state.pending_write_count = 0

    result_str = _handle_test_failure(name, args, result_str)
    result_str = await _maybe_auto_retest(name, lint_clean, result_str)

    elapsed = time.perf_counter() - t0
    if timed:
        return result_str, elapsed
    return result_str


def _tool_detail(name: str, args: dict) -> str:
    if name in ("write_file", "read_file", "edit_file"):
        return f" \u2192 {args.get('path', '?')}"
    elif name == "list_directory":
        return f" \u2192 {args.get('path', '.')}"
    elif name == "search_files":
        return f" \u2192 {args.get('query', '?')}"
    elif name == "grep_files":
        return f" \u2192 '{args.get('pattern', '?')}' in {args.get('path', '.')}"
    elif name == "run_shell":
        return f" \u2192 {args.get('command', '?')}"
    elif name in ("create_expert", "create_skill"):
        return f" \u2192 {args.get('name', '?')}"
    elif name == "list_registry":
        return ""
    elif name.startswith("git_"):
        if name == "git_commit":
            return f" \u2192 {args.get('message', '?')[:50]}"
        elif name == "git_checkout":
            return f" \u2192 {args.get('target', '?')}"
        elif name == "git_diff":
            return f" \u2192 {args.get('path', 'all')}"
        elif name == "git_branch":
            return f" \u2192 {args.get('name', 'list')}"
        elif name == "git_show_file":
            return f" \u2192 {args.get('path', '?')}@{args.get('ref', 'HEAD')}"
        elif name == "git_blame":
            return f" \u2192 {args.get('path', '?')}"
        elif name == "git_stash":
            return f" \u2192 {args.get('action', 'list')}"
        elif name == "git_init":
            return ""
        return ""
    elif name == "run_tests":
        return f" \u2192 {args.get('command', 'auto-detect')}"
    elif name in ("fetch_page", "extract_links"):
        return f" \u2192 {args.get('url', '?')[:60]}"
    elif name in ("web_search", "x_search"):
        return f" \u2192 '{args.get('query', '?')[:60]}'"
    elif name == "screenshot_page":
        return f" \u2192 {args.get('url', '?')[:40]} \u2192 {args.get('save_path', 'screenshot.png')}"
    elif name == "analyze_image":
        return f" \u2192 {args.get('path', '?')}"
    elif name.startswith("skill_"):
        ctx = args.get("context", "")
        return f" \u2192 {ctx[:60]}" if ctx else ""
    elif name.startswith("mcp_"):
        # Show first meaningful argument value for MCP tools
        for key in ("query", "slug", "keyword", "name"):
            val = args.get(key)
            if val:
                return f" \u2192 {key}='{str(val)[:60]}'"
        if args:
            first_key = next(iter(args))
            return f" \u2192 {first_key}='{str(args[first_key])[:50]}'"
        return ""
    return ""


async def _stream_with_tools(conversation: list) -> str:
    from grokswarm.agents import _record_usage
    from grokswarm import llm
    full_response = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    xai_tools = llm.convert_tools(TOOL_SCHEMAS)
    for _round in range(MAX_TOOL_ROUNDS):
        full_response = ""
        collected_tool_calls = []
        finish_reason = None
        stream_response = None

        for _stream_attempt in range(1 + MAX_STREAM_RETRIES):
            try:
                shared._set_status("thinking...")
                # Split system prompt for xAI automatic prompt caching:
                # stable base (personality, rules, tools) stays identical across turns → cache hit
                _cache_messages = []
                for _m in conversation:
                    if _m["role"] == "system":
                        _sys_content = _m.get("content", "")
                        _base, _, _ctx = _sys_content.partition("## Project Context")
                        if _ctx:
                            _cache_messages.append({"role": "system", "content": _base.rstrip()})
                            _cache_messages.append({"role": "system", "content": "## Project Context" + _ctx})
                        else:
                            _cache_messages.append(_m)
                    else:
                        _cache_messages.append(_m)
                _chat = llm.create_chat(shared.MODEL, tools=xai_tools, max_tokens=shared.MAX_TOKENS)
                llm.populate_chat(_chat, _cache_messages)
                # xai-sdk streaming: tool calls arrive complete (no incremental accumulation)
                async for response, chunk in _chat.stream():
                    stream_response = response
                    if chunk.content:
                        full_response += chunk.content
                    for tc in chunk.tool_calls:
                        collected_tool_calls.append(tc)
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if _stream_attempt < MAX_STREAM_RETRIES:
                    shared.console.print(f"  [swarm.warning]\u26a0 Stream interrupted: {type(e).__name__}: {str(e)[:80]}[/swarm.warning]")
                    shared.console.print(f"  [swarm.dim]  retrying stream (attempt {_stream_attempt + 2}/{1 + MAX_STREAM_RETRIES})...[/swarm.dim]")
                    full_response = ""
                    collected_tool_calls = []
                    finish_reason = None
                    stream_response = None
                    await asyncio.sleep(2)
                else:
                    raise

        # Usage is available on the final response object after streaming completes
        if stream_response is not None:
            _usage = llm.extract_usage(stream_response)
            pt = _usage["prompt_tokens"]
            ct = _usage["completion_tokens"]
            cached = _usage["cached_tokens"]
            if pt or ct:
                total_prompt_tokens += pt
                total_completion_tokens += ct
                _record_usage(shared.MODEL, pt, ct, cached)
            finish_reason = stream_response.finish_reason

        if finish_reason == "length":
            shared.console.print("  [swarm.warning]\u26a0 Response was truncated (token limit). Output may be incomplete.[/swarm.warning]")

        if not collected_tool_calls:
            shared._clear_status()
            if full_response:
                try:
                    shared.console.print(Markdown(full_response))
                except Exception:
                    shared.console.print(full_response)
                conversation.append({"role": "assistant", "content": full_response})
            if total_prompt_tokens or total_completion_tokens:
                total = total_prompt_tokens + total_completion_tokens
                shared.console.print(f"  [dim]tokens: {total_prompt_tokens:,} in + {total_completion_tokens:,} out = {total:,} total[/dim]")
            return full_response

        if full_response:
            has_spawn = any(tc.function.name == "spawn_agent" for tc in collected_tool_calls)
            if not has_spawn:
                try:
                    shared.console.print(Markdown(full_response))
                except Exception:
                    shared.console.print(full_response)

        tool_calls_list = [llm.tool_call_to_dict(tc) for tc in collected_tool_calls]
        conversation.append({
            "role": "assistant",
            "content": full_response or None,
            "tool_calls": tool_calls_list,
        })

        tool_count = len(collected_tool_calls)
        if shared.state.verbose_mode:
            shared.console.print(f"  [swarm.dim]round {_round + 1}/{MAX_TOOL_ROUNDS} \u2014 {tool_count} tool{'s' if tool_count != 1 else ''}[/swarm.dim]")
        shared._log(f"round {_round + 1}/{MAX_TOOL_ROUNDS} - {tool_count} tools")

        parsed_tools: list[tuple[dict, str, dict]] = []
        for tc_proto in collected_tool_calls:
            tc_id = tc_proto.id
            name = tc_proto.function.name
            raw_args = tc_proto.function.arguments
            # Flat dict matching the shape expected by tool execution code
            tc = {"id": tc_id, "name": name, "arguments": raw_args}
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                try:
                    args = json.loads(_repair_json(raw_args))
                except json.JSONDecodeError as e:
                    if shared.state.verbose_mode:
                        shared.console.print(f"  [swarm.accent]\u26a1 {name}[/swarm.accent][dim] (invalid arguments)[/dim]")
                    shared._log(f"\u26a1 {name} (invalid arguments)")
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": f"Error: Failed to parse tool arguments as JSON: {e}. Raw arguments: {raw_args[:200]}. Please retry with valid JSON arguments.",
                    })
                    continue
            parsed_tools.append((tc, name, args))

        # Determine threadable set for this round
        if shared.state.trust_mode or shared.state.agent_mode > 0:
            _round_threadable = _THREADABLE_TOOLS_TRUSTED
        else:
            _round_threadable = _THREADABLE_TOOLS_BASE

        def _is_threadable(n: str) -> bool:
            return n in _round_threadable or n.startswith("mcp_")

        try:
            all_read_only = all(n in READ_ONLY_TOOLS for _, n, _ in parsed_tools)
            can_parallelize = all_read_only and len(parsed_tools) > 1

            if can_parallelize:
                if not shared.state.verbose_mode:
                    _par_names = ", ".join(n for _, n, _ in parsed_tools)
                    shared._set_status(f"working on: {_par_names}")
                else:
                    shared.console.print(f"  [swarm.dim]\u21c4 executing {len(parsed_tools)} read-only tools in parallel[/swarm.dim]")
                async def _run_one(tc, name, args):
                    detail = _tool_detail(name, args)
                    if shared.state.verbose_mode:
                        shared.console.print(f"  [swarm.accent]\u26a1 {name}[/swarm.accent][dim]{detail}[/dim]")
                    shared._log(f"\u26a1 {name}{detail}")
                    try:
                        result_str, elapsed = await _execute_tool(name, args, True)
                    except Exception as e:
                        result_str, elapsed = f"Error: {e}", 0.0
                    if shared.state.verbose_mode:
                        shared.console.print(f"    [dim]\u2514 {tc['name']} done ({elapsed:.1f}s)[/dim]")
                    shared._log(f"\u2514 {tc['name']} done ({elapsed:.1f}s)")
                    return tc, result_str
                results = await asyncio.gather(*[_run_one(tc, name, args) for tc, name, args in parsed_tools])
                for tc, result_str in results:
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })
            else:
                # Check if ANY tool in this round needs approval (non-threadable).
                # If so, suspend prompt_toolkit ONCE for the entire batch to
                # prevent prompt flashing between sequential approvals.
                _needs_suspension = any(
                    not _is_threadable(n) for _, n, _ in parsed_tools
                )
                if _needs_suspension:
                    shared._clear_status()
                    await _suspend_prompt()

                if not shared.state.verbose_mode:
                    _seq_names = ", ".join(n for _, n, _ in parsed_tools)
                    shared._set_status(f"working on: {_seq_names}")
                try:
                    for tc, name, args in parsed_tools:
                        detail = _tool_detail(name, args)
                        if shared.state.verbose_mode:
                            shared.console.print(f"  [swarm.accent]\u26a1 {name}[/swarm.accent][dim]{detail}[/dim]")
                        shared._log(f"\u26a1 {name}{detail}")
                        t0 = time.perf_counter()
                        result_str = await _execute_tool(name, args)
                        elapsed = time.perf_counter() - t0
                        if shared.state.verbose_mode and elapsed >= 0.5:
                            shared.console.print(f"    [dim]\u2514 done ({elapsed:.1f}s)[/dim]")
                        shared._log(f"\u2514 {name} done ({elapsed:.1f}s)")
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result_str,
                        })
                finally:
                    if _needs_suspension:
                        _resume_prompt()

        except KeyboardInterrupt:
            # Make sure prompt is resumed on interrupt
            if shared._is_prompt_suspended:
                _resume_prompt()
            shared.console.print("\n  [swarm.warning]\u26a0 Tool execution interrupted by user.[/swarm.warning]")
            responded_ids = {m["tool_call_id"] for m in conversation if m.get("role") == "tool"}
            for tc, name, args in parsed_tools:
                if tc["id"] not in responded_ids:
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "Cancelled: tool execution interrupted by user.",
                    })
            return full_response

    return full_response
