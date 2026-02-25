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
        summary_response = await shared._api_call_with_retry(
            lambda: shared.client.chat.completions.create(
                model=shared.MODEL,
                messages=[
                    {"role": "system", "content": "Summarize this conversation history into a concise paragraph. Capture key decisions, files modified, tasks completed, and important context. Be factual and brief."},
                    {"role": "user", "content": old_text}
                ],
                max_tokens=500,
            ),
            label="Compaction"
        )
        if hasattr(summary_response, 'usage') and summary_response.usage:
            from grokswarm.agents import _record_usage
            _record_usage(shared.MODEL, summary_response.usage.prompt_tokens, summary_response.usage.completion_tokens)
        summary = summary_response.choices[0].message.content.strip()
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


async def _suspend_prompt_and_run(func):
    """Suspend prompt_toolkit, run *func* in a thread, then resume.

    Used for tools that call Confirm.ask() / console.input() for approval.
    If no REPL is running, simply run in a thread without the suspend dance.
    """
    app = shared._toolbar_app_ref
    if not app or not getattr(app, "is_running", False):
        # No active REPL (e.g. tests, CLI commands) -- just run in thread
        return await asyncio.to_thread(func)

    async with shared._suspend_lock:
        shared._toolbar_suspended = True
        shared._is_prompt_suspended = True

        try:
            shared._saved_prompt_text = app.current_buffer.text
            app.exit(result="__MAGIC_SUSPEND__")
        except Exception:
            pass

        await shared._prompt_suspend_event.wait()
        shared._prompt_suspend_event.clear()

        try:
            return await asyncio.to_thread(func)
        finally:
            shared._is_prompt_suspended = False
            shared._toolbar_suspended = False
            shared._prompt_resume_event.set()


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
            else:
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
    return ""


async def _stream_with_tools(conversation: list) -> str:
    from grokswarm.agents import _record_usage
    full_response = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for _round in range(MAX_TOOL_ROUNDS):
        full_response = ""
        tool_calls_data = {}
        finish_reason = None
        round_usage = None

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
                stream = await shared._api_call_with_retry(
                    lambda: shared.client.chat.completions.create(
                        model=shared.MODEL,
                        messages=_cache_messages,
                        tools=TOOL_SCHEMAS,
                        stream=True,
                        stream_options={"include_usage": True},
                        max_tokens=shared.MAX_TOKENS,
                    ),
                    label="Chat"
                )
                async for chunk in stream:
                    if hasattr(chunk, 'usage') and chunk.usage:
                        round_usage = chunk.usage
                        continue
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    if delta.content:
                        full_response += delta.content
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_data:
                                tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.id:
                                tool_calls_data[idx]["id"] = tc.id
                            if tc.function and tc.function.name:
                                tool_calls_data[idx]["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls_data[idx]["arguments"] += tc.function.arguments
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if _stream_attempt < MAX_STREAM_RETRIES:
                    shared.console.print(f"  [swarm.warning]\u26a0 Stream interrupted: {type(e).__name__}: {str(e)[:80]}[/swarm.warning]")
                    shared.console.print(f"  [swarm.dim]  retrying stream (attempt {_stream_attempt + 2}/{1 + MAX_STREAM_RETRIES})...[/swarm.dim]")
                    full_response = ""
                    tool_calls_data = {}
                    finish_reason = None
                    round_usage = None
                    await asyncio.sleep(2)
                else:
                    raise

        if round_usage:
            pt = getattr(round_usage, 'prompt_tokens', 0) or 0
            ct = getattr(round_usage, 'completion_tokens', 0) or 0
            total_prompt_tokens += pt
            total_completion_tokens += ct
            _record_usage(shared.MODEL, pt, ct)

        if finish_reason == "length":
            shared.console.print("  [swarm.warning]\u26a0 Response was truncated (token limit). Output may be incomplete.[/swarm.warning]")

        if not tool_calls_data:
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
            has_spawn = any(tc["name"] == "spawn_agent" for tc in tool_calls_data.values())
            if not has_spawn:
                try:
                    shared.console.print(Markdown(full_response))
                except Exception:
                    shared.console.print(full_response)

        tool_calls_list = []
        for idx in sorted(tool_calls_data.keys()):
            tc = tool_calls_data[idx]
            tool_calls_list.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            })
        conversation.append({
            "role": "assistant",
            "content": full_response or None,
            "tool_calls": tool_calls_list,
        })

        tool_count = len(tool_calls_data)
        if shared.state.verbose_mode:
            shared.console.print(f"  [swarm.dim]round {_round + 1}/{MAX_TOOL_ROUNDS} \u2014 {tool_count} tool{'s' if tool_count != 1 else ''}[/swarm.dim]")
        shared._log(f"round {_round + 1}/{MAX_TOOL_ROUNDS} - {tool_count} tools")

        parsed_tools: list[tuple[dict, str, dict]] = []
        for idx in sorted(tool_calls_data.keys()):
            tc = tool_calls_data[idx]
            name = tc["name"]
            try:
                args = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                try:
                    args = json.loads(_repair_json(tc["arguments"]))
                except json.JSONDecodeError as e:
                    if shared.state.verbose_mode:
                        shared.console.print(f"  [swarm.accent]\u26a1 {name}[/swarm.accent][dim] (invalid arguments)[/dim]")
                    shared._log(f"\u26a1 {name} (invalid arguments)")
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"Error: Failed to parse tool arguments as JSON: {e}. Raw arguments: {tc['arguments'][:200]}. Please retry with valid JSON arguments.",
                    })
                    continue
            parsed_tools.append((tc, name, args))

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
                if not shared.state.verbose_mode:
                    _seq_names = ", ".join(n for _, n, _ in parsed_tools)
                    shared._set_status(f"working on: {_seq_names}")
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

        except KeyboardInterrupt:
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
