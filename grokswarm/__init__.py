"""GrokSwarm package -- re-exports everything for backward compatibility."""

import yaml  # noqa: F401 -- tests use _main.yaml

# -- models --
from grokswarm.models import AgentState, AgentInfo, SwarmState

# -- shared --
from grokswarm.shared import (
    console, app, state,
    SWARM_THEME, THINKING_FRAMES,
    XAI_API_KEY, VERSION, MODEL, BASE_URL, MAX_TOKENS, CODE_MODEL,
    MODEL_PRICING, _get_pricing,
    PROJECT_DIR, PROJECT_CONTEXT, SYSTEM_PROMPT,
    GROKSWARM_HOME, SKILLS_DIR, EXPERTS_DIR, TEAMS_DIR, MEMORY_DIR,
    SESSIONS_DIR, CONTEXT_CACHE_DIR, _RECENT_PROJECTS_FILE,
    _background_tasks, _bus_instance, _agent_counter,
    _input_queue, _drain_input_queue,
    _set_status, _clear_status,
    _toolbar_status, _toolbar_spinner_idx, _toolbar_app_ref, _toolbar_suspended,
    _prompt_suspend_event, _prompt_resume_event, _saved_prompt_text, _is_prompt_suspended, _suspend_lock,
    _open_session_log, _log,
    _redact_secrets, _sanitize_surrogates,
    SafeFileHistory,
    _terminal_confirm, _auto_approve,
    AUTO_CHECKPOINT_THRESHOLD, MAX_EDIT_HISTORY,
    MAX_API_RETRIES, RETRY_BACKOFF,
    _api_call_with_retry,
    _SECRET_PATTERNS,
)

# -- context --
from grokswarm.context import (
    CONTEXT_FILES, MAX_CONTEXT_FILE_SIZE, MAX_TREE_DEPTH, MAX_TREE_FILES,
    MAX_SCAN_FILES, MAX_INDEX_FILE_SIZE,
    IGNORE_DIRS, _IGNORE_PATTERNS, _IGNORE_LITERALS,
    CODE_EXTENSIONS, BASE_SYSTEM_PROMPT,
    _should_ignore, _iter_project_files, _scan_tree,
    _read_context_file,
    _build_python_symbol_index, _extract_python_imports,
    _find_symbol_in_file, find_symbol, find_references,
    _file_size_ok, _build_import_graph, _build_deep_symbol_index,
    _detect_language_stats,
    scan_project_context, scan_project_context_cached,
    _context_cache_key, _project_mtime,
    _load_cached_context, _save_context_cache,
    format_context_for_prompt, build_system_prompt,
    _safe_path, _grokswarm_read_path,
    _incremental_context_refresh,
)

# -- tools_fs --
from grokswarm.tools_fs import list_dir, read_file, write_file, edit_file, search_files, grep_files

# -- tools_shell --
from grokswarm.tools_shell import DANGEROUS_PATTERNS, _is_dangerous_command, _approval_prompt, _explain_command_safety, run_shell

# -- tools_test --
from grokswarm.tools_test import (
    TEST_COMMANDS, _detect_test_framework, _run_tests_raw, run_tests,
    run_app_capture, capture_tui_screenshot, LINT_COMMANDS, _lint_file,
    MAX_TEST_FIX_ATTEMPTS,
)

# -- tools_git --
from grokswarm.tools_git import (
    git_status, git_diff, git_log, git_commit, git_checkout,
    git_branch, git_show_file, git_blame, git_stash, git_init,
)

# -- tools_browser --
from grokswarm.tools_browser import fetch_page, screenshot_page, extract_links, _atexit_close_browser

# -- tools_search --
from grokswarm.tools_search import web_search, x_search, code_execution, _check_ssrf

# -- tools_image --
from grokswarm.tools_image import analyze_image, generate_image, edit_image

# -- registry_helpers --
from grokswarm.registry_helpers import (
    seed_defaults, list_experts, list_skills, save_memory,
    propose_expert, propose_skill, get_registry,
)

# -- tools_registry --
from grokswarm.tools_registry import (
    TOOL_SCHEMAS, TOOL_DISPATCH, READ_ONLY_TOOLS,
    _FILE_MUTATION_TOOLS, _READONLY_BLOCKED_TOOLS,
    AGENT_TOOL_SCHEMAS, get_agent_tool_schemas,
    _invoke_skill, _register_skill_tool, _load_skill_tools,
)

# -- bugs --
from grokswarm.bugs import (
    BugTracker, Bug, get_self_tracker, get_project_tracker,
    log_self_bug, log_project_bug, log_exception,
    report_bug_impl, list_bugs_impl, update_bug_impl,
)

# -- tools_mcp --
from grokswarm.tools_mcp import register_mcp_tools, _load_mcp_config, _discover_mcp_tools, _call_mcp_tool

# -- engine --
from grokswarm.engine import (
    _estimate_tokens, _repair_json,
    _compact_conversation, _trim_conversation,
    _suspend_prompt, _resume_prompt, _suspend_prompt_and_run,
    _execute_tool, _tool_detail,
    _handle_test_failure, _maybe_auto_retest,
    _stream_with_tools,
    _THREADABLE_TOOLS_BASE, _THREADABLE_TOOLS_TRUSTED,
    MAX_TOOL_ROUNDS, MAX_TOOL_RESULT_SIZE, MAX_STREAM_RETRIES, STREAM_TIMEOUT_SEC,
    MAX_CONVERSATION_MESSAGES, COMPACTION_THRESHOLD,
    COMPACTION_KEEP_RECENT, COMPACTION_TOKEN_LIMIT,
)

# -- agents --
from grokswarm.agents import (
    SwarmBus, get_bus,
    _costs_file, _load_project_costs, _save_project_costs, _record_usage, _extract_cached_tokens,
    EXPERT_DEFAULT_MAX_ROUNDS, _auto_checkpoint_before_agent,
    _detect_tech_stack, _build_completion_report,
    run_supervisor, run_expert,
    _spawn_agent_impl, _send_message_impl, _check_messages_impl, _list_agents_impl,
)

# -- repl --
from grokswarm.repl import (
    SwarmCompleter,
    _load_recent_projects, _update_recent_projects,
    _switch_project, _switch_project_async,
    save_session, load_session, list_sessions, delete_session,
    _show_help, _show_context, _handle_session_command, _run_doctor,
    show_welcome, main, chat, _chat_async, _swarm_async,
)

# -- dashboard --
from grokswarm.dashboard import (
    _build_dashboard, _build_swarm_monitor, _build_swarm_feed,
    _build_swarm_view, _watch_agents, _check_quit_key, dashboard,
)

# -- commands (registers @app.command() on import) --
import grokswarm.commands  # noqa: F401
