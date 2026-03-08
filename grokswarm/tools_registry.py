"""TOOL_SCHEMAS, TOOL_DISPATCH, skill registration, registry helpers."""

import yaml

import grokswarm.shared as shared
from grokswarm.tools_image import analyze_image, generate_image, edit_image
from grokswarm.registry_helpers import (
    seed_defaults, list_experts, list_skills, save_memory,
    propose_expert, propose_skill, get_registry,
)
from grokswarm.tools_fs import list_dir, read_file, write_file, edit_file, search_files, grep_files
from grokswarm.tools_shell import run_shell
from grokswarm.tools_test import run_tests, run_app_capture, capture_tui_screenshot
from grokswarm.tools_git import (
    git_status, git_diff, git_log, git_commit, git_checkout,
    git_branch, git_show_file, git_blame, git_stash, git_init,
)
from grokswarm.tools_browser import fetch_page, screenshot_page, extract_links
from grokswarm.tools_search import web_search, x_search, code_execution
from grokswarm.context import find_symbol, find_references


# -- Tool Schemas --
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in a project directory. Returns formatted listing with sizes. Use @grokswarm/ prefix to list GrokSwarm's own source files (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to list. Defaults to project root."}
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the project. Supports reading specific line ranges for large files. Use @grokswarm/ prefix to read GrokSwarm's own source files (read-only self-knowledge).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "start_line": {"type": "integer", "description": "Optional: 1-based start line number. Omit to read entire file."},
                    "end_line": {"type": "integer", "description": "Optional: 1-based end line number. Omit to read to end."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the project. User will be prompted to approve. Use ONLY for creating new files. For modifying existing files, use edit_file instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path for the file."},
                    "content": {"type": "string", "description": "Full content to write."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Surgically edit a file by finding and replacing specific text. Supports SINGLE edit (old_text/new_text) or MULTI-EDIT (edits array) for multiple changes in one call. Each old_text must match exactly ONE occurrence. ALWAYS prefer this over write_file for modifications. Use multi-edit when you need to change multiple places in the same file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file to edit."},
                    "old_text": {"type": "string", "description": "The exact text to find and replace (for single edit). Must match exactly once. Include 2-3 lines of surrounding context for uniqueness."},
                    "new_text": {"type": "string", "description": "The replacement text (for single edit)."},
                    "edits": {
                        "type": "array",
                        "description": "For multi-edit: array of {old_text, new_text} objects. All edits are validated before any are applied. Use this when changing multiple places in the same file.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {"type": "string", "description": "Text to find and replace."},
                                "new_text": {"type": "string", "description": "Replacement text."}
                            },
                            "required": ["old_text", "new_text"]
                        }
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files by name in the project directory tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search string to match against file names (case-insensitive)."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": "Search inside file contents for a text pattern (like grep). Returns matching lines with file:line references. Supports regex and context lines (like grep -C). Use this to find code, config values, or any text across the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text pattern to search for (case-insensitive). Supports regex when is_regex=true."},
                    "path": {"type": "string", "description": "Optional: relative path to limit search scope (file or directory). Defaults to project root."},
                    "is_regex": {"type": "boolean", "description": "If true, treat pattern as a Python regex. Default false."},
                    "context_lines": {"type": "integer", "description": "Number of context lines to show around each match (like grep -C). Default 0, max 10."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command in the project directory. User will be prompted to approve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_registry",
            "description": "List all registered experts and skills with their mindsets/descriptions.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_expert",
            "description": "Propose a new expert agent. User will be prompted to approve before saving. Use when a task needs a specialist that doesn't exist yet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Display name for the expert (e.g. 'DevOps_Engineer')."},
                    "mindset": {"type": "string", "description": "The expert's permanent personality and approach (1-2 sentences)."},
                    "objectives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of core objectives the expert should pursue."
                    }
                },
                "required": ["name", "mindset", "objectives"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Propose a new reusable skill. User will be prompted to approve before saving. Skills capture repeatable workflows or knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Display name for the skill (e.g. 'code_review')."},
                    "description": {"type": "string", "description": "What this skill does and when to use it."},
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional ordered steps for executing this skill."
                    }
                },
                "required": ["name", "description"]
            }
        }
    },
    {"type": "function", "function": {"name": "git_status", "description": "Show the current git status (branch, staged/unstaged/untracked files).", "parameters": {"type": "object", "properties": {}}}},
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show git diff of changes. Can diff a specific file or all changes. Use staged=true for staged changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional: specific file to diff."},
                    "staged": {"type": "boolean", "description": "If true, show staged changes instead of unstaged. Default false."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent git commit history (oneline format with decorations).",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of commits to show (default 10, max 50)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage all changes and create a git commit. User will be prompted to approve. Use this to create checkpoints before risky changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message."}
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_checkout",
            "description": "Restore a file to its last committed state, or switch branches. User will be prompted to approve. Use to undo changes to specific files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "File path to restore, or branch/commit to switch to."}
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_branch",
            "description": "List branches, create a new branch, or delete a branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Branch name to create. Omit to list all branches."},
                    "delete": {"type": "boolean", "description": "If true, delete the named branch instead of creating it."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_show_file",
            "description": "Show the contents of a file at a specific git ref (commit SHA, branch name, or tag). Useful for comparing current version to an older one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "ref": {"type": "string", "description": "Git ref: commit SHA, branch name, or tag. Default: HEAD."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_blame",
            "description": "Show git blame for a file -- who changed each line and when. Useful for understanding code history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_stash",
            "description": "Manage git stash. Actions: 'list' (default, show stashes), 'push' (stash current changes), 'pop' (apply+drop top stash), 'drop' (discard top stash).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Stash action: list, push, pop, drop. Default: list.", "enum": ["list", "push", "pop", "drop"]},
                    "message": {"type": "string", "description": "Optional message for 'push' action."}
                }
            }
        }
    },
    {"type": "function", "function": {"name": "git_init", "description": "Initialize a new git repository in the project directory. Only needed if no .git exists.", "parameters": {"type": "object", "properties": {}}}},
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run project tests. Auto-detects the test framework (pytest, jest, go test, etc) if no command is given. Use after code changes to verify correctness.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Optional: custom test command. If omitted, auto-detects framework."},
                    "pattern": {"type": "string", "description": "Optional: filter pattern to run specific tests (e.g. test name or file)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_app_capture",
            "description": "Launch an app or command, let it run for up to `timeout` seconds, and capture all stdout/stderr output. Use this to see what a CLI, TUI, or server actually outputs when executed. For interactive apps, provide stdin_text to simulate user input (e.g. keystrokes, typed answers). Essential for verifying end-to-end behavior beyond unit tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run (e.g. 'python app.py', 'node server.js')."},
                    "timeout": {"type": "integer", "description": "Max seconds to let the process run before killing it. Default: 10."},
                    "stdin_text": {"type": "string", "description": "Optional: text to feed to the process's stdin (simulate user input)."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "capture_tui_screenshot",
            "description": "Capture a screenshot of a Textual TUI app by running it headlessly. Saves an SVG image that can be analyzed with analyze_image to verify visual layout, content rendering, and UI correctness. Use this after fixing visual/UI bugs in Textual apps to verify the fix actually works. Supports simulating key presses before capturing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Path to the Python file containing the Textual App (e.g. 'app.py'), or a module name."},
                    "save_path": {"type": "string", "description": "Where to save the screenshot. Default: tui_screenshot.svg"},
                    "timeout": {"type": "integer", "description": "Max seconds for the headless run. Default: 15."},
                    "press": {"type": "string", "description": "Optional: comma-separated key presses to simulate before capture (e.g. 'tab,tab,enter' to switch tabs)."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch a web page and return its text content. Use for reading documentation, articles, web content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "screenshot_page",
            "description": "Take a screenshot of a web page and save it to a file in the project. User will be prompted to approve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to screenshot."},
                    "save_path": {"type": "string", "description": "Relative path to save the screenshot (default: screenshot.png)."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_links",
            "description": "Extract all links from a web page. Useful for finding resources, documentation links, or navigating site structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to extract links from."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": "Find where a symbol (class, function, variable) is defined in the project. Uses AST for Python (full detail: methods, args), regex for other languages. Like go-to-definition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The symbol name to look up (e.g. 'MyClass', 'parse_config', 'MAX_RETRIES')."}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_references",
            "description": "Find all files that import or reference a given module or symbol. Uses AST import analysis for Python, word-boundary search for other languages. Like find-all-references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The module or symbol name to search for references to."}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web in real-time using xAI server-side search. Returns summarized results with source URLs. Use for current events, documentation, research, facts. Faster and more reliable than fetch_page for finding information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "domains": {"type": "array", "items": {"type": "string"}, "description": "Optional: limit results to these domains (e.g. ['python.org', 'docs.python.org'])."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "x_search",
            "description": "Search X (Twitter) posts in real-time using xAI server-side search. Returns summarized posts with links. Use for opinions, trending topics, social media sentiment, real-time reactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_execution",
            "description": "Execute code server-side via xAI's sandboxed code execution environment. Returns stdout/stderr output. Use for running calculations, data processing, or testing code snippets without local execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to execute."},
                    "language": {"type": "string", "description": "Programming language. Default: python.", "enum": ["python", "javascript"]}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze an image file using vision AI. Can describe contents, read text/diagrams, answer questions about the image. Supports png, jpg, gif, webp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the image file."},
                    "question": {"type": "string", "description": "Optional question about the image. Default: describe the image."},
                    "detail": {"type": "string", "enum": ["auto", "low", "high"], "description": "Image detail level. 'low' uses fewer tokens, 'high' for fine detail. Default: auto."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate images from a text prompt using xAI's image generation model (grok-2-image). Returns URLs of generated images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate."},
                    "n": {"type": "integer", "description": "Number of images to generate (1-4). Default: 1."},
                    "size": {"type": "string", "description": "Image size. Default: 1024x1024.", "enum": ["1024x1024", "1024x768", "768x1024"]}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": "Edit an existing image using a text prompt via xAI's image editing model. Returns URL of the edited image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Relative path to the source image file."},
                    "prompt": {"type": "string", "description": "Text description of the edits to make."}
                },
                "required": ["image_path", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_agent",
            "description": "Spawn a sub-agent to work on a subtask asynchronously. The agent runs in the background using the specified expert profile. Returns immediately with the agent ID. Use check_messages to get results when the agent finishes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expert": {"type": "string", "description": "Name of the expert profile to use (e.g. 'coder', 'researcher', 'assistant')."},
                    "task": {"type": "string", "description": "Detailed description of the subtask for this agent."},
                    "name": {"type": "string", "description": "Optional: unique name for this agent instance. Auto-generated if not provided."},
                    "token_budget": {"type": "integer", "description": "Optional: max tokens this agent can use. 0 = unlimited."},
                    "cost_budget": {"type": "number", "description": "Optional: max cost in USD. 0 = unlimited."}
                },
                "required": ["expert", "task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a direct message to another agent via the SwarmBus. Use for coordination, sharing results, or requesting information from a specific agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient agent name (or '*' for broadcast)."},
                    "body": {"type": "string", "description": "Message content."},
                    "kind": {"type": "string", "description": "Message type: 'request', 'result', 'status', 'error'. Default: 'request'."}
                },
                "required": ["to", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_messages",
            "description": "Check the SwarmBus for messages addressed to you (or all agents). Returns recent messages. Use to see results from spawned agents or messages from peers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Your agent name to check messages for. Use '*' for all messages."},
                    "since_id": {"type": "integer", "description": "Only return messages with id > this value. Default: 0 (all messages)."}
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_agents",
            "description": "List all active agents in the swarm with their current state, task, and token usage.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
]


# -- Tool Dispatch --
TOOL_DISPATCH = {
    "list_directory": lambda args: list_dir(args.get("path", ".")),
    "read_file": lambda args: read_file(args["path"], args.get("start_line"), args.get("end_line")),
    "write_file": lambda args: write_file(args["path"], args["content"]),
    "edit_file": lambda args: edit_file(args["path"], args.get("old_text", ""), args.get("new_text", ""), args.get("edits")),
    "search_files": lambda args: search_files(args["query"]),
    "grep_files": lambda args: grep_files(args["pattern"], args.get("path", "."), args.get("is_regex", False), args.get("context_lines", 0)),
    "run_shell": lambda args: run_shell(args["command"]),
    "run_tests": lambda args: run_tests(args.get("command"), args.get("pattern")),
    "run_app_capture": lambda args: run_app_capture(args["command"], args.get("timeout", 10), args.get("stdin_text")),
    "capture_tui_screenshot": lambda args: capture_tui_screenshot(args["command"], args.get("save_path", "tui_screenshot.svg"), args.get("timeout", 15), args.get("press")),
    "list_registry": lambda args: get_registry(),
    "create_expert": lambda args: propose_expert(args["name"], args["mindset"], args["objectives"]),
    "create_skill": lambda args: propose_skill(args["name"], args["description"], args.get("steps")),
    "git_status": lambda args: git_status(),
    "git_diff": lambda args: git_diff(args.get("path"), args.get("staged", False)),
    "git_log": lambda args: git_log(args.get("count", 10)),
    "git_commit": lambda args: git_commit(args["message"]),
    "git_checkout": lambda args: git_checkout(args["target"]),
    "git_branch": lambda args: git_branch(args.get("name"), args.get("delete", False)),
    "git_show_file": lambda args: git_show_file(args["path"], args.get("ref", "HEAD")),
    "git_blame": lambda args: git_blame(args["path"]),
    "git_stash": lambda args: git_stash(args.get("action", "list"), args.get("message")),
    "git_init": lambda args: git_init(),
    "fetch_page": lambda args: fetch_page(args["url"]),
    "screenshot_page": lambda args: screenshot_page(args["url"], args.get("save_path", "screenshot.png")),
    "extract_links": lambda args: extract_links(args["url"]),
    "find_symbol": lambda args: find_symbol(args["name"]),
    "find_references": lambda args: find_references(args["name"]),
    "web_search": lambda args: web_search(args["query"], args.get("domains")),
    "x_search": lambda args: x_search(args["query"]),
    "code_execution": lambda args: code_execution(args["code"], args.get("language", "python")),
    "analyze_image": lambda args: analyze_image(args["path"], args.get("question", "Describe this image in detail."), args.get("detail", "auto")),
    "generate_image": lambda args: generate_image(args["prompt"], args.get("n", 1), args.get("size", "1024x1024")),
    "edit_image": lambda args: edit_image(args["image_path"], args["prompt"]),
}


# -- Read-Only Tools --
READ_ONLY_TOOLS = {
    "list_directory", "read_file", "search_files", "grep_files",
    "list_registry", "git_status", "git_diff", "git_log",
    "git_branch", "git_show_file", "git_blame",
    "fetch_page", "extract_links",
    "find_symbol", "find_references",
    "web_search", "x_search", "code_execution",
    "analyze_image",
    "check_messages", "list_agents",
}

_FILE_MUTATION_TOOLS = {"edit_file", "write_file"}

_READONLY_BLOCKED_TOOLS = {"write_file", "edit_file", "run_shell", "git_commit",
                           "git_checkout", "git_branch", "git_stash", "git_init",
                           "create_expert", "create_skill", "screenshot_page"}


# -- Agent Plan/Progress Tracking --
def _update_plan_impl(agent_name: str, steps: list[dict]) -> str:
    agent = shared.state.get_agent(agent_name)
    if not agent:
        return f"Agent '{agent_name}' not found."
    valid_statuses = {"pending", "in-progress", "done", "skipped"}
    plan = []
    for s in steps:
        step_text = s.get("step", "").strip()
        step_status = s.get("status", "pending").strip()
        if not step_text:
            continue
        if step_status not in valid_statuses:
            step_status = "pending"
        plan.append({"step": step_text, "status": step_status})
    agent.plan = plan
    done_count = sum(1 for s in plan if s["status"] == "done")
    return f"Plan updated: {done_count}/{len(plan)} steps complete."


TOOL_DISPATCH["update_plan"] = lambda args: _update_plan_impl(
    args.get("_agent_name", "unknown"), args.get("steps", [])
)

_UPDATE_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "update_plan",
        "description": "Create or update your work plan. Call this FIRST to outline your steps, then call it again as you complete each step (set status to 'done'). The user can see your plan in real-time via /watch and /peek.",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "Your full plan as an ordered list of steps. Include ALL steps every time (not just changed ones). Set status to reflect current progress.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string", "description": "Short description of this step (e.g. 'Read app.py to understand structure')."},
                            "status": {"type": "string", "enum": ["pending", "in-progress", "done", "skipped"], "description": "Current status of this step."}
                        },
                        "required": ["step", "status"]
                    }
                }
            },
            "required": ["steps"]
        }
    }
}

_AGENT_EXCLUDED_TOOLS = {"spawn_agent"}


def get_agent_tool_schemas(allowed_tools: set[str] | None = None) -> list[dict]:
    """Build agent tool schemas dynamically so MCP tools registered after import are included.
    If allowed_tools is provided, only include tools in that set."""
    base = [t for t in TOOL_SCHEMAS if t.get("function", {}).get("name") not in _AGENT_EXCLUDED_TOOLS]
    base.append(_UPDATE_PLAN_SCHEMA)
    if allowed_tools:
        # Always allow update_plan even if not explicitly listed
        allowed_with_plan = allowed_tools | {"update_plan"}
        base = [t for t in base if t.get("function", {}).get("name") in allowed_with_plan]
    return base


# Static snapshot for backward compat — prefer get_agent_tool_schemas() for fresh list
AGENT_TOOL_SCHEMAS = get_agent_tool_schemas()


# -- Skill Tools --
def _invoke_skill(skill_name: str, context: str = "") -> str:
    path = shared.SKILLS_DIR / f"{skill_name}.yaml"
    if not path.exists():
        return f"Skill '{skill_name}' not found."
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = f"=== Skill: {data.get('name', skill_name)} ===\n"
    result += f"Description: {data.get('description', '')}\n"
    steps = data.get("steps", [])
    if steps:
        result += "\nSteps:\n"
        for i, step in enumerate(steps, 1):
            result += f"  {i}. {step}\n"
    if context:
        result += f"\nApply to: {context}\n"
    result += "\nFollow these steps to complete the task."
    return result


def _register_skill_tool(safe_name: str, description: str):
    tool_name = f"skill_{safe_name}"
    if any(t["function"]["name"] == tool_name for t in TOOL_SCHEMAS if "function" in t):
        return
    TOOL_SCHEMAS.append({
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"Invoke the '{safe_name}' skill: {description}",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "What to apply this skill to (file path, topic, etc)."}
                },
            }
        }
    })
    TOOL_DISPATCH[tool_name] = lambda args, _sn=safe_name: _invoke_skill(_sn, args.get("context", ""))
    READ_ONLY_TOOLS.add(tool_name)


def _load_skill_tools():
    for f in sorted(shared.SKILLS_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            _register_skill_tool(f.stem, data.get("description", "(no description)"))
        except Exception:
            pass


# Initialize on import
seed_defaults()
_load_skill_tools()

from grokswarm.tools_mcp import register_mcp_tools
register_mcp_tools()
