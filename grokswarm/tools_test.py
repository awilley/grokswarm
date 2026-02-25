"""Testing tools: run_tests, run_app_capture, capture_tui_screenshot, _lint_file."""

import sys
import json
import subprocess
from pathlib import Path

import grokswarm.shared as shared
from grokswarm.context import _safe_path, _iter_project_files

# -- Test Framework Detection --
TEST_COMMANDS = {
    "pytest": {"detect": ["pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py"], "cmd": "python -m pytest -v"},
    "unittest": {"detect": [], "cmd": "python -m unittest discover -v"},
    "jest": {"detect": ["jest.config.js", "jest.config.ts"], "cmd": "npx jest --verbose"},
    "mocha": {"detect": [".mocharc.yml", ".mocharc.json"], "cmd": "npx mocha"},
    "go": {"detect": ["go.mod"], "cmd": "go test ./... -v"},
    "cargo": {"detect": ["Cargo.toml"], "cmd": "cargo test"},
}

MAX_TEST_FIX_ATTEMPTS = 3


def _detect_test_framework() -> str | None:
    for name, info in TEST_COMMANDS.items():
        for detect_file in info["detect"]:
            if (shared.PROJECT_DIR / detect_file).exists():
                return name
    test_files = [p for p, _ in _iter_project_files(shared.PROJECT_DIR, extensions={".py"}, max_files=200)
                  if p.name.startswith("test_") or p.name.endswith("_test.py")]
    if test_files:
        return "pytest"
    if (shared.PROJECT_DIR / "package.json").exists():
        try:
            pkg = json.loads((shared.PROJECT_DIR / "package.json").read_text())
            if "jest" in pkg.get("devDependencies", {}) or "jest" in pkg.get("dependencies", {}):
                return "jest"
        except Exception:
            pass
    return None


def _run_tests_raw(command: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=shared.PROJECT_DIR, timeout=timeout
        )
        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"\nSTDOUT:\n{result.stdout}"
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if len(output) > 6000:
            output = output[:6000] + "\n... (truncated)"
        if result.returncode == 0:
            return "[PASS] TESTS PASSED\n" + output
        else:
            return "[FAIL] TESTS FAILED\n" + output
    except subprocess.TimeoutExpired:
        return "[FAIL] TESTS FAILED\nError: test command timed out (120s limit)."
    except Exception as e:
        return f"Error: {e}"


def run_tests(command: str | None = None, pattern: str | None = None) -> str:
    if not command:
        framework = _detect_test_framework()
        if not framework:
            return "No test framework detected. Use the 'command' parameter to specify a test command."
        command = TEST_COMMANDS[framework]["cmd"]
        shared.console.print(f"[swarm.accent]Detected: {framework}[/swarm.accent]")
    if pattern:
        command += f" -k {pattern}" if "pytest" in command else f" {pattern}"
    shared.console.print(f"[bold yellow]About to RUN TESTS:[/bold yellow] {command}")
    if shared._auto_approve("Approve test run?"):
        output = _run_tests_raw(command)
        return output
    return "Test run cancelled."


def run_app_capture(command: str, timeout: int = 10, stdin_text: str | None = None) -> str:
    shared.console.print(f"[bold yellow]About to RUN (capture {timeout}s):[/bold yellow] {command}")
    if not shared._auto_approve("Approve app launch?"):
        return "Cancelled by user."
    try:
        proc = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin_text else subprocess.DEVNULL,
            cwd=shared.PROJECT_DIR, text=True,
        )
        try:
            stdout, stderr = proc.communicate(
                input=stdin_text, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=5)
            stdout = (stdout or "") + f"\n[TIMEOUT: process killed after {timeout}s]"
        output = ""
        if stdout and stdout.strip():
            output += f"STDOUT:\n{stdout.strip()}\n"
        if stderr and stderr.strip():
            output += f"STDERR:\n{stderr.strip()}\n"
        output += f"EXIT CODE: {proc.returncode}"
        if len(output) > 12000:
            output = output[:12000] + "\n... (truncated)"
        return output or "(no output)"
    except Exception as e:
        return f"Error: {e}"


def capture_tui_screenshot(command: str, save_path: str = "tui_screenshot.svg",
                           timeout: int = 15, press: str | None = None) -> str:
    full_path = _safe_path(save_path)
    if not full_path:
        return "Access denied: outside project directory."
    shared.console.print(f"[bold yellow]About to SCREENSHOT TUI:[/bold yellow] {command}")
    shared.console.print(f"[dim]Save to: {full_path}[/dim]")
    if not shared._auto_approve("Approve TUI screenshot?"):
        return "Cancelled by user."

    press_code = ""
    if press:
        keys = [k.strip() for k in press.split(",")]
        press_lines = "\n        ".join([f'await pilot.press("{k}")' for k in keys])
        press_code = f"\n        await pilot.pause()\n        {press_lines}"

    driver_script = f'''
import asyncio, sys, importlib.util, os
os.chdir({str(shared.PROJECT_DIR)!r})

module_path = {command!r}
if module_path.endswith(".py"):
    spec = importlib.util.spec_from_file_location("_tui_mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
else:
    mod = importlib.import_module(module_path)

from textual.app import App as _BaseApp
app_cls = None
for name in dir(mod):
    obj = getattr(mod, name)
    if isinstance(obj, type) and issubclass(obj, _BaseApp) and obj is not _BaseApp:
        app_cls = obj
        break

if not app_cls:
    print("ERROR: No Textual App subclass found in " + module_path)
    sys.exit(1)

async def _capture():
    app = app_cls()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(){press_code}
        await pilot.pause()
        app.save_screenshot({str(full_path)!r})

asyncio.run(_capture())
print("SCREENSHOT_SAVED:" + {str(full_path)!r})
'''
    preflight_script = f'''
import sys, importlib.util, os
os.chdir({str(shared.PROJECT_DIR)!r})
module_path = {command!r}
try:
    if module_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("_tui_mod", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        importlib.import_module(module_path)
    print("PREFLIGHT_OK")
except Exception as e:
    print(f"PREFLIGHT_FAIL: {{type(e).__name__}}: {{e}}")
    import traceback; traceback.print_exc()
    sys.exit(1)
'''
    try:
        preflight = subprocess.run(
            [sys.executable, "-c", preflight_script],
            capture_output=True, text=True, cwd=shared.PROJECT_DIR, timeout=10
        )
        if preflight.returncode != 0:
            diag = (preflight.stdout + preflight.stderr).strip()
            return (
                f"Screenshot BLOCKED — the app fails to import.\n"
                f"Fix these errors first, then retry capture_tui_screenshot.\n\n"
                f"--- DIAGNOSTIC OUTPUT ---\n{diag}\n\n"
                f"COMMON TEXTUAL CAUSES:\n"
                f"- @on(Key, \"x\") is invalid — Key events have no CSS selector. Use def key_x(self) method instead.\n"
                f"- Nested CSS selectors (DataTable {{ table {{ }} }}) are invalid — Textual CSS is flat, not SCSS.\n"
                f"- Invalid CSS values (text-style: bold 200%) — Textual has limited CSS support.\n"
                f"- Bad widget params (Input(variants=...)) — check the Textual docs for valid parameters."
            )
    except subprocess.TimeoutExpired:
        pass

    try:
        result = subprocess.run(
            [sys.executable, "-c", driver_script],
            capture_output=True, text=True, cwd=shared.PROJECT_DIR, timeout=timeout
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0 and full_path.exists():
            size = full_path.stat().st_size
            return f"TUI screenshot saved to {save_path} ({size:,} bytes).\nUse analyze_image to inspect visual output.\n{output}"
        else:
            return (
                f"Screenshot failed (exit {result.returncode}):\n{output}\n\n"
                f"NEXT STEPS: Read the traceback above. Use run_app_capture to run "
                f"'python -c \"from {command.removesuffix('.py')} import *\"' for a "
                f"cleaner error. Fix the bug, then retry capture_tui_screenshot to verify."
            )
    except subprocess.TimeoutExpired:
        return f"Timeout: TUI app did not respond within {timeout}s."
    except Exception as e:
        return f"Error: {e}"


# -- Auto-Lint After Edits --
LINT_COMMANDS: dict[str, list[str]] = {
    ".py": ["python", "-m", "py_compile"],
    ".js": ["node", "--check"],
    ".ts": ["npx", "tsc", "--noEmit", "--pretty"],
}


def _lint_file(path: Path) -> str | None:
    ext = path.suffix.lower()
    base_cmd = LINT_COMMANDS.get(ext)
    if not base_cmd:
        return None
    try:
        if ext == ".py":
            import py_compile
            py_compile.compile(str(path), doraise=True)
            return None
        else:
            result = subprocess.run(
                base_cmd + [str(path)],
                capture_output=True, text=True, cwd=shared.PROJECT_DIR, timeout=15
            )
            if result.returncode != 0:
                err = (result.stderr or result.stdout).strip()
                return err[:1500] if err else f"Lint failed (exit {result.returncode})"
            return None
    except Exception as e:
        if 'py_compile' in str(type(e).__module__):
            return str(e)[:1500]
        return None
