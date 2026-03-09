"""Git tools + checkpoint constants."""

import subprocess
from pathlib import Path

import grokswarm.shared as shared
from grokswarm.context import _safe_path

# -- Auto-Checkpoint Constants --
AUTO_CHECKPOINT_THRESHOLD = 5
MAX_EDIT_HISTORY = 20

# -- Worktree directory --
WORKTREE_DIR = ".grokswarm/worktrees"


def _run_git(*args: str, cwd: str | Path | None = None) -> str:
    work_dir = str(cwd) if cwd else str(shared.get_project_dir())
    try:
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True, text=True, cwd=work_dir, timeout=30
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"Error (exit {result.returncode}): {err}" if err else f"Error (exit {result.returncode})"
        return output if output else "(no output)"
    except FileNotFoundError:
        return "Error: git is not installed or not in PATH."
    except subprocess.TimeoutExpired:
        return "Error: git command timed out."


def git_status() -> str:
    return _run_git("status", "--short", "--branch")


def git_diff(path: str | None = None, staged: bool = False) -> str:
    args = ["diff"]
    if staged:
        args.append("--staged")
    args.append("--stat")
    if path:
        safe = _safe_path(path)
        if not safe:
            return "Access denied: outside project directory."
        args.append(path)
    summary = _run_git(*args)
    detail_args = ["diff"]
    if staged:
        detail_args.append("--staged")
    if path:
        detail_args.append(path)
    detail = _run_git(*detail_args)
    if len(detail) > 3000:
        detail = detail[:3000] + "\n... (truncated)"
    return f"{summary}\n\n{detail}"


def git_log(count: int = 10) -> str:
    count = min(max(count, 1), 50)
    return _run_git("log", f"--oneline", f"-{count}", "--decorate")


def git_commit(message: str) -> str:
    shared.console.print(f"[bold yellow]About to COMMIT:[/bold yellow] {message}")
    status = _run_git("status", "--short")
    shared.console.print(f"[dim]{status}[/dim]")
    if shared._auto_approve("Approve commit? (will stage all changes)"):
        _run_git("add", "-A")
        return _run_git("commit", "-m", message)
    return "Commit cancelled by user."


def git_checkout(target: str) -> str:
    shared.console.print(f"[bold yellow]About to CHECKOUT:[/bold yellow] {target}")
    safe = _safe_path(target)
    if safe and safe.exists():
        shared.console.print("[dim]This will discard uncommitted changes to this file.[/dim]")
        if shared._auto_approve("Approve file restore?", default=False):
            return _run_git("checkout", "--", target)
        return "Checkout cancelled by user."
    else:
        shared.console.print(f"[dim]Switching to branch/commit: {target}[/dim]")
        if shared._auto_approve("Approve branch switch?"):
            return _run_git("checkout", target)
        return "Checkout cancelled by user."


def git_branch(name: str | None = None, delete: bool = False) -> str:
    if not name:
        return _run_git("branch", "-a", "--no-color")
    if delete:
        shared.console.print(f"[bold yellow]About to DELETE branch:[/bold yellow] {name}")
        if shared._auto_approve("Approve branch deletion?", default=False):
            return _run_git("branch", "-d", name)
        return "Branch deletion cancelled by user."
    return _run_git("branch", name)


def git_show_file(path: str, ref: str = "HEAD") -> str:
    safe = _safe_path(path)
    if not safe:
        return "Access denied: outside project directory."
    return _run_git("show", f"{ref}:{path}")


def git_blame(path: str) -> str:
    safe = _safe_path(path)
    if not safe:
        return "Access denied: outside project directory."
    result = _run_git("blame", "--date=short", path)
    if len(result) > 5000:
        result = result[:5000] + "\n... (truncated)"
    return result


def git_stash(action: str = "list", message: str | None = None) -> str:
    action = action.lower()
    if action == "list":
        return _run_git("stash", "list") or "(no stashes)"
    elif action == "push":
        shared.console.print(f"[bold yellow]About to STASH changes:[/bold yellow] {message or '(no message)'}")
        if shared._auto_approve("Approve stash?"):
            args = ["stash", "push"]
            if message:
                args.extend(["-m", message])
            return _run_git(*args)
        return "Stash cancelled by user."
    elif action == "pop":
        shared.console.print("[bold yellow]About to POP stash[/bold yellow] (apply + drop top stash)")
        if shared._auto_approve("Approve stash pop?"):
            return _run_git("stash", "pop")
        return "Stash pop cancelled by user."
    elif action == "drop":
        shared.console.print("[bold yellow]About to DROP top stash[/bold yellow] (permanently removes it)")
        if shared._auto_approve("Approve stash drop?", default=False):
            return _run_git("stash", "drop")
        return "Stash drop cancelled by user."
    else:
        return f"Unknown stash action: '{action}'. Use: list, push, pop, drop."


def git_init() -> str:
    git_dir = shared.get_project_dir() / ".git"
    if git_dir.exists():
        return "Already a git repository."
    shared.console.print("[bold yellow]About to INIT git repo[/bold yellow]")
    if shared._auto_approve("Initialize a new git repository here?"):
        return _run_git("init")
    return "Git init cancelled by user."


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def git_merge(branch: str, *, no_ff: bool = False, message: str | None = None) -> str:
    """Merge a branch into the current branch."""
    shared.console.print(f"[bold yellow]About to MERGE:[/bold yellow] {branch}")
    if not shared._auto_approve(f"Approve merge of '{branch}' into current branch?"):
        return "Merge cancelled by user."
    args = ["merge"]
    if no_ff:
        args.append("--no-ff")
    if message:
        args.extend(["-m", message])
    args.append(branch)
    result = _run_git(*args)
    if "CONFLICT" in result or "Automatic merge failed" in result:
        # Get the list of conflicted files
        conflicts = _run_git("diff", "--name-only", "--diff-filter=U")
        return f"MERGE CONFLICT:\n{result}\n\nConflicted files:\n{conflicts}"
    return result


# ---------------------------------------------------------------------------
# Worktree management
# ---------------------------------------------------------------------------

def git_worktree_add(branch_name: str, base_ref: str = "HEAD") -> str:
    """Create a git worktree for an agent to work in isolation.

    Creates a new branch from base_ref and checks it out in a worktree
    directory under .grokswarm/worktrees/<branch_name>/.

    Returns the absolute path to the worktree directory.
    """
    worktree_base = shared.PROJECT_DIR / WORKTREE_DIR
    worktree_base.mkdir(parents=True, exist_ok=True)
    worktree_path = worktree_base / branch_name

    if worktree_path.exists():
        return f"Error: worktree path already exists: {worktree_path}"

    # Create a new branch + worktree in one step
    result = _run_git("worktree", "add", "-b", branch_name, str(worktree_path), base_ref)
    if result.startswith("Error"):
        return result

    return str(worktree_path)


def git_worktree_remove(branch_name: str, *, force: bool = False) -> str:
    """Remove a worktree and optionally delete its branch."""
    worktree_path = shared.PROJECT_DIR / WORKTREE_DIR / branch_name

    if not worktree_path.exists():
        return f"Error: worktree not found: {worktree_path}"

    # Remove the worktree
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(worktree_path))
    result = _run_git(*args)
    if result.startswith("Error"):
        return result

    # Delete the branch (best-effort)
    _run_git("branch", "-D", branch_name)

    return f"Worktree removed: {worktree_path}"


def git_worktree_list() -> str:
    """List all active worktrees."""
    return _run_git("worktree", "list")


def git_merge_abort() -> str:
    """Abort an in-progress merge."""
    return _run_git("merge", "--abort")
