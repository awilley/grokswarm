"""Filesystem tools: list_dir, read_file, write_file, edit_file, search_files, grep_files."""

import os
import re
import difflib
import shutil
import subprocess
from pathlib import Path

import grokswarm.shared as shared
from grokswarm.context import (
    _safe_path, _grokswarm_read_path, _should_ignore,
    _iter_project_files,
)


def list_dir(path: str = "."):
    if path.startswith("@grokswarm/"):
        full_path = _grokswarm_read_path(path)
        if not full_path:
            return "Access denied: outside grokswarm directory."
    else:
        full_path = _safe_path(path)
        if not full_path:
            return "Access denied: outside project directory."
    if not full_path.exists():
        return "Path not found."
    entries = sorted(full_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = []
    for p in entries:
        if p.is_dir():
            lines.append(f"  {p.name}/")
        else:
            size = p.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024*1024):.1f} MB"
            lines.append(f"  {p.name:<35} {size_str}")
    return "\n".join(lines) if lines else "(empty directory)"


def read_file(path: str, start_line: int | None = None, end_line: int | None = None):
    if path.startswith("@grokswarm/"):
        full_path = _grokswarm_read_path(path)
        if not full_path:
            return "Access denied: outside grokswarm directory."
        if not (full_path.exists() and full_path.is_file()):
            return "File not found in grokswarm directory."
    else:
        full_path = _safe_path(path)
        if not full_path:
            return "Access denied: outside project directory."
        if not (full_path.exists() and full_path.is_file()):
            return "File not found."
    if start_line is None and end_line is None:
        try:
            fsize = full_path.stat().st_size
            if fsize > 1_048_576:
                return f"Warning: {path} is {fsize:,} bytes ({fsize / 1048576:.1f} MB). Use read_file with start_line/end_line for partial reads, or grep_files to search for specific content."
        except OSError:
            pass
    text = full_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    total = len(lines)
    if start_line is not None or end_line is not None:
        s = max(1, start_line or 1)
        e = min(total, end_line or total)
        selected = lines[s-1:e]
        numbered = [f"{i:>4} | {line}" for i, line in enumerate(selected, s)]
        header = f"[{path} lines {s}-{e} of {total}]"
        return header + "\n" + "\n".join(numbered)
    return text


def write_file(path: str, content: str):
    full_path = _safe_path(path)
    if not full_path:
        return "Access denied: outside project directory."
    if shared.state.agent_mode == 0:
        shared.console.print(f"[bold yellow]About to WRITE to:[/bold yellow] {full_path}")
        if full_path.is_file():
            try:
                from rich.syntax import Syntax
                current = full_path.read_text(encoding="utf-8", errors="ignore")
                diff_lines = list(difflib.unified_diff(
                    current.splitlines(), content.splitlines(),
                    fromfile="current", tofile="new", lineterm=""))
                if diff_lines:
                    display = diff_lines[:50]
                    extra = len(diff_lines) - 50
                    diff_text = "\n".join(display)
                    if extra > 0:
                        diff_text += f"\n... (+{extra} more lines)"
                    shared.console.print(Syntax(diff_text, "diff", theme="monokai"))
                else:
                    shared.console.print("[dim]No changes detected.[/dim]")
            except Exception:
                preview = content[:300] + ("..." if len(content) > 300 else "")
                shared.console.print(f"[dim]Preview:[/dim]\n{preview}")
        else:
            shared.console.print(f"[dim]New file ({len(content)} chars):[/dim]")
            preview = content[:300] + ("..." if len(content) > 300 else "")
            shared.console.print(f"[dim]{preview}[/dim]")
    if shared._auto_approve("Approve write?", default=False):
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"Written: {full_path}"
    return "Cancelled."


def _apply_single_edit(full_path: Path, content: str, old_text: str, new_text: str, show_preview: bool = True) -> tuple[str | None, str]:
    count = content.count(old_text)
    if count == 0:
        return None, f"Error: old_text not found in {full_path.name}. The text to replace must match exactly (including whitespace and indentation). Use read_file to verify the current content."
    if count > 1:
        return None, f"Error: old_text found {count} times in {full_path.name}. Include more surrounding context to make it unique."
    if show_preview:
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()
        shared.console.print(f"[dim]  Replacing {len(old_lines)} line(s) with {len(new_lines)} line(s):[/dim]")
        for line in old_lines[:8]:
            shared.console.print(f"    [red]- {line}[/red]")
        if len(old_lines) > 8:
            shared.console.print(f"    [dim]  ... ({len(old_lines) - 8} more lines)[/dim]")
        for line in new_lines[:8]:
            shared.console.print(f"    [green]+ {line}[/green]")
        if len(new_lines) > 8:
            shared.console.print(f"    [dim]  ... ({len(new_lines) - 8} more lines)[/dim]")
    return content.replace(old_text, new_text, 1), f"{len(old_text.splitlines())} lines -> {len(new_text.splitlines())} lines"


def edit_file(path: str, old_text: str = "", new_text: str = "", edits: list | None = None) -> str:
    full_path = _safe_path(path)
    if not full_path:
        return "Access denied: outside project directory."
    if not full_path.exists() or not full_path.is_file():
        return f"File not found: {path}"
    content = full_path.read_text(encoding="utf-8", errors="ignore")
    edit_list = edits if edits else [{"old_text": old_text, "new_text": new_text}]
    if not edit_list or (len(edit_list) == 1 and not edit_list[0].get("old_text")):
        return "Error: no edits provided. Supply old_text/new_text or an edits array."
    if shared.state.agent_mode == 0:
        shared.console.print(f"[bold yellow]About to EDIT:[/bold yellow] {full_path} ({len(edit_list)} edit{'s' if len(edit_list) != 1 else ''})")
    for i, edit in enumerate(edit_list):
        ot = edit.get("old_text", "")
        if not ot:
            return f"Error: edit #{i+1} has empty old_text."
        cnt = content.count(ot)
        if cnt == 0:
            return f"Error: edit #{i+1} old_text not found in {path}. Use read_file to verify current content."
        if cnt > 1:
            return f"Error: edit #{i+1} old_text found {cnt} times in {path}. Include more surrounding context to make it unique."
    if shared.state.agent_mode == 0:
        for i, edit in enumerate(edit_list):
            if len(edit_list) > 1:
                shared.console.print(f"  [swarm.accent]edit {i+1}/{len(edit_list)}:[/swarm.accent]")
            _apply_single_edit(full_path, content, edit["old_text"], edit.get("new_text", ""), show_preview=True)
    if shared._auto_approve("Approve edit?"):
        results = []
        for i, edit in enumerate(edit_list):
            new_content, msg = _apply_single_edit(full_path, content, edit["old_text"], edit.get("new_text", ""), show_preview=False)
            if new_content is None:
                return f"Error applying edit #{i+1}: {msg}"
            content = new_content
            results.append(msg)
        full_path.write_text(content, encoding="utf-8")
        summary = "; ".join(results)
        return f"Edited: {full_path} ({len(edit_list)} edit{'s' if len(edit_list) != 1 else ''}: {summary})"
    return "Edit cancelled."


def search_files(query: str):
    results = []
    for p, rel in _iter_project_files(shared.get_project_dir()):
        if query.lower() in p.name.lower():
            results.append(f"  {rel}")
    for dirpath, dirnames, _ in os.walk(shared.get_project_dir()):
        dirnames[:] = [d for d in dirnames if not _should_ignore(d) and not d.startswith(".")]
        dp = Path(dirpath)
        if query.lower() in dp.name.lower() and dp != shared.get_project_dir():
            results.append(f"  {dp.relative_to(shared.get_project_dir())}/")
    return "\n".join(results) if results else "No matches found."


def grep_files(pattern: str, path: str = ".", is_regex: bool = False, context_lines: int = 0) -> str:
    search_root = _safe_path(path)
    if not search_root:
        return "Access denied: outside project directory."
    if not search_root.exists():
        return "Path not found."
    if shutil.which("rg"):
        try:
            rg_args = ["rg", "--no-heading", "--line-number", "--color=never",
                       "-i", "--max-count=200"]
            if context_lines:
                rg_args += [f"-C{min(context_lines, 10)}"]
            if not is_regex:
                rg_args.append("--fixed-strings")
            rg_args += ["--", pattern, str(search_root)]
            rg = subprocess.run(rg_args, capture_output=True, text=True, cwd=shared.get_project_dir(), timeout=15)
            if rg.returncode in (0, 1):
                rg_out = rg.stdout.strip()
                return rg_out if rg_out else "No matches found."
        except Exception:
            pass
    results = []
    if is_regex:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex: {e}"
    else:
        search_lower = pattern.lower()
    ctx = min(max(context_lines, 0), 10)
    if search_root.is_dir():
        targets = _iter_project_files(search_root, max_files=2000)
    else:
        targets = [(search_root, search_root.relative_to(shared.get_project_dir()))]
    for p, rel in targets:
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2",
                                  ".ttf", ".eot", ".zip", ".tar", ".gz", ".exe", ".dll",
                                  ".so", ".dylib", ".pyc", ".pyo", ".db", ".sqlite"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        file_lines = text.splitlines()
        matched_ranges: set[int] = set()
        for i, line in enumerate(file_lines):
            match = rx.search(line) if is_regex else (search_lower in line.lower())
            if match:
                for offset in range(-ctx, ctx + 1):
                    matched_ranges.add(i + offset)
        if not matched_ranges:
            continue
        prev_idx = -2
        for idx in sorted(matched_ranges):
            if idx < 0 or idx >= len(file_lines):
                continue
            if prev_idx >= 0 and idx > prev_idx + 1:
                results.append("  --")
            line_num = idx + 1
            is_match = (rx.search(file_lines[idx]) if is_regex else (search_lower in file_lines[idx].lower()))
            marker = ":" if is_match else "-"
            results.append(f"  {rel}{marker}{line_num}: {file_lines[idx].rstrip()[:120]}")
            prev_idx = idx
            if len(results) >= 200:
                results.append("  ... (max 200 matches reached)")
                return "\n".join(results)
    return "\n".join(results) if results else "No matches found."
