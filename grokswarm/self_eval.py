"""Self-evaluation loop: run evals, analyze failures, fix, re-eval.

The /self-eval command chains:
1. Run eval suite → capture results
2. Parse failures → identify what broke
3. Feed failures into agent as fix targets
4. Re-run eval to verify fixes
"""

import sys
import asyncio
import subprocess
from pathlib import Path

import grokswarm.shared as shared


async def run_self_eval_loop(category: str = "all", max_rounds: int = 3) -> str:
    """Run eval → fix → re-eval loop up to max_rounds times."""
    from grokswarm.engine import _stream_with_tools, _trim_conversation

    project_dir = shared.PROJECT_DIR
    eval_script = project_dir / "eval_grokswarm.py"
    if not eval_script.exists():
        return "eval_grokswarm.py not found in project directory."

    results_log = []

    for round_num in range(1, max_rounds + 1):
        shared.console.print(f"\n[bold cyan]Self-Eval Round {round_num}/{max_rounds}[/bold cyan]")

        # Step 1: Run eval
        shared.console.print("[dim]Running evaluation suite...[/dim]")
        eval_args = [sys.executable, str(eval_script), "--live"]
        if category != "all":
            eval_args.extend(["--category", category])

        proc = await asyncio.to_thread(
            subprocess.run, eval_args,
            capture_output=True, text=True, timeout=600,
            cwd=str(project_dir),
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
        )

        output = proc.stdout or ""
        stderr = proc.stderr or ""
        full_output = output + "\n" + stderr

        # Step 2: Parse results
        failures = _parse_eval_failures(full_output)
        total, passed = _parse_eval_summary(full_output)

        results_log.append({
            "round": round_num,
            "total": total,
            "passed": passed,
            "failures": failures,
        })

        shared.console.print(f"[bold]Results: {passed}/{total} passed[/bold]")

        if not failures:
            shared.console.print("[bold green]All evals passed![/bold green]")
            break

        shared.console.print(f"[yellow]Failures ({len(failures)}):[/yellow]")
        for f in failures:
            shared.console.print(f"  [red]- {f}[/red]")

        if round_num >= max_rounds:
            shared.console.print("[yellow]Max rounds reached. Stopping.[/yellow]")
            break

        # Step 3: Run tests first to see current state
        shared.console.print("[dim]Running test suite to check baseline...[/dim]")
        test_proc = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "pytest", "test_grokswarm.py", "-q", "--tb=short"],
            capture_output=True, text=True, timeout=120,
            cwd=str(project_dir),
        )
        test_output = test_proc.stdout or ""
        test_pass = test_proc.returncode == 0

        if not test_pass:
            shared.console.print("[yellow]Tests are failing — fixing tests first before re-eval.[/yellow]")

        # Step 4: Feed failures to agent for fixing
        failure_desc = "\n".join(f"- {f}" for f in failures)
        fix_prompt = f"""[SELF-EVAL FIX PROTOCOL]

The evaluation suite found these failures (round {round_num}):

{failure_desc}

Test suite status: {"PASSING" if test_pass else "FAILING"}
{("Test output:\n" + test_output[-500:]) if not test_pass else ""}

Your task:
1. Analyze each failure — read the relevant eval task definitions and test code
2. Identify the root cause in the GrokSwarm source code
3. Fix the issues (edit the source files, NOT the eval tasks)
4. Run the test suite to verify your fixes don't break anything

IMPORTANT: Only fix the GrokSwarm source code. Do NOT modify eval tasks or test expectations.
Focus on making the eval pass by fixing the underlying capability."""

        conversation = [
            {"role": "system", "content": shared.SYSTEM_PROMPT},
            {"role": "user", "content": fix_prompt},
        ]
        conversation = await _trim_conversation(conversation)
        await _stream_with_tools(conversation)

    # Summary
    lines = ["\n[Self-Eval Summary]"]
    for r in results_log:
        status = "PASS" if not r["failures"] else f"FAIL ({len(r['failures'])} failures)"
        lines.append(f"  Round {r['round']}: {r['passed']}/{r['total']} — {status}")

    return "\n".join(lines)


def _parse_eval_failures(output: str) -> list[str]:
    """Parse eval output for failure descriptions."""
    failures = []
    for line in output.split("\n"):
        line = line.strip()
        # Look for FAIL/FAILED markers
        if "FAIL" in line.upper() and ("eval" in line.lower() or "task" in line.lower()):
            failures.append(line[:200])
        elif line.startswith("FAILED "):
            failures.append(line[:200])
        elif "score: 0" in line.lower() or "0/1" in line:
            failures.append(line[:200])
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in failures:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique[:10]  # cap at 10


def _parse_eval_summary(output: str) -> tuple[int, int]:
    """Parse eval output for total/passed counts."""
    import re
    # Look for patterns like "3/3" or "Score: 100%" or "12 passed"
    for line in reversed(output.split("\n")):
        m = re.search(r"(\d+)/(\d+)", line)
        if m:
            passed, total = int(m.group(1)), int(m.group(2))
            if 0 < total <= 100:
                return total, passed
        m = re.search(r"(\d+)\s+passed", line)
        if m:
            return int(m.group(1)), int(m.group(1))
    return 0, 0
