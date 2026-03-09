"""
eval_runner.py — Live evaluation runner for GrokSwarm

Runs real coding tasks against the live xAI API and measures results.
Can be run directly or by GrokSwarm itself (via /run or coder expert).

Usage:
  python eval_runner.py                    # Run all tasks
  python eval_runner.py --task A1 B2       # Run specific tasks
  python eval_runner.py --category B       # Run all bug-fix tasks
  python eval_runner.py --list             # List available tasks
  python eval_runner.py --dry-run          # Show what would run without calling API
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure we can import grokswarm
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure API key is set
if not os.environ.get("XAI_API_KEY"):
    print("ERROR: XAI_API_KEY not set. Export it or create .env file.")
    sys.exit(1)

from eval_grokswarm import (
    EVAL_TASKS, EvalTask, EvalResult,
    _setup_workspace, _run_checks, run_eval_task_live, format_report,
)
from dataclasses import asdict
import grokswarm.shared as shared


async def run_all(task_ids: list[str] | None = None, category: str | None = None,
                  dry_run: bool = False):
    """Run eval tasks and produce report."""
    tasks = EVAL_TASKS

    if task_ids:
        tasks = [t for t in tasks if t.id in task_ids]
    if category:
        tasks = [t for t in tasks if t.category == category.upper()]

    if not tasks:
        print("No matching tasks found.")
        return

    print(f"\nGrokSwarm Live Eval — {len(tasks)} task(s)")
    print(f"Model: {shared.MODEL}")
    print(f"Project: {shared.PROJECT_DIR}")
    print()

    if dry_run:
        print("DRY RUN — no API calls will be made.\n")
        for task in tasks:
            print(f"  [{task.category}] {task.id}: {task.description}")
            print(f"       Expert: {task.expert} | Checks: {len(task.checks)}")
            for rel_path in task.setup_files:
                print(f"       Setup: {rel_path}")
        return

    results: list[EvalResult] = []

    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(tasks)}] {task.id} — {task.description}")
        print(f"  Expert: {task.expert} | Category: {task.category}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory(prefix=f"gseval_{task.id}_") as tmp:
            workspace = Path(tmp)
            result = await run_eval_task_live(task, workspace)
            results.append(result)

            # Print immediate feedback
            status = "PASS" if result.correct >= 1.0 else "PARTIAL" if result.correct > 0 else "FAIL"
            print(f"\n  Result: {status} ({result.correct:.0%})")
            print(f"  Rounds: {result.rounds_used} | Tokens: {result.tokens_used:,} | "
                  f"Cost: ${result.cost_usd:.4f} | Time: {result.time_seconds:.1f}s")

            if result.error:
                print(f"  ERROR: {result.error[:100]}")

            for detail in result.check_details:
                mark = "+" if detail["passed"] else "x"
                print(f"  [{mark}] {detail['message'][:70]}")

    # Final report
    report = format_report(results)
    print(f"\n{report}")

    # Save results
    report_dir = Path(".grokswarm")
    report_dir.mkdir(exist_ok=True)

    report_path = report_dir / "eval_report.txt"
    report_path.write_text(report)

    json_path = report_dir / "eval_results.json"
    json_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": shared.MODEL,
        "tasks_run": len(results),
        "avg_score": sum(r.correct for r in results) / len(results) if results else 0,
        "total_tokens": sum(r.tokens_used for r in results),
        "total_cost_usd": sum(r.cost_usd for r in results),
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(json_data, indent=2))

    print(f"\nReport: {report_path}")
    print(f"JSON:   {json_path}")

    # Summary
    passed = sum(1 for r in results if r.correct >= 1.0)
    partial = sum(1 for r in results if 0 < r.correct < 1.0)
    failed = sum(1 for r in results if r.correct == 0)
    print(f"\nSummary: {passed} passed, {partial} partial, {failed} failed out of {len(results)}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GrokSwarm Live Eval Runner")
    parser.add_argument("--task", nargs="*", help="Specific task IDs (e.g., A1 B2)")
    parser.add_argument("--category", help="Run all tasks in category (A/B/C/D)")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument("--dry-run", action="store_true", help="Show tasks without running")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'ID':<25} {'Cat':>4} {'Expert':<12} {'Checks':>6}  Description")
        print("-" * 80)
        for t in EVAL_TASKS:
            print(f"{t.id:<25} {t.category:>4} {t.expert:<12} {len(t.checks):>6}  {t.description}")
        print(f"\nTotal: {len(EVAL_TASKS)} tasks")
        return

    task_ids = args.task if args.task else None
    asyncio.run(run_all(task_ids=task_ids, category=args.category, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
