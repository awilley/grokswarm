"""Run live eval tasks by category. Usage: python run_eval.py [A|B|C|D|all]"""
import os
import sys

# Force UTF-8 on Windows to handle emoji in API responses
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE importing eval_grokswarm (which sets a test key fallback)
load_dotenv()

from eval_grokswarm import EVAL_TASKS, run_eval_task_live, format_report

async def run(categories):
    tasks = [t for t in EVAL_TASKS if t.category in categories]
    if not tasks:
        print(f"No tasks for categories: {categories}")
        return

    results = []
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Running: {task.id} -- {task.description}")
        print(f"{'='*60}", flush=True)

        with tempfile.TemporaryDirectory(prefix=f"eval_{task.id}_", ignore_cleanup_errors=True) as tmp:
            workspace = Path(tmp)
            r = await run_eval_task_live(task, workspace)
            results.append(r)
            print(f"  Score: {r.correct:.0%} | Rounds: {r.rounds_used} | "
                  f"Cost: ${r.cost_usd:.4f} | Time: {r.time_seconds:.1f}s")
            for d in r.check_details:
                status = "PASS" if d["passed"] else "FAIL"
                print(f"  [{status}] {d['check']}: {d['message'][:60]}")
            if r.error:
                print(f"  ERROR: {r.error[:200]}")

    report = format_report(results)
    print(f"\n{report}")

    # Save results
    report_path = Path(".grokswarm") / "eval_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report)

    import json
    from dataclasses import asdict
    json_path = Path(".grokswarm") / "eval_results.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nSaved: {report_path}, {json_path}")


if __name__ == "__main__":
    cats = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if "all" in cats:
        categories = {"A", "B", "C", "D"}
    else:
        categories = set(c.upper() for c in cats)
    asyncio.run(run(categories))
