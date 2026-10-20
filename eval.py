"""Run the agent on every task in evals/tasks.json and report how many it solved.

Usage:
    python eval.py                           # all tasks
    python eval.py --task inv-fix-low-stock  # one task (repeatable)
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import agent
import tools
from logger import RunLogger

ROOT = Path(__file__).parent
TASKS_FILE = ROOT / "evals" / "tasks.json"
REPOS_DIR = ROOT / "evals" / "repos"
RESULTS_DIR = ROOT / "eval_results"


# ---------- Grading helpers ----------
def run_pytest(workspace):
    """Run the repo's tests. Returns (could_run, set of failed test ids, short output)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-rf", "-p", "no:cacheprovider"],
        cwd=workspace, capture_output=True, text=True, timeout=120,
    )
    failed = set(re.findall(r"^FAILED (\S+)", proc.stdout, re.MULTILINE))
    could_run = proc.returncode in (0, 1)   # 0 = all passed, 1 = some failed, anything else = broken
    return could_run, failed, proc.stdout.strip().splitlines()[-1:] or [proc.stderr[-200:]]


def grade(task, answer, workspace, failing_before):
    """Return (passed, reason)."""
    expect = task["expect"]

    if "answer_contains" in expect:
        missing = [s for s in expect["answer_contains"] if s.lower() not in answer.lower()]
        if missing:
            return False, f"answer is missing: {missing}"

    if "tests_pass" in expect:
        could_run, failing_after, last_line = run_pytest(workspace)
        if not could_run:
            return False, f"tests could not run (broken code?): {last_line}"
        still_failing = [t for t in expect["tests_pass"] if t in failing_after]
        if still_failing:
            return False, f"still failing: {still_failing}"
        newly_broken = sorted(failing_after - failing_before)
        if newly_broken:
            return False, f"fixed the target but broke: {newly_broken}"

    return True, "ok"


# ---------- Running one task ----------
def auto_approve(diff_text):
    print(diff_text)
    print("[eval] edit auto-approved")
    return True, ""


def run_task(task):
    # 1. Fresh copy of the repo, so tasks never affect each other
    workspace = Path(tempfile.mkdtemp(prefix=f"eval-{task['id']}-")) / task["repo"]
    shutil.copytree(REPOS_DIR / task["repo"], workspace,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))

    # 2. Point the tools at the copy, and approve edits automatically
    tools.WORKSPACE = workspace.resolve()
    tools.ask_approval = auto_approve
    _, failing_before, _ = run_pytest(workspace) if "tests_pass" in task["expect"] else (True, set(), "")

    # 3. Run the agent with its own log
    log = RunLogger(task["prompt"], agent.MODEL, tools.WORKSPACE)
    log.record["task_id"] = task["id"]
    try:
        answer = agent.run_agent(task["prompt"], log=log)
        passed, reason = grade(task, answer, workspace, failing_before)   # 4. Grade the result
    except Exception as e:
        print(f"[eval] agent crashed: {type(e).__name__}: {e}")
        passed, reason = False, f"agent crashed: {type(e).__name__}: {e}"
    summary = log.record.get("summary", {})
    shutil.rmtree(workspace.parent, ignore_errors=True)
    return {
        "id": task["id"], "kind": task["kind"], "passed": passed, "reason": reason,
        "outcome": summary.get("outcome"), "steps": summary.get("steps", 0),
        "tokens": summary.get("tokens_in", 0) + summary.get("tokens_out", 0),
        "tool_errors": summary.get("tool_errors", 0), "api_errors": summary.get("api_errors", 0),
        "seconds": summary.get("seconds", 0), "log": f"{log.run_id}.json",
    }


# ---------- Report ----------
def print_report(results):
    print("\n" + "=" * 92)
    print(f"{'task':<24}{'result':<8}{'steps':>6}{'tokens':>9}{'errors':>8}   reason")
    print("-" * 92)
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        errors = r["tool_errors"] + r["api_errors"]
        print(f"{r['id']:<24}{mark:<8}{r['steps']:>6}{r['tokens']:>9}{errors:>8}   {r['reason']}")
    print("-" * 92)
    n = len(results)
    passed = sum(r["passed"] for r in results)
    print(f"Passed {passed}/{n} ({100 * passed / n:.0f}%) | "
          f"avg steps {sum(r['steps'] for r in results) / n:.1f} | "
          f"total tokens {sum(r['tokens'] for r in results)}")
    for kind in sorted({r["kind"] for r in results}):
        group = [r for r in results if r["kind"] == kind]
        print(f"  {kind}: {sum(r['passed'] for r in group)}/{len(group)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", help="run only this task id (can repeat)")
    args = parser.parse_args()

    tasks = json.loads(TASKS_FILE.read_text())
    if args.task:
        tasks = [t for t in tasks if t["id"] in args.task]
        if not tasks:
            raise SystemExit(f"No tasks match {args.task}")

    results = []
    for i, task in enumerate(tasks, start=1):
        print(f"\n######## [{i}/{len(tasks)}] {task['id']} ########")
        results.append(run_task(task))

    print_report(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"model": agent.MODEL, "results": results}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()