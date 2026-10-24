"""Run the agent on every task in evals/tasks.json and report how many it solved.

Usage:
    python eval.py                                  # all tasks, model from .env / defaults
    python eval.py --task inv-fix-low-stock         # one task (repeatable)
    python eval.py --provider ollama --model qwen2.5-coder:7b   # compare another model
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from agent import Agent
from llm import LLM
from logger import RunLogger
from tools import Workspace

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


def grade(task, answer, workspace, failing_before, before_snapshot):
    """Return (passed, reason)."""
    expect = task["expect"]

    if task["kind"] == "question":
        after = snapshot(workspace)
        changed = sorted(f for f in set(before_snapshot) | set(after) if before_snapshot.get(f) != after.get(f))
        if changed:
            return False, f"changed files during a question task: {changed}"

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


def snapshot(workspace):
    """Fingerprint every file, so we can tell afterwards whether anything changed."""
    return {
        str(p.relative_to(workspace)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(workspace.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts
    }


def hide_tests(workspace, hidden_dir):
    """Move test files out of the agent's view. Returns the list of moved paths."""
    moved = []
    for test_file in sorted(workspace.rglob("test_*.py")):
        rel = test_file.relative_to(workspace)
        target = hidden_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(test_file), target)
        moved.append(rel)
    return moved


def restore_tests(workspace, hidden_dir, moved):
    """Put the original tests back, overwriting anything the agent created with the same name."""
    for rel in moved:
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hidden_dir / rel, target)


def run_task(task, llm):
    # 1. Fresh copy of the repo, so tasks never affect each other
    temp = Path(tempfile.mkdtemp(prefix=f"eval-{task['id']}-"))
    workspace = temp / task["repo"]
    shutil.copytree(REPOS_DIR / task["repo"], workspace,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    if not any(workspace.iterdir()):
        raise SystemExit(f"Setup error: {REPOS_DIR / task['repo']} is empty.")

    # 2. Record which tests fail BEFORE the agent runs; the target tests must be among them
    failing_before = set()
    if "tests_pass" in task["expect"]:
        could_run, failing_before, last_line = run_pytest(workspace)
        not_failing = [t for t in task["expect"]["tests_pass"] if t not in failing_before]
        if not could_run or not_failing:
            raise SystemExit(f"Setup error in {task['id']}: these tests should fail before the agent runs, "
                             f"but don't: {not_failing or last_line}. Is the bug still in {REPOS_DIR / task['repo']}?")

    # 3. Hide the tests from the agent (fix tasks only, unless the task says otherwise)
    moved = []
    if task.get("hide_tests", task["kind"] == "fix"):
        moved = hide_tests(workspace, temp / "hidden_tests")
    before_snapshot = snapshot(workspace)

    # 4. An agent that works on the copy and approves its own edits (the copy is thrown away anyway)
    agent = Agent(llm, Workspace(workspace, approve=auto_approve))

    # 5. Run the agent with its own log, then put the real tests back and grade
    log = RunLogger(task["prompt"], llm.model, agent.workspace.root)
    log.record["task_id"] = task["id"]
    try:
        answer = agent.run(task["prompt"], log=log)
        restore_tests(workspace, temp / "hidden_tests", moved)
        passed, reason = grade(task, answer, workspace, failing_before, before_snapshot)
    except Exception as e:
        print(f"[eval] agent crashed: {type(e).__name__}: {e}")
        passed, reason = False, f"agent crashed: {type(e).__name__}: {e}"
    summary = log.record.get("summary", {})
    shutil.rmtree(temp, ignore_errors=True)
    return {
        "id": task["id"], "kind": task["kind"], "passed": passed, "reason": reason,
        "outcome": summary.get("outcome"), "steps": summary.get("steps", 0),
        "tokens": summary.get("tokens_in", 0) + summary.get("tokens_out", 0),
        "tool_errors": summary.get("tool_errors", 0), "api_errors": summary.get("api_errors", 0),
        "seconds": summary.get("seconds", 0), "log": f"{log.run_id}.json",
    }


# ---------- Report ----------
def print_report(results, llm):
    print("\n" + "=" * 92)
    print(f"Model: {llm.provider} / {llm.model}")
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
    parser.add_argument("--provider", help="groq, openai, openrouter, ollama, ... (default: AGENT_PROVIDER or groq)")
    parser.add_argument("--model", help="model name (default: AGENT_MODEL or the provider's default)")
    parser.add_argument("--base-url", help="for any other OpenAI-compatible server")
    args = parser.parse_args()
    llm = LLM.from_env(args.provider, args.model, args.base_url)
    print(f"[eval] model: {llm}")

    tasks = json.loads(TASKS_FILE.read_text())
    if args.task:
        tasks = [t for t in tasks if t["id"] in args.task]
        if not tasks:
            raise SystemExit(f"No tasks match {args.task}")

    results = []
    for i, task in enumerate(tasks, start=1):
        print(f"\n######## [{i}/{len(tasks)}] {task['id']} ########")
        results.append(run_task(task, llm))

    print_report(results, llm)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"provider": llm.provider, "model": llm.model, "results": results}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()