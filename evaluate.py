"""
evaluate.py
===========
ASSIGNMENT: Task 9 — Evaluation

PURPOSE:
    Runs a structured functional test suite against the agent and produces:
    1. Per-test PASS/FAIL with timing
    2. Weighted score per category and overall
    3. Scores pushed to Langfuse dashboard (Task 8 integration)
    4. JSON report saved to evaluation/evaluation_report.json

TASKS SATISFIED:
    Task 9 — automated evaluation, weighted scoring, JSON report
    Task 8 — per-category and overall scores sent to Langfuse via add_score()

HOW TO RUN:
    # Local (imports app modules directly — no server needed)
    python evaluation/evaluate.py

    # Against running API server
    python evaluation/evaluate.py --api http://localhost:8000

EVALUATION APPROACH — Functional / Behavioural Testing:
    Each test case uses a check_fn lambda that searches for semantic indicators
    in the response rather than matching exact strings. This is the correct
    approach for LLM evaluation: the same correct behaviour can be expressed
    in many phrasings. Weighted scoring emphasises critical behaviours
    (e.g. not hallucinating task IDs) over peripheral ones (greeting responses).

DEPENDENCIES:
    app.graph           (run_agent — for local mode)
    app.observability   (add_score, flush_langfuse — for Langfuse score push)
    httpx               (for --api mode)
    stdlib: sys, os, json, time, argparse, uuid, datetime
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


# ── Test case definition ───────────────────────────────────────────────────────

class TestCase:
    def __init__(self, name: str, category: str, input_message: str,
                 check_fn: Callable[[str], bool], weight: float = 1.0):
        self.name          = name
        self.category      = category
        self.input_message = input_message
        self.check_fn      = check_fn
        self.weight        = weight


def _any(response: str, keywords: list[str]) -> bool:
    r = response.lower()
    return any(k.lower() in r for k in keywords)

def _all(response: str, keywords: list[str]) -> bool:
    r = response.lower()
    return all(k.lower() in r for k in keywords)


# ── Test suite (13 cases, 4 categories) ───────────────────────────────────────

TEST_CASES = [
    # ── tool_calling ──────────────────────────────────────────────────────────
    TestCase("add_single_task", "tool_calling",
             "Add a task: Buy groceries",
             lambda r: _any(r, ["added", "created", "buy groceries", "✅"]), 1.0),

    TestCase("add_task_synonym", "tool_calling",
             "Create a new task called: Finish the report",
             lambda r: _any(r, ["added", "created", "finish the report", "task"]), 1.0),

    TestCase("add_task_natural", "tool_calling",
             "I need to remember to call the dentist",
             lambda r: _any(r, ["added", "saved", "created", "dentist", "task"]), 1.0),

    TestCase("view_all_tasks", "tool_calling",
             "Show me all my tasks",
             lambda r: _any(r, ["task", "tasks", "📋", "found", "no tasks"]), 1.0),

    TestCase("view_pending", "tool_calling",
             "List only my pending tasks",
             lambda r: _any(r, ["pending", "incomplete", "task", "no tasks", "found"]), 1.0),

    TestCase("view_completed", "tool_calling",
             "Show my completed tasks",
             lambda r: _any(r, ["completed", "done", "finished", "task", "no tasks"]), 1.0),

    # ── task_management ───────────────────────────────────────────────────────
    TestCase("complete_by_name", "task_management",
             "Mark the grocery task as done",
             lambda r: _any(r, ["completed", "marked", "done", "✅", "not found", "no tasks"]), 1.5),

    TestCase("task_id_in_response", "task_management",
             "Add task: Pick up prescription",
             lambda r: _any(r, ["task_", "id:", "ID:", "added", "✅"]), 1.0),

    TestCase("no_hallucinated_ids", "task_management",
             "Complete task task_FAKEID12345",
             lambda r: _any(r, ["not found", "❌", "invalid", "doesn't exist"]), 1.5),

    # ── conversation_quality ──────────────────────────────────────────────────
    TestCase("clarify_ambiguous", "conversation_quality",
             "Delete it",
             lambda r: _any(r, ["which", "clarify", "specify", "what", "id",
                                 "not found", "please", "could you"]), 1.0),

    TestCase("greeting", "conversation_quality",
             "Hello! Can you help me manage my tasks?",
             lambda r: _any(r, ["hello", "hi", "help", "task", "sure", "yes"]), 0.5),

    # ── edge_cases ────────────────────────────────────────────────────────────
    TestCase("empty_list_graceful", "edge_cases",
             "What tasks do I have?",
             lambda r: _any(r, ["task", "tasks", "📋", "no tasks", "found", "none"]), 1.0),

    TestCase("reject_invalid_id", "edge_cases",
             "Complete task with ID: task_doesnotexist999",
             lambda r: _any(r, ["not found", "invalid", "doesn't exist", "❌", "error"]), 1.0),
]


# ── Runners ────────────────────────────────────────────────────────────────────

def _run_local(message: str) -> str:
    from app.graph import run_agent
    return run_agent(message)


def _run_api(message: str, base_url: str) -> str:
    import httpx
    try:
        r = httpx.post(f"{base_url}/chat",
                       json={"message": message, "session_id": str(uuid.uuid4())},
                       timeout=60.0)
        r.raise_for_status()
        return r.json()["response"]
    except Exception as e:
        return f"API_ERROR: {e}"


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(use_api: bool = False, api_url: str = "http://localhost:8000") -> dict:
    print("=" * 70)
    print("SMART TASK MANAGEMENT AGENT — EVALUATION SUITE")
    print("=" * 70)
    print(f"Mode      : {'HTTP API (' + api_url + ')' if use_api else 'Local Python'}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Timestamp : {datetime.now(timezone.utc).isoformat()}")
    print()

    results = []
    cat_stats: dict[str, dict] = {}
    weighted_score = 0.0
    total_weight   = sum(tc.weight for tc in TEST_CASES)

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i:02d}/{len(TEST_CASES)}] {tc.name}  ({tc.category})")
        print(f"        Input   : \"{tc.input_message}\"")

        t0 = time.perf_counter_ns()
        try:
            response = _run_api(tc.input_message, api_url) if use_api else _run_local(tc.input_message)
            duration_ms = round((time.perf_counter_ns() - t0) / 1_000_000, 2)
            passed = tc.check_fn(response)
        except Exception as e:
            response = f"EXCEPTION: {e}"
            duration_ms = round((time.perf_counter_ns() - t0) / 1_000_000, 2)
            passed = False

        status = "✅ PASS" if passed else "❌ FAIL"
        preview = response[:100] + ("..." if len(response) > 100 else "")
        print(f"        Response: \"{preview}\"")
        print(f"        {status} | {duration_ms}ms | weight={tc.weight}")
        print()

        if passed:
            weighted_score += tc.weight

        if tc.category not in cat_stats:
            cat_stats[tc.category] = {"passed": 0, "total": 0, "w_pass": 0.0, "w_total": 0.0}
        cat_stats[tc.category]["total"]   += 1
        cat_stats[tc.category]["w_total"] += tc.weight
        if passed:
            cat_stats[tc.category]["passed"] += 1
            cat_stats[tc.category]["w_pass"]  += tc.weight

        results.append({"name": tc.name, "category": tc.category,
                         "input": tc.input_message, "response": response[:500],
                         "passed": passed, "duration_ms": duration_ms, "weight": tc.weight})

    # ── Summary ────────────────────────────────────────────────────────────────
    overall = weighted_score / total_weight
    passed_n = sum(1 for r in results if r["passed"])

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Overall pass rate : {passed_n}/{len(TEST_CASES)} tests  ({overall:.1%} weighted)")
    print()
    print(f"{'Category':<30} {'Passed':>6} {'Score':>8}")
    print("-" * 48)
    for cat, s in cat_stats.items():
        score = s["w_pass"] / s["w_total"] if s["w_total"] else 0
        print(f"{cat:<30} {s['passed']}/{s['total']:>3}   {score:>7.0%}")
    print()

    # ── Langfuse scores ────────────────────────────────────────────────────────
    try:
        from app.observability import add_score, flush_langfuse
        eval_id = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        add_score(eval_id, "overall_pass_rate", overall,
                  comment=f"{passed_n}/{len(TEST_CASES)} tests passed")
        for cat, s in cat_stats.items():
            add_score(eval_id, f"category_{cat}", s["w_pass"] / s["w_total"] if s["w_total"] else 0)
        flush_langfuse()
        print("Scores pushed to Langfuse dashboard ✅")
    except Exception as e:
        print(f"Langfuse scoring skipped: {e}")

    # ── JSON report ────────────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "api" if use_api else "local",
        "total_tests": len(TEST_CASES),
        "passed": passed_n,
        "failed": len(TEST_CASES) - passed_n,
        "pass_rate": round(overall, 4),
        "category_scores": {
            cat: round(s["w_pass"] / s["w_total"], 4) if s["w_total"] else 0
            for cat, s in cat_stats.items()
        },
        "results": results,
    }
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {report_path}")

    verdict = ("🎉 EXCELLENT" if overall >= 0.8 else
               "⚠️  ACCEPTABLE" if overall >= 0.6 else "❌ NEEDS WORK")
    print(f"\n{verdict} — weighted score: {overall:.1%}")
    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Smart Task Management Agent")
    parser.add_argument("--api", type=str, default=None,
                        help="Test via HTTP API at this URL (e.g. http://localhost:8000). "
                             "Omit to test locally via Python imports.")
    args = parser.parse_args()

    report = (run_evaluation(use_api=True, api_url=args.api)
              if args.api else run_evaluation(use_api=False))

    sys.exit(0 if report["pass_rate"] >= 0.6 else 1)
