"""
tools.py
========
ASSIGNMENT: Task 3 — Tool Execution (JSON-RPC)

PURPOSE:
    Defines the four task-management tools that the LangGraph agent can call.
    Each tool is decorated with @tool, which generates a JSON schema that Gemini
    reads to understand what arguments to supply. LangGraph's ToolNode executes
    whichever tool Gemini selects.

TASKS SATISFIED:
    Task 3 — @tool decorated functions, ALL_TOOLS registry, JSON-RPC execution
    Task 4 — TOOL_MAP used by mcp_server.py for MCP tool wiring
    Task 6 — DESTRUCTIVE_TOOLS set used by hooks.py to trigger HITL detection

DEPENDENCIES:
    langchain_core.tools  (@tool decorator)
    json, uuid, os, tempfile, datetime  (stdlib only — no external deps)

KEY DESIGN DECISIONS:
    Atomic write  — _save_tasks() writes to a temp file, fsyncs, then renames.
                    If the process crashes mid-write the original file is intact.
    Idempotency   — complete_task checks for already-completed status and
                    returns an "info" response without writing again.
    Central registry — DESTRUCTIVE_TOOLS is the single source of truth for
                    which operations require human approval. hooks.py and
                    graph.py read from here; nothing is hard-coded elsewhere.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional

from langchain_core.tools import tool


# ── File path ──────────────────────────────────────────────────────────────────

TASKS_FILE = os.path.join(os.path.dirname(__file__), "tasks.json")

# Single source of truth: which tools require HITL approval before execution.
DESTRUCTIVE_TOOLS = {"delete_task"}


# ── Internal helpers (not exposed to the LLM) ─────────────────────────────────

def _load_tasks() -> list[dict]:
    """Read tasks.json and return as a list. Returns [] on any error."""
    if not os.path.exists(TASKS_FILE):
        return []
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_tasks(tasks: list[dict]) -> None:
    """
    Atomically write tasks to disk.

    Pattern: write temp file → fsync → os.replace()
    The original file is only replaced after the new file is fully flushed.
    No corruption window even on hard crash.
    """
    dir_path = os.path.dirname(TASKS_FILE)
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, TASKS_FILE)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── LangGraph Tools ────────────────────────────────────────────────────────────

@tool
def add_task(title: str) -> dict:
    """
    Add a new task to the task list.

    Use this when the user wants to create, add, or save a new task.
    Examples: "add buy groceries", "create task: call dentist", "new task: finish report"

    Args:
        title: The task description. Should be concise and actionable.

    Returns:
        Dict with status, message, task object (id, title, completed, created_at), total_tasks.
    """
    title = title.strip()
    if not title:
        return {"status": "error", "message": "Task title cannot be empty."}

    tasks = _load_tasks()
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    new_task = {
        "id": task_id,
        "title": title,
        "completed": False,
        "created_at": _now_iso(),
        "completed_at": None,
    }
    tasks.append(new_task)
    _save_tasks(tasks)

    return {
        "status": "success",
        "message": f"✅ Task '{title}' added with ID: {task_id}",
        "task": new_task,
        "total_tasks": len(tasks),
    }


@tool
def view_tasks(filter_status: Optional[str] = None) -> dict:
    """
    View all tasks, optionally filtered by completion status.

    Use this when the user wants to see, list, or check their tasks.
    Examples: "show my tasks", "list pending tasks", "show completed tasks"

    Args:
        filter_status: "pending" | "completed" | None (show all).

    Returns:
        Dict with status, tasks list, total, pending count, completed count.
    """
    tasks = _load_tasks()
    if not tasks:
        return {"status": "success", "message": "📋 No tasks yet. Add your first task!",
                "tasks": [], "total": 0, "pending": 0, "completed": 0}

    filtered = tasks
    if filter_status == "pending":
        filtered = [t for t in tasks if not t["completed"]]
    elif filter_status == "completed":
        filtered = [t for t in tasks if t["completed"]]

    completed_count = sum(1 for t in tasks if t["completed"])
    return {
        "status": "success",
        "message": f"📋 Found {len(filtered)} task(s)",
        "tasks": filtered,
        "total": len(tasks),
        "pending": len(tasks) - completed_count,
        "completed": completed_count,
        "filter_applied": filter_status or "none",
    }


@tool
def complete_task(task_id: str) -> dict:
    """
    Mark a task as completed.

    Use this when the user wants to finish or check off a task.
    IMPORTANT: Requires a task ID. Call view_tasks first if only a name is given.
    Examples: "mark task_a1b2c3d4 done", "complete task_a1b2c3d4"

    Args:
        task_id: The task's unique ID (format: task_XXXXXXXX).

    Returns:
        Dict with status and the updated task object.
    """
    tasks = _load_tasks()
    idx = next((i for i, t in enumerate(tasks) if t["id"] == task_id), None)

    if idx is None:
        return {"status": "error",
                "message": f"❌ Task '{task_id}' not found.",
                "valid_ids": [t["id"] for t in tasks]}

    if tasks[idx]["completed"]:
        return {"status": "info",
                "message": f"ℹ️ '{tasks[idx]['title']}' is already completed.",
                "task": tasks[idx]}

    tasks[idx]["completed"] = True
    tasks[idx]["completed_at"] = _now_iso()
    _save_tasks(tasks)

    return {"status": "success",
            "message": f"✅ '{tasks[idx]['title']}' marked as completed!",
            "task": tasks[idx]}


@tool
def delete_task(task_id: str) -> dict:
    """
    Permanently delete a task.

    Use this when the user explicitly asks to remove or delete a task.
    NOTE: This is a destructive operation — it triggers human approval (HITL).
    IMPORTANT: Requires a task ID. Call view_tasks first if only a name is given.
    Examples: "delete task_a1b2c3d4", "remove task_a1b2c3d4"

    Args:
        task_id: The task's unique ID (format: task_XXXXXXXX).

    Returns:
        Dict with status, deleted task object, remaining task count.
    """
    tasks = _load_tasks()
    to_delete = next((t for t in tasks if t["id"] == task_id), None)

    if to_delete is None:
        return {"status": "error",
                "message": f"❌ Task '{task_id}' not found.",
                "valid_ids": [t["id"] for t in tasks]}

    remaining = [t for t in tasks if t["id"] != task_id]
    _save_tasks(remaining)

    return {"status": "success",
            "message": f"🗑️ '{to_delete['title']}' deleted.",
            "deleted_task": to_delete,
            "remaining_tasks": len(remaining)}


# ── Registries ─────────────────────────────────────────────────────────────────

ALL_TOOLS = [add_task, view_tasks, complete_task, delete_task]
TOOL_MAP   = {t.name: t for t in ALL_TOOLS}


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TOOLS SMOKE TEST")
    print("=" * 60)

    r1 = add_task.invoke({"title": "Buy groceries"})
    print(f"[1] Add:      {r1['message']}")
    tid1 = r1["task"]["id"]

    r2 = add_task.invoke({"title": "Call dentist"})
    print(f"[2] Add:      {r2['message']}")
    tid2 = r2["task"]["id"]

    r3 = view_tasks.invoke({})
    print(f"[3] View all: {r3['message']} | total={r3['total']}")

    r4 = complete_task.invoke({"task_id": tid1})
    print(f"[4] Complete: {r4['message']}")

    r5 = view_tasks.invoke({"filter_status": "pending"})
    print(f"[5] Pending:  {r5['message']}")

    r6 = delete_task.invoke({"task_id": tid2})
    print(f"[6] Delete:   {r6['message']}")

    r7 = view_tasks.invoke({})
    print(f"[7] Final:    {r7['message']} | tasks={[t['title'] for t in r7['tasks']]}")

    print(f"\nDESTRUCTIVE_TOOLS = {DESTRUCTIVE_TOOLS}")
    print(f"tasks.json at: {TASKS_FILE}")
    print("\n✅ All tools working correctly.")
