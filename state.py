"""
state.py
========
ASSIGNMENT: Task 1 — State Management

PURPOSE:
    Defines the AgentState TypedDict that flows through every node in the
    LangGraph agent. Every field here is actively read or written by at least
    one node. No ghost fields.

TASKS SATISFIED:
    Task 1 — AgentState schema, add_messages reducer, factory function
    Task 5 — hook_log field carries lifecycle hook audit trail through state
    Task 6 — pending_action and hitl_approved coordinate HITL interrupt
    Task 7 — session_id used as checkpointer thread_id
    Task 8 — session_id used as Langfuse trace_id
"""

from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langgraph.graph.message import add_messages


# ── Supporting Types ───────────────────────────────────────────────────────────

class TaskItem(TypedDict):
    """Schema for a single persisted task."""
    id: str
    title: str
    completed: bool
    created_at: str           # ISO 8601 UTC
    completed_at: Optional[str]


class HookLogEntry(TypedDict):
    """
    One lifecycle hook audit record appended to state.hook_log.
    Persisted inside SQLite checkpoints — full audit trail survives restarts.
    """
    hook: str                  # pre_reasoning | post_reasoning | pre_tool | post_tool
    tool_name: str             # tool name, or "N/A" for reasoning hooks
    timestamp: str             # ISO 8601 UTC
    metadata: Dict[str, Any]  # duration_ms, success, session_id, etc.


# ── Primary State Schema ───────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The single shared state object that flows through every LangGraph node.

    Field descriptions
    ------------------
    messages
        Full conversation history. The add_messages reducer APPENDS new
        messages rather than replacing the list — preserving multi-turn context.

    pending_action
        Task 6 (HITL): When a destructive tool is requested, the action is
        stored here and the graph pauses at the human_review interrupt point.
        Format: {"action": "delete_task", "args": {"task_id": "task_abc"}}

    hitl_approved
        Task 6 (HITL): Injected by the API after the human responds.
        True  → execute the pending action.
        False → cancel and inform the user.
        None  → no HITL decision pending.

    hook_log
        Task 5 (Hooks): List of HookLogEntry structs appended by pre/post
        lifecycle hooks. Accumulates across the full session.

    last_action_result
        Passthrough for the most recent tool return value. Read by the
        reasoning node to compose its response.

    error
        If any node fails, the error string is stored here rather than
        raising an exception. The reasoning node can then respond gracefully.

    session_id
        Unique identifier per conversation session.
        Used as checkpointer thread_id (Task 7) and Langfuse trace_id (Task 8).
    """
    messages:            Annotated[list, add_messages]
    pending_action:      Optional[Dict[str, Any]]
    hitl_approved:       Optional[bool]
    hook_log:            List[HookLogEntry]
    last_action_result:  Optional[Dict[str, Any]]
    error:               Optional[str]
    session_id:          str


# ── Factory ────────────────────────────────────────────────────────────────────

def create_initial_state(session_id: str) -> AgentState:
    """
    Return a fully initialised AgentState with safe defaults.

    Always use this factory instead of constructing AgentState manually —
    it ensures every field is present and correctly typed from the start.

    Args:
        session_id: UUID string for this conversation session.
    """
    return AgentState(
        messages=[],
        pending_action=None,
        hitl_approved=None,
        hook_log=[],
        last_action_result=None,
        error=None,
        session_id=session_id,
    )
