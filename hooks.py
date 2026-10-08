"""
hooks.py
========
ASSIGNMENT: Task 5 — Pre/Post Lifecycle Hooks

PURPOSE:
    Four lifecycle hook functions that wrap every reasoning step and tool
    execution. Each hook is a pure state transformer: it receives AgentState,
    does its work (logging, timing, HITL detection), and returns a dict of
    state updates. No side-effectful mutation of shared objects.

TASKS SATISFIED:
    Task 5 — pre_reasoning_hook, post_reasoning_hook, pre_tool_hook, post_tool_hook
    Task 6 — pre_tool_hook detects DESTRUCTIVE_TOOLS and sets pending_action

DEPENDENCIES:
    app.state  (AgentState, HookLogEntry)
    app.tools  (DESTRUCTIVE_TOOLS)

HOW HOOKS ARE WIRED INTO THE GRAPH (graph.py):
    reasoning_node         → calls pre_reasoning_hook BEFORE LLM, post_reasoning_hook AFTER
    pre_tool_interceptor   → is pre_tool_hook (entire node body)
    post_tool_interceptor  → calls post_tool_hook + Langfuse span recording
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any

from state import AgentState, HookLogEntry
from tools import DESTRUCTIVE_TOOLS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# In-process timing store: "session_id:key" → start time in nanoseconds.
# Sufficient for single-worker deployments; swap to a ContextVar for multi-worker.
_TIMING: dict[str, int] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_entry(hook: str, tool_name: str, metadata: dict[str, Any]) -> HookLogEntry:
    return HookLogEntry(hook=hook, tool_name=tool_name,
                        timestamp=_now_iso(), metadata=metadata)


def _extract_tool_name(state: AgentState) -> str:
    """
    Read the tool name from the last AIMessage's tool_calls attribute.
    Returns "unknown" if no tool call is found.
    """
    for msg in reversed(state.get("messages", [])):
        calls = getattr(msg, "tool_calls", None)
        if calls:
            return calls[0].get("name", "unknown")
        # Older LangChain versions surface tool calls in additional_kwargs
        ak_calls = getattr(msg, "additional_kwargs", {}).get("tool_calls", [])
        if ak_calls:
            return ak_calls[0].get("function", {}).get("name", "unknown")
    return "unknown"


# ── Pre-Reasoning Hook ─────────────────────────────────────────────────────────

def pre_reasoning_hook(state: AgentState) -> AgentState:
    """
    Runs BEFORE each call to Gemini inside reasoning_node.

    Actions:
    - Logs session ID and current message count
    - Records nanosecond start time for latency measurement
    - Appends a HookLogEntry to state.hook_log
    """
    sid = state.get("session_id", "unknown")
    msg_count = len(state.get("messages", []))
    logger.info(f"[PRE-REASON] session={sid} | messages={msg_count}")
    _TIMING[f"{sid}:reason"] = time.perf_counter_ns()

    entry = _make_entry("pre_reasoning", "N/A", {"session_id": sid, "message_count": msg_count})
    return {**state, "hook_log": state.get("hook_log", []) + [entry]}


def post_reasoning_hook(state: AgentState, result: dict) -> dict:
    """
    Runs AFTER Gemini responds inside reasoning_node.

    Actions:
    - Calculates Gemini latency in ms
    - Logs whether tool calls were requested
    - Appends a HookLogEntry to result["hook_log"]
    """
    sid = state.get("session_id", "unknown")
    start_ns = _TIMING.pop(f"{sid}:reason", None)
    duration_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2) if start_ns else -1

    new_msgs = result.get("messages", [])
    tool_calls_requested = []
    if new_msgs:
        last = new_msgs[-1]
        tc = getattr(last, "tool_calls", [])
        if tc:
            tool_calls_requested = [c.get("name", "?") for c in tc]

    logger.info(f"[POST-REASON] session={sid} | duration_ms={duration_ms} | tools={tool_calls_requested or 'none'}")

    entry = _make_entry(
        "post_reasoning",
        ", ".join(tool_calls_requested) or "N/A",
        {"session_id": sid, "duration_ms": duration_ms,
         "has_tool_calls": bool(tool_calls_requested), "tools_requested": tool_calls_requested},
    )
    result["hook_log"] = state.get("hook_log", []) + [entry]
    return result


# ── Pre-Tool Hook ──────────────────────────────────────────────────────────────

def pre_tool_hook(state: AgentState) -> dict:
    """
    Runs BEFORE tool execution (entire body of pre_tool_interceptor node).

    Actions:
    - Identifies the tool about to be executed
    - Records nanosecond start time
    - Logs whether the tool is destructive
    - If destructive: sets state.pending_action so hitl_router pauses the graph

    HITL detection happens here because this is the only safe moment —
    after Gemini decided which tool to call, but before execution.
    """
    sid = state.get("session_id", "unknown")
    tool_name = _extract_tool_name(state)
    is_destructive = tool_name in DESTRUCTIVE_TOOLS

    logger.info(f"[PRE-TOOL] session={sid} | tool={tool_name} | destructive={is_destructive}")
    _TIMING[f"{sid}:tool:{tool_name}"] = time.perf_counter_ns()

    entry = _make_entry("pre_tool", tool_name,
                        {"session_id": sid, "is_destructive": is_destructive})
    updates: dict = {"hook_log": state.get("hook_log", []) + [entry]}

    if is_destructive:
        # Extract tool arguments so the HITL prompt can be informative
        tool_args: dict = {}
        tool_call_id: str = ""
        for msg in reversed(state.get("messages", [])):
            calls = getattr(msg, "tool_calls", [])
            if calls:
                tool_args = calls[0].get("args", {})
                tool_call_id = calls[0].get("id", "")
                break

        logger.warning(f"[HITL-REQUIRED] session={sid} | tool={tool_name} | args={tool_args}")
        updates["pending_action"] = {"action": tool_name, "args": tool_args, "tool_call_id": tool_call_id}
        updates["hitl_approved"] = None

    return updates


# ── Post-Tool Hook ─────────────────────────────────────────────────────────────

def post_tool_hook(state: AgentState) -> dict:
    """
    Runs AFTER tool execution (called inside post_tool_interceptor node).

    Actions:
    - Finds the most recent ToolMessage result in state.messages
    - Calculates tool execution duration
    - Logs success/failure
    - Appends a HookLogEntry to state.hook_log
    """
    sid = state.get("session_id", "unknown")
    tool_name = "unknown"
    tool_output = ""

    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "tool":
            tool_name = getattr(msg, "name", "unknown")
            tool_output = str(msg.content)[:300]
            break

    start_ns = _TIMING.pop(f"{sid}:tool:{tool_name}", None)
    duration_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2) if start_ns else -1
    is_success = '"status": "success"' in tool_output or "success" in tool_output.lower()

    logger.info(f"[POST-TOOL] session={sid} | tool={tool_name} | duration_ms={duration_ms} | success={is_success}")

    entry = _make_entry("post_tool", tool_name,
                        {"session_id": sid, "duration_ms": duration_ms,
                         "success": is_success, "result_preview": tool_output[:120]})
    return {"hook_log": state.get("hook_log", []) + [entry]}


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from state import create_initial_state
    from langchain_core.messages import AIMessage

    print("=" * 60)
    print("HOOKS SMOKE TEST")
    print("=" * 60)

    state = create_initial_state("smoke-test-001")

    print("\n[1] pre_reasoning_hook ...")
    state = pre_reasoning_hook(state)
    print(f"    hook_log entries: {len(state['hook_log'])} | last: {state['hook_log'][-1]['hook']}")

    print("\n[2] post_reasoning_hook ...")
    mock_result = {"messages": [AIMessage(content="Test")], "error": None}
    result = post_reasoning_hook(state, mock_result)
    print(f"    hook_log entries: {len(result['hook_log'])} | duration_ms: {result['hook_log'][-1]['metadata']['duration_ms']}")

    print("\n[3] pre_tool_hook (non-destructive, no tool_calls in messages) ...")
    r2 = pre_tool_hook(state)
    print(f"    pending_action set: {r2.get('pending_action') is not None}")

    print(f"\nDESTRUCTIVE_TOOLS = {DESTRUCTIVE_TOOLS}")
    print("\n✅ Hooks smoke test complete.")
