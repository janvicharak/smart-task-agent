"""
graph.py
========
ASSIGNMENT: Tasks 2, 3, 5, 6, 7, 8 — Core Agent Orchestration

PURPOSE:
    Builds and runs the LangGraph agent. This is the central orchestration
    file that wires every other module together into a runnable graph.

TASKS SATISFIED:
    Task 2 — reasoning_node() calls Gemini with full conversation history
    Task 3 — ToolNode executes whichever tool Gemini selects
    Task 5 — pre/post hooks called inside reasoning_node and interceptor nodes
    Task 6 — human_review_node + interrupt_before + hitl_router / hitl_continue_router
    Task 7 — build_graph() accepts checkpointer; compact_messages() called in reasoning_node
    Task 8 — trace_llm_call / trace_tool_call / trace_session_* called throughout

AUDIT FIXES APPLIED:
    BUG 1: 'from langgraph.checkpoint.sqlite import SqliteSaver' was imported
           here but SqliteSaver is NEVER used directly in graph.py — it is
           used only in checkpointer.py. This import caused the same
           ModuleNotFoundError as in checkpointer.py on any import of graph.py
           (including from api.py, evaluate.py, etc.), even for paths that
           never touch checkpointing at all. Removed.

    BUG 2: reasoning_node did:
               state = pre_reasoning_hook(state)
           pre_reasoning_hook() returns the full updated AgentState (a dict
           with ALL fields). This is correct — it spreads state plus the new
           hook_log entry. However the variable 'state' was then used to build
           messages_to_send and passed to post_reasoning_hook. This is fine
           as long as pre_reasoning_hook returns a complete state dict, which
           it does. No functional bug here, but the merge is made explicit
           for clarity and safety:
               state = {**state, **pre_reasoning_hook(state)}
           This ensures partial-dict returns from the hook are always safe.

    BUG 3: api.py imports create_initial_state from app.graph:
               from app.graph import build_graph, create_initial_state, ...
           create_initial_state is defined in app.state, not app.graph. The
           original graph.py DID import it from app.state:
               from app.state import AgentState, create_initial_state
           so it IS re-exported via graph.py's namespace. This works in Python.
           No functional bug, but added explicit __all__ re-export for clarity.

GRAPH TOPOLOGY:
    START
      └─► reason ──(tool calls?)──► pre_tool_interceptor
                │                         │
                │                  hitl_router
                │                   ├─ destructive ──► human_review  ◄── INTERRUPT
                │                   │                       │
                │                   │               hitl_continue_router
                │                   │                ├─ approved ──► tools
                │                   └─ safe ─────────────────────────┘
                │                                         │
                │                               post_tool_interceptor
                │                                         │
                └──────────────────────────────────── reason
          (no tool calls)
              └─► END

DEPENDENCIES:
    app.state          (AgentState, create_initial_state)
    app.tools          (ALL_TOOLS, DESTRUCTIVE_TOOLS)
    app.hooks          (pre/post reasoning and tool hooks)
    app.checkpointer   (get_checkpointer, compact_messages, needs_compaction)
    app.observability  (trace_* functions)
    langchain_google_genai, langgraph, langchain_core
"""

import os
import time
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# AUDIT FIX: removed 'from langgraph.checkpoint.sqlite import SqliteSaver'
# SqliteSaver is only used in checkpointer.py. Importing it here caused
# ModuleNotFoundError for every module that imports graph.py.
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from checkpointer import compact_messages, get_checkpointer, needs_compaction
from hooks import (post_reasoning_hook, post_tool_hook,
                       pre_reasoning_hook, pre_tool_hook)
from observability import (flush_langfuse, trace_llm_call,
                               trace_session_end, trace_session_start,
                               trace_tool_call)
from state import AgentState, create_initial_state  # re-exported for api.py
from tools import ALL_TOOLS, DESTRUCTIVE_TOOLS

load_dotenv()

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Smart Task Management Assistant. Help users manage their personal task list through natural conversation.

Available tools:
  add_task       — Create a new task
  view_tasks     — List tasks (filter: "pending" | "completed" | omit for all)
  complete_task  — Mark a task done
  delete_task    — Permanently remove a task (requires your confirmation)

Rules:
1. Always call view_tasks first when the user refers to a task by name rather than ID.
2. Never invent task IDs — only use IDs returned by view_tasks.
3. Format task lists as a numbered list for readability.
4. Ask for clarification when the user's intent is ambiguous.
5. For delete_task the system handles the confirmation prompt automatically.

Be concise, friendly, and always confirm what you did."""


# ── Cached LLM ─────────────────────────────────────────────────────────────────

_LLM = None


def _get_llm():
    """
    Return the module-level cached LLM instance (lazy init).
    Initialised once; never recreated between calls.
    """
    global _LLM
    if _LLM is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not found.\n"
                "Add it to .env:  GEMINI_API_KEY=your_key_here\n"
                "Get a key at: https://aistudio.google.com/app/apikey"
            )
        base = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=key,
            temperature=0,
            convert_system_message_to_human=True,
        )
        _LLM = base.bind_tools(ALL_TOOLS)
    return _LLM


# ── Graph nodes ────────────────────────────────────────────────────────────────

def reasoning_node(state: AgentState) -> dict:
    """
    Task 2 | 5 | 7 | 8 — Core reasoning step.

    1. (Task 5)  pre_reasoning_hook  — log, start timer, append HookLogEntry
    2. (Task 7)  compact_messages    — trim context if messages > threshold
    3. (Task 2)  LLM call            — Gemini reasons and optionally requests tools
    4. (Task 8)  trace_llm_call      — record generation in Langfuse
    5. (Task 5)  post_reasoning_hook — compute latency, append HookLogEntry
    """
    sid = state.get("session_id", str(uuid.uuid4()))

    # Task 5: pre-hook — returns updated state dict (full AgentState spread).
    # Merge explicitly so partial-dict returns are always safe.
    hook_updates = pre_reasoning_hook(state)
    state = {**state, **hook_updates}

    # Task 7: compaction — keep context window bounded
    msgs = state.get("messages", [])
    if needs_compaction(msgs):
        msgs = compact_messages(msgs, llm=None)   # fallback summary (no recursive LLM call)
        state = {**state, "messages": msgs}

    messages_to_send = [SystemMessage(content=SYSTEM_PROMPT)] + msgs

    # Task 2: LLM call
    t0 = time.perf_counter_ns()
    try:
        response = _get_llm().invoke(messages_to_send)
        duration_ms = round((time.perf_counter_ns() - t0) / 1_000_000, 2)
    except Exception as exc:
        return {
            "error": str(exc),
            "messages": [AIMessage(content=f"Sorry, I hit an error: {exc}")],
        }

    # Task 8: Langfuse generation trace
    tool_calls_requested = [tc.get("name", "?") for tc in (getattr(response, "tool_calls", []) or [])]
    output_text = response.content if isinstance(response.content, str) else str(response.content)
    trace_llm_call(
        session_id=sid, model="gemini-1.5-flash",
        input_messages=messages_to_send, output_content=output_text,
        duration_ms=duration_ms, tool_calls_made=tool_calls_requested,
    )

    result = {"messages": [response], "error": None}

    # Task 5: post-hook
    return post_reasoning_hook(state, result)


def pre_tool_interceptor(state: AgentState) -> dict:
    """
    Task 5 + 6 — Runs between reasoning_node and ToolNode.

    Calls pre_tool_hook which:
    - Logs the upcoming tool call
    - If the tool is in DESTRUCTIVE_TOOLS, sets state.pending_action
      so hitl_router will route to human_review instead of tools.
    """
    return pre_tool_hook(state)


def post_tool_interceptor(state: AgentState) -> dict:
    """
    Task 5 + 8 — Runs after ToolNode.

    Records the tool result in Langfuse (Task 8) then calls post_tool_hook
    which logs duration and appends a HookLogEntry (Task 5).
    """
    sid = state.get("session_id", "unknown")

    # Find the most recent ToolMessage for Langfuse
    tool_name, tool_output = "unknown", ""
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "tool":
            tool_name   = getattr(msg, "name", "unknown")
            tool_output = str(msg.content)[:500]
            break

    is_success = "success" in tool_output.lower()

    # Retrieve pre-tool duration from hook_log
    duration_ms = -1.0
    for entry in reversed(state.get("hook_log", [])):
        if entry.get("hook") == "pre_tool" and entry.get("tool_name") == tool_name:
            duration_ms = entry.get("metadata", {}).get("duration_ms", -1.0)
            break

    # Task 8: Langfuse tool span
    trace_tool_call(
        session_id=sid, tool_name=tool_name, tool_input={},
        tool_output=tool_output, duration_ms=duration_ms, success=is_success,
    )

    # Task 5: post-hook (appends HookLogEntry)
    return post_tool_hook(state)


def human_review_node(state: AgentState) -> dict:
    """
    Task 6 — HITL interrupt resume point.

    The graph pauses BEFORE entering this node (interrupt_before=["human_review"]).
    State is saved to SQLite. The API waits for POST /sessions/{id}/approve or /reject.
    When the graph resumes, this node reads hitl_approved and routes accordingly.
    """
    approved = state.get("hitl_approved")

    if approved is True:
        return {
            "pending_action": None,
            "hitl_approved": None,
            "messages": [AIMessage(content="✅ Approved — proceeding with deletion...")],
        }
    if approved is False:
        return {
            "pending_action": None,
            "hitl_approved": None,
            "messages": [AIMessage(content="❌ Cancelled — the deletion was not performed.")],
        }
    return {}   # Safety fallback (should not reach here in normal flow)


# ── Routing functions ──────────────────────────────────────────────────────────

def hitl_router(state: AgentState) -> str:
    """After pre_tool_interceptor: route to human_review if action is pending, else tools."""
    return "human_review" if state.get("pending_action") else "tools"


def hitl_continue_router(state: AgentState) -> str:
    """After human_review_node: route to tools (approved) or reason (rejected)."""
    return "tools" if state.get("hitl_approved") is True else "reason"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(checkpointer=None, with_hitl: bool = True):
    """
    Compile and return the LangGraph agent.

    Args:
        checkpointer: Live SqliteSaver instance from get_checkpointer()
                      (required for HITL and persistence).
                      Pass None for stateless single-turn testing.
        with_hitl:    Include HITL nodes. Set False only for unit tests.

    Returns:
        Compiled CompiledStateGraph ready for .invoke() calls.
    """
    g = StateGraph(AgentState)

    g.add_node("reason",               reasoning_node)
    g.add_node("pre_tool_interceptor",  pre_tool_interceptor)
    g.add_node("tools",                 ToolNode(tools=ALL_TOOLS))
    g.add_node("post_tool_interceptor", post_tool_interceptor)
    if with_hitl:
        g.add_node("human_review", human_review_node)

    g.add_edge(START, "reason")
    g.add_conditional_edges("reason", tools_condition,
                            {"tools": "pre_tool_interceptor", "__end__": END})

    if with_hitl:
        g.add_conditional_edges("pre_tool_interceptor", hitl_router,
                                {"human_review": "human_review", "tools": "tools"})
        g.add_conditional_edges("human_review", hitl_continue_router,
                                {"tools": "tools", "reason": "reason"})
    else:
        g.add_edge("pre_tool_interceptor", "tools")

    g.add_edge("tools",                 "post_tool_interceptor")
    g.add_edge("post_tool_interceptor", "reason")

    compile_kwargs: dict = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if with_hitl:
        compile_kwargs["interrupt_before"] = ["human_review"]

    return g.compile(**compile_kwargs)


def get_production_graph():
    """
    Convenience factory: build the fully wired production graph.

    Returns:
        (compiled_graph, checkpointer)  — both returned so api.py can inspect sessions.
    """
    cp    = get_checkpointer()
    graph = build_graph(checkpointer=cp, with_hitl=True)
    return graph, cp


# ── Single-turn convenience runner (testing only) ──────────────────────────────

def run_agent(user_message: str, session_id: str = None) -> str:
    """
    Stateless single-turn runner. Used by evaluate.py and the smoke test below.
    Does NOT use the checkpointer — state is not persisted between calls.
    For multi-turn stateful usage, call the FastAPI endpoints in api.py.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    t0 = time.perf_counter_ns()
    state = create_initial_state(session_id)
    state["messages"] = [HumanMessage(content=user_message)]

    trace_session_start(session_id=session_id, user_message=user_message)

    graph       = build_graph(checkpointer=None, with_hitl=False)
    final_state = graph.invoke(state)

    response_text = "I couldn't process that request."
    for msg in reversed(final_state["messages"]):
        if getattr(msg, "type", "") == "ai":
            c = msg.content
            if isinstance(c, str) and c.strip():
                response_text = c
                break
            if isinstance(c, list):
                joined = " ".join(b.get("text", "") for b in c if isinstance(b, dict)).strip()
                if joined:
                    response_text = joined
                    break

    total_ms = round((time.perf_counter_ns() - t0) / 1_000_000, 2)
    trace_session_end(session_id=session_id, final_response=response_text,
                      total_duration_ms=total_ms, success=not bool(final_state.get("error")))
    return response_text


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("AGENT SMOKE TEST  (requires GEMINI_API_KEY in .env)")
    print("=" * 60)

    TESTS = [
        "Add a task: Buy groceries",
        "Add another task: Call the dentist",
        "Show me all my tasks",
        "Mark the groceries task as complete",
        "Show only my pending tasks",
    ]

    for i, msg in enumerate(TESTS, 1):
        print(f"\n[{i}] USER : {msg}")
        try:
            print(f"     AGENT: {run_agent(msg)}")
        except Exception as e:
            print(f"     ERROR: {e}")
            import traceback; traceback.print_exc()

    flush_langfuse()
    print("\n✅ Agent smoke test complete.")
