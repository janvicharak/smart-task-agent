"""
observability.py
================
ASSIGNMENT: Task 8 — Langfuse Observability

PURPOSE:
    Wraps the Langfuse Python SDK to record every LLM call, tool execution,
    and session as structured traces visible in the Langfuse cloud dashboard.

    Langfuse hierarchy used here:
        Trace      → one per user session  (trace_session_start / trace_session_end)
        Generation → one per Gemini call   (trace_llm_call)
        Span       → one per tool call     (trace_tool_call)
        Score      → evaluation metric     (add_score — called from evaluate.py)

TASKS SATISFIED:
    Task 8 — Langfuse traces, generations, spans, scores, flush on shutdown

GRACEFUL DEGRADATION:
    If LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY are missing, _get_langfuse()
    returns None and every public function becomes a no-op. The agent never
    fails because observability infrastructure is unavailable.

SETUP (add to .env):
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com   # optional, this is the default

DEPENDENCIES:
    langfuse  (pip install langfuse==2.57.3)
    stdlib: os, logging, datetime
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CLIENT = None                       # Cached Langfuse client
_TRACES: dict[str, Any] = {}         # session_id → open Langfuse trace


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Client initialisation ──────────────────────────────────────────────────────

def _get_langfuse():
    """
    Return the cached Langfuse client, initialising on first call.
    Returns None (silently) if keys are not configured.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    pub = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sec = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not pub or not sec:
        logger.warning("[LANGFUSE] Keys not set — observability disabled. Agent continues normally.")
        return None

    try:
        from langfuse import Langfuse
        client = Langfuse(public_key=pub, secret_key=sec, host=host)
        client.auth_check()
        _CLIENT = client
        logger.info(f"[LANGFUSE] Connected to {host}")
        return client
    except ImportError:
        logger.warning("[LANGFUSE] Package not installed. Run: pip install langfuse")
        return None
    except Exception as e:
        logger.warning(f"[LANGFUSE] Connection failed: {e}")
        return None


# ── Public tracing functions ───────────────────────────────────────────────────

def trace_session_start(session_id: str, user_message: str,
                        metadata: Optional[dict] = None) -> Optional[Any]:
    """
    Open a Langfuse trace for a user session.
    Call this when a new message arrives, before reasoning begins.
    """
    lf = _get_langfuse()
    if not lf:
        return None
    try:
        trace = lf.trace(id=session_id, name="agent-session", input=user_message,
                         metadata=metadata or {}, tags=["task-manager", "langgraph", "gemini"])
        _TRACES[session_id] = trace
        logger.info(f"[LANGFUSE] Trace opened: {session_id}")
        return trace
    except Exception as e:
        logger.warning(f"[LANGFUSE] trace_session_start failed: {e}")
        return None


def trace_llm_call(session_id: str, model: str, input_messages: list,
                   output_content: str, duration_ms: float,
                   tool_calls_made: Optional[list] = None) -> None:
    """Record one Gemini LLM call as a Langfuse generation."""
    lf = _get_langfuse()
    if not lf:
        return
    trace = _TRACES.get(session_id)
    try:
        formatted_input = [
            {"role": _role(m), "content": _content(m)} for m in input_messages
        ]
        kwargs = dict(name="gemini-reasoning", model=model, input=formatted_input,
                      output=output_content,
                      metadata={"duration_ms": duration_ms, "session_id": session_id,
                                "tool_calls_requested": tool_calls_made or []})
        if trace:
            trace.generation(**kwargs)
        else:
            lf.generation(**kwargs)
        logger.info(f"[LANGFUSE] Generation recorded for {session_id} ({duration_ms}ms)")
    except Exception as e:
        logger.warning(f"[LANGFUSE] trace_llm_call failed: {e}")


def trace_tool_call(session_id: str, tool_name: str, tool_input: dict,
                    tool_output: str, duration_ms: float, success: bool) -> None:
    """Record one tool execution as a Langfuse span."""
    lf = _get_langfuse()
    if not lf:
        return
    trace = _TRACES.get(session_id)
    try:
        kwargs = dict(name=f"tool:{tool_name}", input=tool_input, output=tool_output,
                      metadata={"tool_name": tool_name, "duration_ms": duration_ms,
                                "success": success, "session_id": session_id},
                      level="DEFAULT" if success else "WARNING")
        if trace:
            trace.span(**kwargs)
        else:
            lf.span(**kwargs)
        logger.info(f"[LANGFUSE] Tool span recorded: {tool_name} success={success}")
    except Exception as e:
        logger.warning(f"[LANGFUSE] trace_tool_call failed: {e}")


def trace_session_end(session_id: str, final_response: str,
                      total_duration_ms: float, success: bool = True,
                      error: Optional[str] = None) -> None:
    """Close the Langfuse trace for a session with final output and status."""
    lf = _get_langfuse()
    if not lf:
        return
    trace = _TRACES.pop(session_id, None)
    try:
        update = dict(output=final_response,
                      metadata={"total_duration_ms": total_duration_ms,
                                "success": success, "error": error},
                      status_message="success" if success else f"error: {error}")
        if trace:
            trace.update(**update)
        lf.flush()
        logger.info(f"[LANGFUSE] Trace closed: {session_id} success={success} {total_duration_ms}ms")
    except Exception as e:
        logger.warning(f"[LANGFUSE] trace_session_end failed: {e}")


def add_score(session_id: str, score_name: str, score_value: float,
              comment: Optional[str] = None) -> None:
    """Attach an evaluation score (0.0–1.0) to a Langfuse trace."""
    lf = _get_langfuse()
    if not lf:
        return
    try:
        lf.score(trace_id=session_id, name=score_name,
                 value=score_value, comment=comment)
        logger.info(f"[LANGFUSE] Score: {score_name}={score_value} on {session_id}")
    except Exception as e:
        logger.warning(f"[LANGFUSE] add_score failed: {e}")


def flush_langfuse() -> None:
    """Force-flush all queued Langfuse events. Call at application shutdown."""
    lf = _get_langfuse()
    if lf:
        try:
            lf.flush()
            logger.info("[LANGFUSE] Flushed.")
        except Exception as e:
            logger.warning(f"[LANGFUSE] flush failed: {e}")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _role(msg) -> str:
    t = getattr(msg, "type", "")
    return {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(t, "unknown")


def _content(msg) -> str:
    c = getattr(msg, "content", "")
    return (c if isinstance(c, str) else str(c))[:500]


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("LANGFUSE OBSERVABILITY SMOKE TEST")
    print("=" * 60)

    lf = _get_langfuse()
    print(f"\n[1] Langfuse connected: {lf is not None}")
    if not lf:
        print("    → Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to .env")

    SID = "smoke-test-langfuse-001"

    print("\n[2] trace_session_start ...")
    trace_session_start(SID, "Add task: buy milk", {"test": True})

    print("[3] trace_llm_call ...")
    trace_llm_call(SID, "gemini-1.5-flash", [], "Task added.", 840.0, ["add_task"])

    print("[4] trace_tool_call ...")
    trace_tool_call(SID, "add_task", {"title": "buy milk"}, '{"status":"success"}', 12.0, True)

    print("[5] add_score ...")
    add_score(SID, "task_success", 1.0, "Tool returned success status.")

    print("[6] trace_session_end ...")
    trace_session_end(SID, "I've added 'buy milk'!", 1200.0, True)

    if lf:
        print("\n✅ Check https://cloud.langfuse.com — look for trace 'agent-session'.")
    print("\n✅ Observability smoke test complete (no crashes).")
