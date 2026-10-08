"""
checkpointer.py
===============
ASSIGNMENT: Task 7 — Context Compaction & Checkpointing

PURPOSE:
    Two production concerns addressed in one module:

    1. CHECKPOINTING — SqliteSaver persists the full AgentState to a local
       SQLite database after every node execution. On server restart the agent
       loads the last saved state for a given session_id (thread_id) and
       continues seamlessly. This is also required for HITL (Task 6) to work:
       interrupt_before pauses the graph and saves state so execution can be
       resumed after the human responds.

    2. CONTEXT COMPACTION — As conversations grow, the message list sent to
       Gemini grows with them, consuming context window and increasing latency.
       compact_messages() uses a sliding-window + summary strategy: once the
       message count exceeds COMPACTION_THRESHOLD, older messages are replaced
       by a single SystemMessage summary, keeping the context bounded.

TASKS SATISFIED:
    Task 6 — SqliteSaver is required for interrupt_before to function
    Task 7 — SqliteSaver persistence, compact_messages() compaction

AUDIT FIXES APPLIED:
    BUG 1: 'from langgraph.checkpoint.sqlite import SqliteSaver' raises
           ModuleNotFoundError because langgraph-checkpoint-sqlite was missing
           from requirements.txt. Fixed in requirements.txt.

    BUG 2: SqliteSaver.from_conn_string() returns a CONTEXT MANAGER, not a
           live saver. The original get_checkpointer() returned the context
           manager object directly — the SQLite connection was never actually
           opened and the schema tables were never created. Any subsequent
           checkpoint read/write would fail silently or raise.
           Fix: open the context manager once at startup and store the live
           saver as a module-level singleton. The connection stays open for
           the lifetime of the process (correct for a single-worker server).

    BUG 3: get_checkpoint_stats() and list_checkpointed_sessions() accessed
           checkpointer.conn directly. In langgraph-checkpoint-sqlite 2.x,
           .conn is an internal attribute that may not be set unless the
           context manager has been entered. After the BUG 2 fix the saver
           is always properly initialised, but we also switch to a direct
           sqlite3.connect() for stats queries — cleaner, version-safe, and
           doesn't depend on any undocumented internal attribute.

DEPENDENCIES:
    langgraph-checkpoint-sqlite  (langgraph.checkpoint.sqlite.SqliteSaver)
    langchain_core.messages      (SystemMessage, HumanMessage, AIMessage)
    stdlib: os, logging, sqlite3
"""

import logging
import os
import sqlite3

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

CHECKPOINT_DB_PATH = os.path.join(
    os.path.dirname(__file__), "checkpoints", "agent_checkpoints.db"
)

COMPACTION_THRESHOLD = 12   # Compact when message count exceeds this
MESSAGES_TO_KEEP     = 6    # Recent messages preserved verbatim after compaction

# Module-level singleton — opened once, lives for the process lifetime.
_saver: SqliteSaver | None = None


# ── Checkpointer factory ───────────────────────────────────────────────────────

def get_checkpointer() -> SqliteSaver:
    """
    Open (or return the already-open) SqliteSaver for CHECKPOINT_DB_PATH.

    AUDIT FIX: SqliteSaver.from_conn_string() is a context manager factory,
    not a direct constructor. Calling it without entering the context leaves
    the SQLite connection unopened. We enter the context exactly once here
    and cache the live saver for reuse across all calls.

    The connection intentionally stays open for the process lifetime. For a
    single-worker FastAPI server this is correct. For multi-worker deployments
    switch to PostgresSaver (langgraph-checkpoint-postgres) which handles
    concurrent access correctly.
    """
    global _saver
    if _saver is not None:
        return _saver

    os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
    logger.info(f"[CHECKPOINT] Opening SQLite store: {CHECKPOINT_DB_PATH}")

    # Enter the context manager to open the connection and create schema tables.
    cm = SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
    _saver = cm.__enter__()
    return _saver


# ── Context compaction ─────────────────────────────────────────────────────────

def needs_compaction(messages: list) -> bool:
    """Return True when the message list exceeds COMPACTION_THRESHOLD."""
    return len(messages) > COMPACTION_THRESHOLD


def compact_messages(messages: list, llm=None) -> list:
    """
    Reduce a long message list to [summary_message] + recent_messages.

    Strategy: sliding window + LLM summary
      - Split: old = messages[:-MESSAGES_TO_KEEP], recent = messages[-MESSAGES_TO_KEEP:]
      - Summarise old messages (via LLM if available, otherwise a count-based fallback)
      - Return [SystemMessage(summary)] + recent

    Why SystemMessage for the summary?
      SystemMessage sits outside the user/assistant turn structure. Gemini
      treats it as background context rather than as part of the dialogue,
      preventing the agent from being confused by a summary block mid-conversation.

    Args:
        messages: Current state.messages list.
        llm:      Optional bound LLM for AI-generated summaries.
                  Pass None (or omit) to use the plain fallback summary.

    Returns:
        Compacted message list. Input is returned unchanged if no compaction needed.
    """
    if not needs_compaction(messages):
        return messages

    split_at     = len(messages) - MESSAGES_TO_KEEP
    old_messages = messages[:split_at]
    recent       = messages[split_at:]

    logger.info(f"[COMPACT] Compacting {len(old_messages)} → summary. Keeping {len(recent)} recent.")

    if llm is not None:
        summary_text = _llm_summary(old_messages, llm)
    else:
        summary_text = _fallback_summary(old_messages)

    summary_msg = SystemMessage(
        content=f"[CONVERSATION SUMMARY — earlier messages compacted]\n\n{summary_text}"
    )
    compacted = [summary_msg] + list(recent)
    logger.info(f"[COMPACT] {len(messages)} messages → {len(compacted)}")
    return compacted


def _llm_summary(messages: list, llm) -> str:
    """Ask the LLM to write a concise summary of the old messages."""
    lines = []
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content[:300]}")
    transcript = "\n".join(lines)

    prompt = (
        "Summarise this task-management conversation in 3–5 sentences. "
        "Cover: which tasks were discussed, what actions were taken (added/completed/deleted), "
        "current state of relevant tasks, and any important context.\n\n"
        f"CONVERSATION:\n{transcript}"
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        logger.info("[COMPACT] AI-generated summary created.")
        return text
    except Exception as e:
        logger.warning(f"[COMPACT] LLM summary failed ({e}), using fallback.")
        return _fallback_summary(messages)


def _fallback_summary(messages: list) -> str:
    """Plain count-based summary used when the LLM is unavailable."""
    hu = sum(1 for m in messages if isinstance(m, HumanMessage))
    ai = sum(1 for m in messages if isinstance(m, AIMessage))
    return (
        f"[Earlier conversation compacted: {hu} user message(s) and {ai} assistant "
        f"message(s) relating to task management operations.]"
    )


# ── Inspection utilities ───────────────────────────────────────────────────────
# AUDIT FIX: replaced checkpointer.conn with direct sqlite3.connect().
# checkpointer.conn is an internal attribute — not part of the public API and
# not guaranteed to exist in every version of langgraph-checkpoint-sqlite.
# Direct sqlite3.connect() is version-safe and requires no internal knowledge.

def get_checkpoint_stats(_checkpointer=None) -> dict:
    """
    Return summary statistics about the checkpoint database.

    The _checkpointer argument is accepted for API compatibility but unused —
    we query the DB file directly so callers don't need to pass the saver.
    """
    if not os.path.exists(CHECKPOINT_DB_PATH):
        return {
            "total_sessions": 0, "total_checkpoints": 0,
            "db_path": CHECKPOINT_DB_PATH, "db_size_kb": 0,
        }
    try:
        conn = sqlite3.connect(CHECKPOINT_DB_PATH)
        try:
            total_cp = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT thread_id) FROM checkpoints"
            ).fetchone()[0]
        finally:
            conn.close()
        size_kb = round(os.path.getsize(CHECKPOINT_DB_PATH) / 1024, 2)
        return {
            "total_sessions": sessions, "total_checkpoints": total_cp,
            "db_path": CHECKPOINT_DB_PATH, "db_size_kb": size_kb,
        }
    except Exception as e:
        return {"error": str(e)}


def list_checkpointed_sessions(_checkpointer=None) -> list[dict]:
    """
    Return all session thread_ids stored in the checkpoint DB.

    The _checkpointer argument is accepted for API compatibility but unused.
    """
    if not os.path.exists(CHECKPOINT_DB_PATH):
        return []
    try:
        conn = sqlite3.connect(CHECKPOINT_DB_PATH)
        try:
            rows = conn.execute(
                "SELECT DISTINCT thread_id, MAX(checkpoint_id) as latest "
                "FROM checkpoints GROUP BY thread_id ORDER BY latest DESC LIMIT 50"
            ).fetchall()
        finally:
            conn.close()
        return [{"thread_id": r[0], "latest_checkpoint": r[1]} for r in rows]
    except Exception as e:
        logger.warning(f"[CHECKPOINT] list_sessions failed: {e}")
        return []


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage

    print("=" * 60)
    print("CHECKPOINTER + COMPACTION SMOKE TEST")
    print("=" * 60)

    print("\n[1] Initialising SqliteSaver ...")
    cp = get_checkpointer()
    print(f"    DB path : {CHECKPOINT_DB_PATH}")
    print(f"    Exists  : {os.path.exists(CHECKPOINT_DB_PATH)}")
    print(f"    Type    : {type(cp).__name__}")

    print("\n[2] Building 30-message history ...")
    msgs = []
    for i in range(15):
        msgs.append(HumanMessage(content=f"User turn {i+1}"))
        msgs.append(AIMessage(content=f"Agent turn {i+1}"))
    print(f"    Messages before compaction : {len(msgs)}")
    print(f"    needs_compaction           : {needs_compaction(msgs)}")

    print("\n[3] Compacting (fallback summary, no LLM) ...")
    compacted = compact_messages(msgs, llm=None)
    print(f"    Messages after compaction  : {len(compacted)}")
    print(f"    First message type         : {type(compacted[0]).__name__}")
    print(f"    Summary preview            : {compacted[0].content[:100]}...")

    print("\n[4] Checkpoint stats ...")
    stats = get_checkpoint_stats()
    print(f"    Stats : {stats}")

    print("\n✅ Checkpointer smoke test complete.")
