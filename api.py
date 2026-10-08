"""
api.py — Task 9: FastAPI REST Endpoints
Fixed for flat project structure (no app/ folder).
All imports changed from 'from app.X import' to 'from X import'.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from checkpointer import get_checkpoint_stats, list_checkpointed_sessions
from graph import build_graph, create_initial_state, get_production_graph
from observability import flush_langfuse, trace_session_end, trace_session_start
from tools import _load_tasks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_graph        = None
_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _checkpointer
    logger.info("[API] Starting up — building production graph ...")
    _graph, _checkpointer = get_production_graph()
    logger.info("[API] Ready.")
    yield
    logger.info("[API] Shutting down — flushing Langfuse ...")
    flush_langfuse()


app = FastAPI(
    title="Smart Task Management Agent",
    description=(
        "Production-grade AI agent: LangGraph + Gemini + MCP + Langfuse + FastAPI.\n\n"
        "Supports multi-turn sessions, Human-in-the-Loop approval, persistent "
        "checkpointing, and full Langfuse observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    model_config = {"json_schema_extra": {"example": {"message": "Add task: Buy groceries"}}}


class ChatResponse(BaseModel):
    response:       str
    session_id:     str
    duration_ms:    float
    hook_log_count: int = 0


class SessionCreated(BaseModel):
    session_id: str
    message:    str


def _cfg(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _extract_response(messages: list) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai":
            c = msg.content
            if isinstance(c, str) and c.strip():
                return c
            if isinstance(c, list):
                t = " ".join(b.get("text", "") for b in c if isinstance(b, dict)).strip()
                if t:
                    return t
    return "I couldn't process that request."


@app.get("/", tags=["General"])
async def root():
    return {
        "name": "Smart Task Management Agent", "version": "1.0.0", "status": "running",
        "docs": "/docs",
        "endpoints": {
            "stateless_chat":   "POST /chat",
            "new_session":      "POST /sessions",
            "session_chat":     "POST /sessions/{id}/chat",
            "approve":          "POST /sessions/{id}/approve",
            "reject":           "POST /sessions/{id}/reject",
            "session_status":   "GET  /sessions/{id}/status",
            "all_sessions":     "GET  /sessions",
            "checkpoint_stats": "GET  /checkpoints/stats",
            "health":           "GET  /health",
        },
    }


@app.get("/health", tags=["General"])
async def health():
    return {
        "status": "healthy", "version": "1.0.0",
        "agent_ready": _graph is not None,
        "tasks_count": len(_load_tasks()),
        "checkpointer": "sqlite" if _checkpointer else "none",
        "checkpoint_stats": get_checkpoint_stats() if _checkpointer else {},
    }


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    t0  = time.perf_counter_ns()
    trace_session_start(sid, req.message, {"endpoint": "/chat"})
    try:
        stateless = build_graph(checkpointer=None, with_hitl=False)
        state = create_initial_state(sid)
        state["messages"] = [HumanMessage(content=req.message)]
        final = stateless.invoke(state)
        resp     = _extract_response(final["messages"])
        duration = round((time.perf_counter_ns() - t0) / 1_000_000, 2)
        trace_session_end(sid, resp, duration)
        return ChatResponse(response=resp, session_id=sid, duration_ms=duration,
                            hook_log_count=len(final.get("hook_log", [])))
    except Exception as e:
        duration = round((time.perf_counter_ns() - t0) / 1_000_000, 2)
        trace_session_end(sid, "", duration, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions", response_model=SessionCreated, tags=["Sessions"])
async def create_session():
    sid = str(uuid.uuid4())
    logger.info(f"[API] New session: {sid}")
    return SessionCreated(session_id=sid,
                          message=f"Session {sid} created. POST /sessions/{sid}/chat to start.")


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse, tags=["Sessions"])
async def session_chat(session_id: str, req: ChatRequest):
    if not _graph:
        raise HTTPException(503, "Agent not ready")
    t0  = time.perf_counter_ns()
    cfg = _cfg(session_id)
    trace_session_start(session_id, req.message, {"endpoint": "/sessions/chat"})
    try:
        existing = _graph.get_state(cfg)
        if existing and existing.values:
            _graph.update_state(cfg, {"messages": [HumanMessage(content=req.message)]})
            final = _graph.invoke(None, config=cfg)
        else:
            state = create_initial_state(session_id)
            state["messages"] = [HumanMessage(content=req.message)]
            final = _graph.invoke(state, config=cfg)
        cur     = _graph.get_state(cfg)
        paused  = bool(cur and cur.next)
        pending = final.get("pending_action")
        if paused and pending:
            resp = (
                f"⚠️ Approval needed: the agent wants to '{pending.get('action')}' "
                f"with args {pending.get('args', {})}.\n"
                f"POST /sessions/{session_id}/approve  or  /sessions/{session_id}/reject"
            )
        else:
            resp = _extract_response(final.get("messages", []))
        duration = round((time.perf_counter_ns() - t0) / 1_000_000, 2)
        trace_session_end(session_id, resp, duration)
        return ChatResponse(response=resp, session_id=session_id, duration_ms=duration,
                            hook_log_count=len(final.get("hook_log", [])))
    except Exception as e:
        duration = round((time.perf_counter_ns() - t0) / 1_000_000, 2)
        trace_session_end(session_id, "", duration, success=False, error=str(e))
        raise HTTPException(500, str(e))


@app.post("/sessions/{session_id}/approve", tags=["HITL"])
async def approve(session_id: str):
    if not _graph:
        raise HTTPException(503, "Agent not ready")
    try:
        _graph.update_state(_cfg(session_id), {"hitl_approved": True})
        final = _graph.invoke(None, config=_cfg(session_id))
        return {"approved": True, "response": _extract_response(final.get("messages", [])),
                "session_id": session_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/sessions/{session_id}/reject", tags=["HITL"])
async def reject(session_id: str):
    if not _graph:
        raise HTTPException(503, "Agent not ready")
    try:
        _graph.update_state(_cfg(session_id), {"hitl_approved": False})
        final = _graph.invoke(None, config=_cfg(session_id))
        return {"approved": False, "response": _extract_response(final.get("messages", [])),
                "session_id": session_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/sessions/{session_id}/status", tags=["Sessions"])
async def session_status(session_id: str):
    if not _graph:
        raise HTTPException(503, "Agent not ready")
    snap = _graph.get_state(_cfg(session_id))
    if not snap or not snap.values:
        raise HTTPException(404, f"Session {session_id} not found")
    v = snap.values
    return {
        "session_id":     session_id,
        "message_count":  len(v.get("messages", [])),
        "hook_log_count": len(v.get("hook_log", [])),
        "pending_action": v.get("pending_action"),
        "hitl_approved":  v.get("hitl_approved"),
        "error":          v.get("error"),
        "next_nodes":     list(snap.next) if snap.next else [],
        "hook_log_tail":  v.get("hook_log", [])[-5:],
    }


@app.get("/sessions", tags=["Sessions"])
async def list_sessions():
    if not _checkpointer:
        return {"sessions": [], "message": "Checkpointer not available"}
    sessions = list_checkpointed_sessions()
    return {"sessions": sessions, "total": len(sessions)}


@app.get("/checkpoints/stats", tags=["Checkpointing"])
async def checkpoint_stats():
    if not _checkpointer:
        return {"error": "Checkpointer not available"}
    return get_checkpoint_stats()


@app.get("/tasks", tags=["Tasks"])
async def list_tasks():
    tasks = _load_tasks()
    completed = sum(1 for t in tasks if t["completed"])
    return {"tasks": tasks, "total": len(tasks),
            "completed": completed, "pending": len(tasks) - completed}


if __name__ == "__main__":
    import uvicorn
    print("Starting Smart Task Management Agent API ...")
    print("  API  : http://localhost:8000")
    print("  Docs : http://localhost:8000/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")