"""
mcp_server.py
=============
ASSIGNMENT: Task 4 — MCP Tool Discovery

PURPOSE:
    Exposes the task management tools over the Model Context Protocol (MCP).
    Any MCP-compatible client can connect and discover/call the tools without
    knowing anything about the internal implementation.

AUDIT FIXES APPLIED:
    BUG 1: FastMCP 0.4.1 constructor does NOT accept 'version' or 'description'
           keyword arguments. Those kwargs were added in later versions.
           FastMCP(name="...", version="...", description="...") raises:
             TypeError: FastMCP.__init__() got an unexpected keyword argument 'version'
           Fix: only pass 'name', which is the sole supported kwarg in 0.4.1.

    BUG 2: @mcp.custom_route("/health", methods=["GET"]) does not exist in
           fastmcp 0.4.1. custom_route was introduced in fastmcp 2.x (2025).
           Calling it on a 0.4.1 FastMCP instance raises:
             AttributeError: 'FastMCP' object has no attribute 'custom_route'
           Fix: health check is a standard @mcp.tool() with 'health_check' name.
           For HTTP testing, a standalone FastAPI app bridges to MCP tools.

    BUG 3: mcp.asgi_app() does not exist in fastmcp 0.4.1. asgi_app() was
           also introduced in fastmcp 2.x. Calling it raises:
             AttributeError: 'FastMCP' object has no attribute 'asgi_app'
           fastmcp 0.4.1 only supports stdio transport via mcp.run().
           Fix: provide TWO entry points:
             - stdio mode:  python app/mcp_server.py          (MCP protocol, default)
             - http mode:   python app/mcp_server.py --http   (FastAPI bridge on :8001)
           The HTTP bridge wraps the MCP tool functions in standard FastAPI
           endpoints (/health, /tools, /call) for curl testing and Swagger UI.

HOW TO RUN:
    # stdio mode (default — for MCP clients / Claude Desktop)
    python app/mcp_server.py

    # HTTP bridge mode (for curl testing)
    python app/mcp_server.py --http
    curl http://localhost:8001/health
    curl http://localhost:8001/tools

DEPENDENCIES:
    fastmcp==0.4.1   (stdio MCP protocol)
    fastapi          (HTTP bridge, already in requirements.txt)
    uvicorn          (HTTP bridge server, already in requirements.txt)
    app.tools        (all four task tools + DESTRUCTIVE_TOOLS)
"""

import json
import logging
import sys
from typing import Optional

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError("fastmcp not installed. Run: pip install fastmcp==0.4.1")

from tools import (DESTRUCTIVE_TOOLS, ALL_TOOLS,
                       add_task  as _add,
                       view_tasks as _view,
                       complete_task as _complete,
                       delete_task as _delete,
                       _load_tasks)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── MCP server instance ────────────────────────────────────────────────────────
# AUDIT FIX: FastMCP 0.4.1 only accepts 'name'. 'version' and 'description'
# did not exist until 2.x. Removed those kwargs.

mcp = FastMCP(name="smart-task-manager")


# ── MCP tool definitions ───────────────────────────────────────────────────────
# Each @mcp.tool() function is independently discoverable by MCP clients.
# We call the LangChain @tool wrappers internally and serialise results to JSON.

@mcp.tool()
def add_task(title: str) -> str:
    """
    Add a new task to the task management system.

    Args:
        title: Task description. Example: "Buy groceries", "Call dentist at 3pm"

    Returns:
        JSON: status, message, task (id, title, completed, created_at), total_tasks
    """
    logger.info(f"[MCP] add_task | title='{title}'")
    return json.dumps(_add.invoke({"title": title}), indent=2, default=str)


@mcp.tool()
def view_tasks(filter_status: Optional[str] = None) -> str:
    """
    List all tasks, optionally filtered by completion status.

    Args:
        filter_status: "pending" | "completed" | None (show all)

    Returns:
        JSON: status, tasks[], total, pending, completed, filter_applied
    """
    logger.info(f"[MCP] view_tasks | filter='{filter_status}'")
    return json.dumps(_view.invoke({"filter_status": filter_status}), indent=2, default=str)


@mcp.tool()
def complete_task(task_id: str) -> str:
    """
    Mark a task as completed. Idempotent — safe to call on an already-completed task.

    Args:
        task_id: Task unique ID (format: task_XXXXXXXX). Call view_tasks to find IDs.

    Returns:
        JSON: status ("success"|"info"|"error"), message, task
    """
    logger.info(f"[MCP] complete_task | task_id='{task_id}'")
    return json.dumps(_complete.invoke({"task_id": task_id}), indent=2, default=str)


@mcp.tool()
def delete_task(task_id: str) -> str:
    """
    Permanently delete a task. ⚠️ Destructive — cannot be undone.
    When called via the LangGraph agent this triggers HITL approval.
    Via direct MCP call it executes immediately.

    Args:
        task_id: Task unique ID (format: task_XXXXXXXX). Call view_tasks to find IDs.

    Returns:
        JSON: status, message, deleted_task, remaining_tasks
    """
    logger.info(f"[MCP] delete_task | task_id='{task_id}' | DESTRUCTIVE")
    return json.dumps(_delete.invoke({"task_id": task_id}), indent=2, default=str)


@mcp.tool()
def list_tool_metadata() -> str:
    """
    MCP Tool Discovery — return full metadata for all available tools.

    This is the discovery endpoint: it tells MCP clients what each tool does,
    its input schema, whether it is destructive (HITL required), and usage examples.
    This function directly satisfies Task 4: MCP Tool Discovery.

    Returns:
        JSON: server info, total_tools, destructive_tools[], tools[] with full metadata
    """
    logger.info("[MCP] list_tool_metadata — Tool Discovery request")
    metadata = [
        {
            "name": "add_task",
            "description": "Create a new task with a title",
            "destructive": False, "hitl_required": False,
            "input_schema": {"title": "string (required)"},
            "example": {"title": "Buy groceries"},
        },
        {
            "name": "view_tasks",
            "description": "List tasks with optional status filter",
            "destructive": False, "hitl_required": False,
            "input_schema": {"filter_status": "string (optional): 'pending'|'completed'|null"},
            "example": {"filter_status": "pending"},
        },
        {
            "name": "complete_task",
            "description": "Mark a task done by its ID",
            "destructive": False, "hitl_required": False,
            "input_schema": {"task_id": "string (required): task_XXXXXXXX"},
            "example": {"task_id": "task_a1b2c3d4"},
        },
        {
            "name": "delete_task",
            "description": "Permanently remove a task by its ID",
            "destructive": True, "hitl_required": True,
            "input_schema": {"task_id": "string (required): task_XXXXXXXX"},
            "example": {"task_id": "task_a1b2c3d4"},
            "warning": "DESTRUCTIVE — cannot be undone. HITL approval required in agent mode.",
        },
    ]
    return json.dumps({
        "server": "smart-task-manager", "version": "1.0.0",
        "total_tools": len(metadata),
        "destructive_tools": list(DESTRUCTIVE_TOOLS),
        "tools": metadata,
    }, indent=2)


# ── Health check (as MCP tool — replaces the broken @mcp.custom_route) ────────
# AUDIT FIX: custom_route does not exist in fastmcp 0.4.1.
# Exposed as a standard MCP tool and also available via the HTTP bridge below.

@mcp.tool()
def health_check() -> str:
    """
    Server health check. Returns status, version, task count, and available tools.

    Returns:
        JSON: status, server, version, tasks_count, mcp_tools[]
    """
    return json.dumps({
        "status": "healthy", "server": "smart-task-manager", "version": "1.0.0",
        "tasks_count": len(_load_tasks()),
        "mcp_tools": [t.name for t in ALL_TOOLS] + ["list_tool_metadata", "health_check"],
    }, indent=2)


# ── HTTP bridge (replaces the broken mcp.asgi_app()) ──────────────────────────
# AUDIT FIX: mcp.asgi_app() does not exist in fastmcp 0.4.1.
# fastmcp 0.4.1 is a stdio-only MCP server. For HTTP access during development
# and for the Render/HuggingFace health-check endpoint, we wrap the MCP tool
# functions in a standard FastAPI app. This gives identical functionality.

def create_http_bridge(port: int = 8001):
    """
    Create a FastAPI app that bridges HTTP → MCP tool functions.

    Endpoints:
      GET  /health   — server health check
      GET  /tools    — list all tool schemas (same as list_tool_metadata)
      POST /call     — call any tool by name: {"name": "add_task", "args": {...}}
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn

    bridge = FastAPI(
        title="Smart Task Manager — MCP HTTP Bridge",
        description=(
            "HTTP bridge for fastmcp 0.4.1 stdio server.\n\n"
            "Provides /health, /tools, and /call endpoints for curl testing.\n"
            "For production MCP clients use the stdio transport (default mode)."
        ),
        version="1.0.0",
    )

    _TOOL_FNS = {
        "add_task":          lambda args: add_task(**args),
        "view_tasks":        lambda args: view_tasks(**args),
        "complete_task":     lambda args: complete_task(**args),
        "delete_task":       lambda args: delete_task(**args),
        "list_tool_metadata": lambda args: list_tool_metadata(),
        "health_check":      lambda args: health_check(),
    }

    @bridge.get("/health", tags=["General"])
    async def http_health():
        return json.loads(health_check())

    @bridge.get("/tools", tags=["MCP"])
    async def http_tools():
        return json.loads(list_tool_metadata())

    @bridge.post("/call", tags=["MCP"])
    async def http_call(body: dict):
        """
        Call an MCP tool by name.
        Body: {"name": "tool_name", "args": {"arg1": "val1", ...}}
        """
        name = body.get("name", "")
        args = body.get("args", {})
        fn = _TOOL_FNS.get(name)
        if fn is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown tool '{name}'. Available: {list(_TOOL_FNS.keys())}"
            )
        try:
            result = fn(args)
            return JSONResponse(content=json.loads(result))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    print("=" * 60)
    print(f"MCP HTTP BRIDGE — Smart Task Manager  (port {port})")
    print("=" * 60)
    print(f"  GET  http://localhost:{port}/health")
    print(f"  GET  http://localhost:{port}/tools")
    print(f"  POST http://localhost:{port}/call")
    print(f"  Docs http://localhost:{port}/docs")
    print()
    print("Test:")
    print(f"  curl http://localhost:{port}/health")
    print(f"  curl http://localhost:{port}/tools")
    print()
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    uvicorn.run(bridge, host="0.0.0.0", port=port, log_level="info")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--http" in sys.argv:
        # HTTP bridge mode — for curl testing and Swagger UI
        create_http_bridge(port=8001)
    else:
        # stdio mode — standard MCP protocol for MCP clients / Claude Desktop
        print("Starting MCP server (stdio transport)...")
        print("Use --http flag for HTTP bridge mode:")
        print("  python app/mcp_server.py --http")
        mcp.run()
