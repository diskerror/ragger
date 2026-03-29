"""
MCP-compliant JSON-RPC server for AI agent integration.

Implements the Model Context Protocol (MCP) specification:
https://modelcontextprotocol.io/docs/spec/

Provides two tools:
- store: Store a memory with optional metadata
- search: Search memories by semantic similarity

Also accepts plain text queries as a search shortcut for interactive use.
"""

import sys
import json
import logging

from .memory import RaggerMemory

# Dedicated MCP logger (JSON-RPC interactions → mcp.log)
mcp_logger = logging.getLogger('ragger_memory.mcp')

logger = logging.getLogger(__name__)

SERVER_NAME = "ragger-memory"
SERVER_VERSION = "0.7.0"
PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "store",
        "description": "Store a memory for later semantic retrieval.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text content to store."
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata (category, tags, source, collection, etc.)."
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "search",
        "description": "Search stored memories by semantic similarity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)."
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum similarity score 0-1 (default: 0.0)."
                },
                "collections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by collection names."
                }
            },
            "required": ["query"]
        }
    }
]


def run_mcp_server():
    """
    MCP JSON-RPC server.
    Reads requests from stdin, writes responses to stdout.
    Supports MCP protocol (initialize, tools/list, tools/call)
    and plain text search shortcuts for interactive use.
    """
    from .config import get_config
    cfg = get_config()
    
    if cfg["single_user"]:
        memory = RaggerMemory()
    else:
        # Multi-user MCP: common DB for shared memories, user DB for private
        import os
        user_db = os.path.expanduser("~/.ragger/memories.db")
        memory = RaggerMemory(uri=cfg["common_db_path"], user_db_path=user_db)

    def send_response(response: dict):
        """Send JSON-RPC response to stdout."""
        print(json.dumps(response), flush=True)

    def handle_request(request: dict):
        """Handle a single JSON-RPC request."""
        method = request.get('method')
        params = request.get('params', {})
        req_id = request.get('id')  # None for notifications

        # Notifications (no id) get no response
        if req_id is None:
            mcp_logger.info(f"notification: {method}")
            return

        try:
            if method == 'initialize':
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION
                    }
                }

            elif method == 'tools/list':
                result = {"tools": TOOLS}

            elif method == 'tools/call':
                tool_name = params.get('name')
                arguments = params.get('arguments', {})
                result = _handle_tool_call(memory, tool_name, arguments)

            else:
                send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                })
                return

            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result
            })

        except Exception as e:
            logger.error(f"Error handling {method}: {e}")
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            })

    def _handle_tool_call(memory, tool_name: str, arguments: dict) -> dict:
        """Dispatch a tools/call request. Returns MCP result with content."""
        if tool_name == 'store':
            text = arguments.get('text')
            metadata = arguments.get('metadata')
            if not text:
                return {
                    "content": [{"type": "text", "text": "Error: text parameter required"}],
                    "isError": True
                }
            memory_id = memory.store(text, metadata)
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({"id": memory_id, "status": "stored"})
                }]
            }

        elif tool_name == 'search':
            query = arguments.get('query')
            limit = arguments.get('limit', 5)
            min_score = arguments.get('min_score', 0.0)
            collections = arguments.get('collections', None)
            if not query:
                return {
                    "content": [{"type": "text", "text": "Error: query parameter required"}],
                    "isError": True
                }
            results = memory.search(query, limit, min_score, collections)
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(results)
                }]
            }

        else:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True
            }

    def handle_plain_text(query: str):
        """Handle a plain text query (non-JSON input) as a search."""
        try:
            search_result = memory.search(query)
            results = search_result.get("results", [])
            timing = search_result.get("timing", {})

            if not results:
                print("No results found.", flush=True)
                return

            for i, r in enumerate(results, 1):
                score = r.get("score", 0.0)
                text = r.get("text", "")
                preview = text[:200] + "..." if len(text) > 200 else text
                meta = r.get("metadata", {})
                source = meta.get("source", "")
                collection = meta.get("collection", "")

                header = f"{i}. [score: {score:.3f}]"
                if source:
                    header += f" ({source})"
                if collection:
                    header += f" [{collection}]"
                print(header, flush=True)
                print(f"   {preview}", flush=True)
                print(flush=True)

            corpus = timing.get("corpus_size", "?")
            total_ms = timing.get("total_ms", "?")
            print(f"Timing: {total_ms}ms ({corpus} chunks)", flush=True)

        except Exception as e:
            print(f"Error: {e}", flush=True)

    # Start housekeeping thread if HTTP server isn't handling it
    import getpass
    import threading

    def _is_http_server_running():
        """Check if any ragger HTTP server is running (PID file with live process)."""
        import glob
        import signal as _sig
        for pattern in ["/var/run/ragger-*.pid", "/tmp/ragger-*.pid"]:
            for pid_path in glob.glob(pattern):
                try:
                    with open(pid_path) as f:
                        pid = int(f.read().strip())
                    os.kill(pid, 0)  # check if alive
                    return True
                except (ValueError, ProcessLookupError, PermissionError, FileNotFoundError):
                    continue
        return False

    def _mcp_housekeeping_loop(mem, username):
        from datetime import datetime, timezone, timedelta
        import sqlite3
        max_age_hours = float(cfg.get("cleanup_max_age_hours", 336))
        while True:
            import time
            time.sleep(60)
            if _is_http_server_running():
                continue
            if max_age_hours <= 0:
                continue
            # Clean expired conversations from user DB
            be = mem._user_backend or mem._backend
            db_path = be.db_path
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute(
                    "DELETE FROM memories WHERE collection = 'conversation' AND timestamp < ?",
                    (cutoff_str,)
                )
                deleted = cursor.rowcount
                conn.commit()
                conn.close()
                if deleted > 0:
                    mcp_logger.info(f"MCP housekeeping: cleaned {deleted} expired conversations")
            except Exception as e:
                mcp_logger.warning(f"MCP housekeeping error: {e}")

    mcp_username = getpass.getuser()
    if not _is_http_server_running():
        hk_thread = threading.Thread(
            target=_mcp_housekeeping_loop, args=(memory, mcp_username), daemon=True)
        hk_thread.start()

    # Main loop: read requests from stdin
    mcp_logger.info("MCP server started, waiting for requests...")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        # Try JSON-RPC first; fall back to plain text search
        if line.startswith("{"):
            try:
                request = json.loads(line)
                mcp_logger.info(f"request: {request.get('method', 'unknown')}")
                handle_request(request)
            except json.JSONDecodeError:
                mcp_logger.info(f"plain text query: {line[:100]}")
                handle_plain_text(line)
        else:
            mcp_logger.info(f"plain text query: {line[:100]}")
            handle_plain_text(line)

    memory.close()
