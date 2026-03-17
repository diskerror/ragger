"""
MCP JSON-RPC server for OpenClaw integration
"""

import sys
import json
import logging

from .memory import RaggerMemory

# Dedicated MCP logger (JSON-RPC interactions → mcp.log)
mcp_logger = logging.getLogger('ragger_memory.mcp')

logger = logging.getLogger(__name__)


def run_mcp_server():
    """
    MCP JSON-RPC server
    Reads requests from stdin, writes responses to stdout
    """
    memory = RaggerMemory()
    
    def send_response(response: dict):
        """Send JSON-RPC response to stdout"""
        print(json.dumps(response), flush=True)
    
    def handle_request(request: dict):
        """Handle a single JSON-RPC request"""
        method = request.get('method')
        params = request.get('params', {})
        req_id = request.get('id')
        
        try:
            if method == 'memory_store':
                text = params.get('text')
                metadata = params.get('metadata')
                if not text:
                    raise ValueError("text parameter required")
                memory_id = memory.store(text, metadata)
                result = {"id": memory_id, "status": "stored"}
            
            elif method == 'memory_search':
                query = params.get('query')
                limit = params.get('limit', 5)
                min_score = params.get('min_score', 0.0)
                collections = params.get('collections', None)
                if not query:
                    raise ValueError("query parameter required")
                results = memory.search(query, limit, min_score, collections)
                result = {"results": results}
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
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
                # Truncate long results for readability
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
    
    # Main loop: read requests from stdin
    # Accepts JSON-RPC (MCP) or plain text (search shortcut)
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
                # Looks like JSON but isn't — treat as plain text
                mcp_logger.info(f"plain text query: {line[:100]}")
                handle_plain_text(line)
        else:
            mcp_logger.info(f"plain text query: {line[:100]}")
            handle_plain_text(line)
    
    memory.close()
