"""
MCP JSON-RPC server for OpenClaw integration
"""

import sys
import json
import logging

from .memory import RaggerMemory

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
                if not query:
                    raise ValueError("query parameter required")
                results = memory.search(query, limit, min_score)
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
    
    # Main loop: read JSON-RPC requests from stdin
    logger.info("MCP server started, waiting for requests...")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            handle_request(request)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
    
    memory.close()
