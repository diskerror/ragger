"""
HTTP server for Ragger Memory
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

from .memory import RaggerMemory
from .auth import load_token, validate_token

logger = logging.getLogger(__name__)
# Dedicated HTTP log (request/response details)
http_logger = logging.getLogger('ragger_memory.http')

from .config import DEFAULT_HOST, DEFAULT_PORT
from . import lang

# Module-level reference so the handler can access it
_memory = None
_server_token = None  # None = auth disabled (backward compat)


class RaggerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for memory operations"""

    def _check_auth(self) -> bool:
        """Check bearer token if auth is enabled. Returns True if authorized."""
        if _server_token is None:
            return True  # Auth disabled
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return validate_token(auth[7:], _server_token)
        return False

    def _read_body(self) -> dict:
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))
    
    def _respond(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def do_POST(self):
        if not self._check_auth():
            self._respond(401, {"error": "unauthorized"})
            return
        try:
            params = self._read_body()
            
            if self.path == '/store':
                text = params.get('text')
                if not text:
                    self._respond(400, {"error": "text required"})
                    return
                metadata = params.get('metadata') or {}
                # Default collection to "memory" if not specified
                if 'collection' not in metadata:
                    metadata['collection'] = 'memory'
                memory_id = _memory.store(text, metadata)
                self._respond(200, {"id": memory_id, "status": "stored"})
            
            elif self.path == '/search':
                query = params.get('query')
                if not query:
                    self._respond(400, {"error": "query required"})
                    return
                limit = params.get('limit', 5)
                min_score = params.get('min_score', 0.0)
                collections = params.get('collections', None)
                search_result = _memory.search(query, limit, min_score, collections)
                results = search_result["results"]
                timing = search_result.get("timing", {})
                # Convert datetime to string for JSON (if not already converted by backend)
                for r in results:
                    ts = r.get('timestamp')
                    if ts and hasattr(ts, 'isoformat'):
                        r['timestamp'] = ts.isoformat()
                self._respond(200, {"results": results, "timing": timing})
            
            elif self.path == '/count':
                self._respond(200, {"count": _memory.count()})
            
            else:
                self._respond(404, {"error": f"unknown endpoint: {self.path}"})
        
        except Exception as e:
            logger.error(lang.ERR_REQUEST.format(error=e))
            self._respond(500, {"error": str(e)})
    
    def do_GET(self):
        # /health is always public (for port detection, monitoring)
        if self.path == '/health':
            self._respond(200, {"status": "ok", "memories": _memory.count()})
            return
        if not self._check_auth():
            self._respond(401, {"error": "unauthorized"})
            return
        if self.path == '/count':
            self._respond(200, {"count": _memory.count()})
        else:
            self._respond(404, {"error": f"unknown endpoint: {self.path}"})
    
    def log_message(self, format, *args):
        """Route HTTP logs through our logger instead of stderr"""
        http_logger.info(f"{self.address_string()} {format % args}")


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """Start the HTTP server"""
    import socket
    import sys

    # Check if port is available before loading model
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        test_sock.bind((host, port))
    except OSError:
        print(lang.ERR_PORT_IN_USE.format(port=port), file=sys.stderr)
        sys.exit(1)
    finally:
        test_sock.close()

    global _memory, _server_token
    
    # Load auth token if available (None = auth disabled)
    _server_token = load_token()
    if _server_token:
        print("Auth: bearer token required (loaded from ~/.ragger/token)")
    else:
        print("Auth: disabled (no token file found)")
    
    _memory = RaggerMemory()
    
    server = HTTPServer((host, port), RaggerHandler)
    print(lang.MSG_SERVER_RUNNING.format(host=host, port=port))
    print(lang.MSG_SERVER_ENDPOINTS)
    print(f"  POST /store   - {{\"text\": \"...\", \"metadata\": {{...}}}}")
    print(f"  POST /search  - {{\"query\": \"...\", \"limit\": 5}}")
    print(f"  GET  /count   - memory count")
    print(f"  GET  /health  - health check")
    print(lang.MSG_SERVER_STOP)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
        _memory.close()
