"""
HTTP server for Ragger Memory
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

from .memory import RaggerMemory
from .auth import load_token, validate_token, hash_token, ensure_token, token_path

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

    def _check_auth(self) -> dict | None:
        """
        Check bearer token and resolve to user.
        
        Returns user dict {"id": ..., "username": ..., "is_admin": ...}
        or None if auth fails. Returns a default user dict if auth is disabled.
        """
        if _server_token is None:
            return {"id": None, "username": "anonymous", "is_admin": True}
        
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        
        token = auth[7:]
        
        # First try DB lookup by token hash
        if _memory and hasattr(_memory, '_backend'):
            user = _memory._backend.get_user_by_token_hash(hash_token(token))
            if user:
                return user
        
        # Fallback: direct token comparison (pre-bootstrap requests)
        # Auto-create user in DB so future lookups use the fast path
        if validate_token(token, _server_token):
            if _memory and hasattr(_memory, '_backend'):
                import getpass
                hashed = hash_token(token)
                username = getpass.getuser()
                try:
                    user_id = _memory._backend.create_user(username, hashed, is_admin=True)
                    return {"id": user_id, "username": username, "is_admin": True}
                except Exception:
                    pass  # user may already exist (race condition)
            return {"id": None, "username": "default", "is_admin": True}
        
        return None

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
                common = params.get('common', False)
                memory_id = _memory.store(text, metadata, common=common)
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
            
            elif self.path == '/delete_batch':
                memory_ids = params.get('ids', [])
                if not memory_ids:
                    self._respond(400, {"error": "ids required"})
                    return
                count = _memory.delete_batch(memory_ids)
                self._respond(200, {"deleted": count})
            
            elif self.path == '/search_by_metadata':
                metadata_filter = params.get('metadata', {})
                if not metadata_filter:
                    self._respond(400, {"error": "metadata filter required"})
                    return
                limit = params.get('limit', None)
                results = _memory.search_by_metadata(metadata_filter, limit)
                self._respond(200, {"results": results, "count": len(results)})
            
            else:
                self._respond(404, {"error": f"unknown endpoint: {self.path}"})
        
        except Exception as e:
            logger.error(lang.ERR_REQUEST.format(error=e))
            self._respond(500, {"error": str(e)})
    
    def do_DELETE(self):
        if not self._check_auth():
            self._respond(401, {"error": "unauthorized"})
            return
        try:
            # DELETE /memory/<id> — delete by ID
            if self.path.startswith('/memory/'):
                memory_id = self.path.split('/')[-1]
                deleted = _memory.delete(memory_id)
                if deleted:
                    self._respond(200, {"status": "deleted", "id": memory_id})
                else:
                    self._respond(404, {"error": "memory not found"})
            else:
                self._respond(404, {"error": f"unknown endpoint: {self.path}"})
        except Exception as e:
            logger.error(lang.ERR_REQUEST.format(error=e))
            self._respond(500, {"error": str(e)})
    
    def do_GET(self):
        # /health is always public (for port detection, monitoring)
        if self.path == '/health':
            from . import __version__
            self._respond(200, {"status": "ok", "version": __version__, "memories": _memory.count()})
            return
        if not self._check_auth():
            self._respond(401, {"error": "unauthorized"})
            return
        if self.path == '/count':
            response = {"count": _memory.count()}
            if _memory.is_multi_db:
                response["user"] = _memory._user_backend.count()
                response["common"] = _memory._backend.count()
            self._respond(200, response)
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
    
    # Ensure auth token exists (create if needed)
    _server_token = ensure_token()
    print(f"Auth: bearer token required (token: {token_path()})")
    
    from .config import get_config
    cfg = get_config()
    if cfg.get("single_user", True):
        _memory = RaggerMemory()
    else:
        # Multi-user: common DB (shared) + user DB (private)
        import os
        common_path = os.path.expanduser(cfg["common_db_path"])
        user_path = os.path.expanduser(cfg["db_path"])
        _memory = RaggerMemory(uri=common_path, user_db_path=user_path)
        print(f"Multi-user mode: common={common_path}, user={user_path}")

    # Bootstrap default user in single-user mode
    if _server_token and hasattr(_memory, '_backend'):
        backend = _memory._backend
        hashed = hash_token(_server_token)
        existing = backend.get_user_by_token_hash(hashed)
        if not existing:
            import getpass
            username = getpass.getuser()
            user_id = backend.create_user(username, hashed, is_admin=True)
            print(f"Created default user: {username} (id={user_id})")
        else:
            print(f"User: {existing['username']} (id={existing['id']})")
    
    server = HTTPServer((host, port), RaggerHandler)
    print(lang.MSG_SERVER_RUNNING.format(host=host, port=port))
    print(lang.MSG_SERVER_ENDPOINTS)
    print(f"  POST   /store               - {{\"text\": \"...\", \"metadata\": {{...}}}}")
    print(f"  POST   /search              - {{\"query\": \"...\", \"limit\": 5}}")
    print(f"  POST   /search_by_metadata  - {{\"metadata\": {{\"category\": \"...\"}}, \"limit\": 10}}")
    print(f"  POST   /delete_batch        - {{\"ids\": [\"1\", \"2\"]}}")
    print(f"  DELETE /memory/<id>         - delete by ID")
    print(f"  GET    /count               - memory count")
    print(f"  GET    /health              - health check")
    print(lang.MSG_SERVER_STOP)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
        _memory.close()
