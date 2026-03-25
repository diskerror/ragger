"""
HTTP server for Ragger Memory
"""

import json
import logging
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

from .memory import RaggerMemory
from .auth import load_token, validate_token, hash_token, ensure_token, token_path
from .inference import InferenceClient

logger = logging.getLogger(__name__)
# Dedicated HTTP log (request/response details)
http_logger = logging.getLogger('ragger_memory.http')

from .config import DEFAULT_HOST, DEFAULT_PORT
from . import lang

# Module-level reference so the handler can access it
_memory = None
_server_token = None  # None = auth disabled (backward compat)
_inference_client = None  # Initialized if inference config is present


class RaggerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for memory operations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rotation_needed = False
        self._rotation_username = None

    def _check_auth(self) -> dict | None:
        """
        Check bearer token and resolve to user.
        
        Returns user dict {"id": ..., "username": ..., "is_admin": ...}
        or None if auth fails. Returns a default user dict if auth is disabled.
        
        Side effect: Sets self._rotation_needed and self._rotation_username
        if token rotation should happen after this request.
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
                # Check if token rotation is needed
                self._check_token_rotation(user["username"])
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

    def _check_token_rotation(self, username: str):
        """
        Check if token rotation is needed for this user.
        Sets self._rotation_needed and self._rotation_username if rotation should happen.
        """
        from .config import get_config
        from datetime import datetime, timezone, timedelta
        
        cfg = get_config()
        rotation_minutes = cfg.get("token_rotation_minutes", 1440)
        
        # 0 = disabled
        if rotation_minutes <= 0:
            return
        
        backend = _memory._backend
        rotated_at_str = backend.get_user_token_rotated_at(username)
        
        # No rotation timestamp yet (shouldn't happen after migration, but handle it)
        if not rotated_at_str:
            now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            backend.update_user_token_rotated_at(username, now)
            return
        
        # Parse timestamp
        try:
            rotated_at = datetime.fromisoformat(rotated_at_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return
        
        now = datetime.now(timezone.utc)
        age_minutes = (now - rotated_at).total_seconds() / 60
        
        # Token is expired
        if age_minutes > rotation_minutes:
            # Grace window: don't rotate if we just rotated within last 60 seconds
            # (prevents re-rotation before client picks up new token)
            grace_seconds = 60
            if age_minutes * 60 < rotation_minutes * 60 + grace_seconds:
                return
            
            # Mark for rotation after this request
            self._rotation_needed = True
            self._rotation_username = username

    def _perform_rotation(self):
        """Rotate the token if needed (called after response is sent)."""
        if not hasattr(self, '_rotation_needed') or not hasattr(self, '_rotation_username'):
            return
        if not self._rotation_needed or not self._rotation_username:
            return
        
        from .auth import rotate_token_for_user
        from datetime import datetime, timezone
        
        try:
            username = self._rotation_username
            new_token, new_hash = rotate_token_for_user(username)
            
            backend = _memory._backend
            backend.update_user_token(username, new_hash)
            
            now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            backend.update_user_token_rotated_at(username, now)
            
            logger.info(f"Rotated token for user: {username}")
        except Exception as e:
            logger.error(f"Token rotation failed for {username}: {e}")

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
        # Perform token rotation if needed (after response is sent)
        self._perform_rotation()
    
    def do_POST(self):
        user = self._check_auth()
        if not user:
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
            
            elif self.path == '/register':
                username = params.get('username')
                if not username:
                    self._respond(400, {"error": "username required"})
                    return
                # Extract the bearer token from the request
                auth_header = self.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    self._respond(401, {"error": "bearer token required"})
                    return
                provided_token = auth_header[7:]
                # Verify: read the user's token file from the filesystem
                import pwd
                try:
                    pw = pwd.getpwnam(username)
                except KeyError:
                    self._respond(400, {"error": f"unknown system user: {username}"})
                    return
                user_token_file = os.path.join(pw.pw_dir, ".ragger", "token")
                if not os.path.exists(user_token_file):
                    self._respond(400, {"error": f"no token file for {username}"})
                    return
                with open(user_token_file) as f:
                    file_token = f.read().strip()
                if not validate_token(provided_token, file_token):
                    self._respond(403, {"error": "token does not match user's token file"})
                    return
                # Register user in DB
                hashed = hash_token(provided_token)
                backend = _memory._backend
                existing = backend.get_user_by_username(username)
                if existing:
                    if existing.get("token_hash") != hashed:
                        backend.update_user_token(username, hashed)
                    self._respond(200, {"status": "exists", "user_id": existing["id"],
                                        "username": username})
                else:
                    is_admin = params.get('is_admin', False)
                    user_id = backend.create_user(username, hashed, is_admin=is_admin)
                    self._respond(200, {"status": "created", "user_id": user_id,
                                        "username": username})

            elif self.path == '/search_by_metadata':
                metadata_filter = params.get('metadata', {})
                if not metadata_filter:
                    self._respond(400, {"error": "metadata filter required"})
                    return
                limit = params.get('limit', None)
                results = _memory.search_by_metadata(metadata_filter, limit)
                self._respond(200, {"results": results, "count": len(results)})
            
            elif self.path == '/user/model':
                # PUT /user/model — set preferred model for authenticated user
                model = params.get('model')
                if not model:
                    self._respond(400, {"error": "model required"})
                    return
                username = user["username"]
                backend = _memory._backend
                backend.update_user_preferred_model(username, model)
                self._respond(200, {"status": "updated", "model": model, "username": username})
            
            elif self.path == '/v1/chat/completions':
                # Inference proxy — respects user's preferred_model
                if not _inference_client:
                    self._respond(503, {"error": "inference not configured"})
                    return
                
                messages = params.get('messages', [])
                if not messages:
                    self._respond(400, {"error": "messages required"})
                    return
                
                # Check user's preferred model, fall back to request model or system default
                username = user["username"]
                backend = _memory._backend
                preferred_model = backend.get_user_preferred_model(username)
                
                request_model = params.get('model')
                use_model = preferred_model or request_model or _inference_client.model
                
                max_tokens = params.get('max_tokens', _inference_client.max_tokens)
                stream = params.get('stream', False)
                
                logger.info(f"Inference request: user={username}, model={use_model}, stream={stream}")
                
                # Proxy to inference endpoint
                try:
                    response = _inference_client.chat(
                        messages=messages,
                        stream=stream,
                        model=use_model,
                        max_tokens=max_tokens
                    )
                    
                    if stream:
                        # Streaming response — forward SSE chunks
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/event-stream')
                        self.send_header('Cache-Control', 'no-cache')
                        self.end_headers()
                        
                        for chunk in response:
                            chunk_json = json.dumps(chunk)
                            self.wfile.write(f"data: {chunk_json}\n\n".encode())
                        
                        self.wfile.write(b"data: [DONE]\n\n")
                        # Perform rotation after streaming completes
                        self._perform_rotation()
                    else:
                        # Non-streaming response
                        self._respond(200, response)
                except Exception as e:
                    logger.error(f"Inference request failed: {e}")
                    self._respond(500, {"error": str(e)})
            
            else:
                self._respond(404, {"error": f"unknown endpoint: {self.path}"})
        
        except Exception as e:
            logger.error(lang.ERR_REQUEST.format(error=e))
            self._respond(500, {"error": str(e)})
    
    def do_DELETE(self):
        user = self._check_auth()
        if not user:
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
            elif self.path == '/user/model':
                # DELETE /user/model — clear preferred model (back to system default)
                username = user["username"]
                backend = _memory._backend
                backend.update_user_preferred_model(username, None)
                self._respond(200, {"status": "cleared", "username": username})
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
        user = self._check_auth()
        if not user:
            self._respond(401, {"error": "unauthorized"})
            return
        if self.path == '/count':
            response = {"count": _memory.count()}
            if _memory.is_multi_db:
                response["user"] = _memory._user_backend.count()
                response["common"] = _memory._backend.count()
            self._respond(200, response)
        elif self.path == '/user/model':
            # GET /user/model — get preferred model for authenticated user
            username = user["username"]
            backend = _memory._backend
            model = backend.get_user_preferred_model(username)
            self._respond(200, {"model": model, "username": username})
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

    global _memory, _server_token, _inference_client
    
    # Ensure auth token exists (create if needed)
    _server_token = ensure_token()
    print(f"Auth: bearer token required (token: {token_path()})")
    
    from .config import get_config
    cfg = get_config()
    
    # Initialize inference client if configured
    if cfg.get("inference_api_url") or cfg.get("inference_endpoints"):
        try:
            _inference_client = InferenceClient.from_config(cfg)
            print(f"Inference: enabled (default model: {_inference_client.model})")
        except Exception as e:
            logger.warning(f"Inference client initialization failed: {e}")
            print("Inference: disabled (no valid configuration)")
    
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
    print(f"  POST   /register            - {{\"username\": \"...\"}}")
    print(f"  POST   /v1/chat/completions - inference proxy (respects user model)")
    print(f"  PUT    /user/model          - set preferred model")
    print(f"  GET    /user/model          - get preferred model")
    print(f"  DELETE /user/model          - clear preferred model")
    print(f"  GET    /health              - health check")
    print(lang.MSG_SERVER_STOP)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
        _memory.close()
