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
from .chat_sessions import (get_or_create_session, load_workspace_files,
                             cleanup_expired_sessions, run_housekeeping)

logger = logging.getLogger(__name__)

# Timelapse file — touched on user activity, mtime signals idle duration
_TIMELAPSE_PATH = "/var/ragger/timelapse"


def _touch_timelapse():
    """Update timelapse file mtime to mark user activity."""
    try:
        with open(_TIMELAPSE_PATH, 'a'):
            os.utime(_TIMELAPSE_PATH, None)
    except OSError:
        pass  # non-fatal — file may not be writable in single-user mode


# Dedicated HTTP log (request/response details)
http_logger = logging.getLogger('ragger_memory.http')

from .config import DEFAULT_HOST, DEFAULT_PORT, get_config
from . import lang

# Module-level reference so the handler can access it
_memory = None
_server_token = None  # None = auth disabled (backward compat)
_inference_client = None  # Initialized if inference config is present
_user_memories = {}  # username → RaggerMemory (per-user cache, multi-user mode)


def _get_memory(username: str = None):
    """
    Get the appropriate RaggerMemory for a request.

    Single-user mode: always returns _memory.
    Multi-user mode: returns a user-scoped view that searches both
    the common DB and the user's private DB (~username/.ragger/memories.db).
    """
    if not username or not _memory:
        return _memory

    cfg = get_config()
    if cfg.get("single_user", True):
        return _memory

    if username not in _user_memories:
        _user_memories[username] = _memory.for_user(username)

    return _user_memories[username]

def _preload_local_model(model_name: str):
    """Preload a model on local inference engines in a background thread."""
    import threading
    if not model_name or not _inference_client:
        return
    try:
        endpoint = _inference_client._resolve_endpoint(model_name)
        url = endpoint.api_url
        if not any(h in url for h in ('localhost', '127.0.0.', '192.168.', '10.', '0.0.0.0')):
            return
    except Exception:
        return

    def _do_preload():
        err = _inference_client.ensure_model_loaded(model_name)
        if err:
            print(f"Model preload skipped: {err}")
        else:
            print(f"Preloaded model: {model_name}")

    threading.Thread(target=_do_preload, daemon=True).start()


# Web session tokens: {token_str: {"username": ..., "expires": timestamp}}
import time
import secrets
_web_sessions = {}


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
        from .config import get_config
        cfg = get_config()
        single_user = cfg.get("single_user", True)
        
        # Single-user mode with no token configured: auth disabled
        if single_user and _server_token is None:
            return {"id": None, "username": "anonymous", "is_admin": True}
        
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            # Check cookie for web session token
            cookie = self.headers.get("Cookie", "")
            for part in cookie.split(";"):
                part = part.strip()
                if part.startswith("ragger_token="):
                    token = part[len("ragger_token="):]
                    try:
                        from urllib.parse import unquote
                        token = unquote(token)
                    except Exception:
                        pass
                    web_user = self._check_web_session(token)
                    if web_user:
                        return web_user
                    # Also try as bearer token
                    if _memory and hasattr(_memory, '_backend'):
                        user = _memory._backend.get_user_by_token_hash(hash_token(token))
                        if user:
                            return user
            return None  # No auth header or cookie → reject
        
        token = auth[7:]
        
        # Check web session tokens first
        web_user = self._check_web_session(token)
        if web_user:
            return web_user
        
        # DB lookup by token hash (works for both modes)
        if _memory and hasattr(_memory, '_backend'):
            user = _memory._backend.get_user_by_token_hash(hash_token(token))
            if user:
                # Check if token rotation is needed
                self._check_token_rotation(user["username"])
                return user
        
        # Fallback: direct token comparison (single-user only, pre-bootstrap)
        if single_user and _server_token and validate_token(token, _server_token):
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
    
    # --- MIME types for static file serving ---
    MIME_TYPES = {
        ".html": "text/html",
        ".css":  "text/css",
        ".js":   "application/javascript",
        ".json": "application/json",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".svg":  "image/svg+xml",
        ".ico":  "image/x-icon",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
    }

    WEB_SESSION_TTL = 86400  # 24 hours

    def _handle_login(self):
        """Handle POST /auth/login — validate password, issue session token."""
        try:
            params = self._read_body()
            username = params.get("username", "").strip()
            password = params.get("password", "")

            if not username or not password:
                self._respond(400, {"error": "username and password required"})
                return

            # Look up user in database
            backend = _memory._backend
            user = backend.get_user_by_username(username)
            if not user:
                self._respond(401, {"error": "invalid credentials"})
                return

            # Verify password
            stored_hash = backend.get_user_password(username)
            if not stored_hash:
                self._respond(401, {"error": "no password set — use 'ragger passwd' first"})
                return

            from .auth import verify_password
            if not verify_password(password, stored_hash):
                self._respond(401, {"error": "invalid credentials"})
                return

            # Generate session token
            session_token = secrets.token_urlsafe(32)
            _web_sessions[session_token] = {
                "username": username,
                "user_id": user.get("id"),
                "is_admin": user.get("is_admin", False),
                "expires": time.time() + self.WEB_SESSION_TTL
            }

            self._respond(200, {
                "token": session_token,
                "username": username,
                "expires_in": self.WEB_SESSION_TTL
            })

        except Exception as e:
            logger.error(f"Login error: {e}")
            self._respond(500, {"error": "login failed"})

    def _check_web_session(self, token: str) -> dict | None:
        """Check if a token is a valid web session. Returns user dict or None."""
        session = _web_sessions.get(token)
        if not session:
            return None
        if time.time() > session["expires"]:
            del _web_sessions[token]
            return None
        return {
            "id": session["user_id"],
            "username": session["username"],
            "is_admin": session["is_admin"]
        }

    def _get_web_root(self) -> str:
        """Resolve the web UI directory. INI 'web_root' overrides default."""
        from .config import get_config
        cfg = get_config()
        custom = cfg.get("web_root", "")
        if custom and os.path.isdir(os.path.expanduser(custom)):
            return os.path.expanduser(custom)
        # Check standard locations
        for d in ["/var/ragger/www",
                  os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")]:
            if os.path.isdir(d):
                return d
        return ""

    def _serve_static(self):
        """Serve static files from the web UI directory."""
        web_root = self._get_web_root()
        if not os.path.isdir(web_root):
            self._respond(404, {"error": f"unknown endpoint: {self.path}"})
            return

        # Map / to /index.html
        path = self.path.split("?")[0]  # strip query string
        if path == "/":
            path = "/index.html"

        # Security: prevent directory traversal
        safe_path = os.path.normpath(path.lstrip("/"))
        if safe_path.startswith("..") or os.path.isabs(safe_path):
            self._respond(403, {"error": "forbidden"})
            return

        file_path = os.path.join(web_root, safe_path)
        if not os.path.isfile(file_path):
            self._respond(404, {"error": f"unknown endpoint: {self.path}"})
            return

        ext = os.path.splitext(file_path)[1].lower()
        content_type = self.MIME_TYPES.get(ext, "application/octet-stream")

        try:
            with open(file_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._respond(500, {"error": str(e)})

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
        # Login endpoint is public (no auth required)
        if self.path == '/auth/login':
            self._handle_login()
            return
        user = self._check_auth()
        if not user:
            self._respond(401, {"error": "unauthorized"})
            return
        try:
            params = self._read_body()
            
            if self.path == '/store':
                _touch_timelapse()
                text = params.get('text')
                if not text:
                    self._respond(400, {"error": "text required"})
                    return
                metadata = params.get('metadata') or {}
                # Default collection to "memory" if not specified
                if 'collection' not in metadata or not metadata['collection']:
                    metadata['collection'] = 'memory'
                # Default source to authenticated username
                if 'source' not in metadata or not metadata['source']:
                    metadata['source'] = user.get('username', 'unknown')
                common = params.get('common', False)
                mem = _get_memory(user.get('username'))
                defer = params.get('defer_embedding', False)
                memory_id = mem.store(text, metadata, common=common, defer_embedding=defer)
                self._respond(200, {"id": memory_id, "status": "stored"})
            
            elif self.path == '/search':
                query = params.get('query')
                if not query:
                    self._respond(400, {"error": "query required"})
                    return
                limit = params.get('limit', 5)
                min_score = params.get('min_score', 0.0)
                collections = params.get('collections', None)
                mem = _get_memory(user.get('username'))
                search_result = mem.search(query, limit, min_score, collections)
                results = search_result["results"]
                timing = search_result.get("timing", {})
                # Convert datetime to string for JSON (if not already converted by backend)
                for r in results:
                    ts = r.get('timestamp')
                    if ts and hasattr(ts, 'isoformat'):
                        r['timestamp'] = ts.isoformat()
                self._respond(200, {"results": results, "timing": timing})
            
            elif self.path == '/count':
                mem = _get_memory(user.get('username'))
                self._respond(200, {"count": mem.count()})
            
            elif self.path == '/delete_batch':
                memory_ids = params.get('ids', [])
                if not memory_ids:
                    self._respond(400, {"error": "ids required"})
                    return
                mem = _get_memory(user.get('username'))
                count = mem.delete_batch(memory_ids)
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
                mem = _get_memory(user.get('username'))
                results = mem.search_by_metadata(metadata_filter, limit)
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
                # Preload on local engines
                if _inference_client:
                    _preload_local_model(model)
                self._respond(200, {"status": "updated", "model": model, "username": username})
            
            elif self.path == '/chat':
                # Memory-augmented chat with SSE streaming
                _touch_timelapse()
                if not _inference_client:
                    self._respond(503, {"error": "inference not configured"})
                    return

                message = params.get('message', '').strip()
                if not message:
                    self._respond(400, {"error": "message required"})
                    return

                session_id = params.get('session_id')
                username = user.get('username', 'anonymous')
                session = get_or_create_session(session_id, username)

                # Search memory for context (user-scoped in multi-user mode)
                memory_context = ""
                mem = _get_memory(username)
                try:
                    cfg_obj = get_config()
                    max_results = cfg_obj.get("chat_max_memory_results", 5)
                    search_result = mem.search(message, max_results, 0.3)
                    chunks = [r['text'] for r in search_result.get('results', [])]
                    if chunks:
                        memory_context = "\n\n---\n\n".join(chunks)
                except Exception as e:
                    logger.warning(f"Memory search failed for /chat: {e}")

                # Resolve model
                backend = _memory._backend
                preferred_model = backend.get_user_preferred_model(username)
                request_model = params.get('model')
                use_model = preferred_model or request_model or _inference_client.model

                # Build messages with persona + memory + history
                system_prompt = load_workspace_files()
                session.add_user_message(message)
                full_messages = session.build_messages(system_prompt, memory_context)

                # Stream response via SSE
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Session-Id', session.session_id)
                self.end_headers()

                response_text = ""
                try:
                    # Ensure model is loaded (auto-load for local engines)
                    load_err = _inference_client.ensure_model_loaded(use_model)
                    if load_err:
                        event = json.dumps({"error": load_err})
                        self.wfile.write(f"data: {event}\n\n".encode())
                        self.wfile.flush()
                        self.wfile.write(f"data: {json.dumps({'done': True})}\n\n".encode())
                        self.wfile.flush()
                        return

                    stream = _inference_client.chat(
                        messages=full_messages,
                        stream=True,
                        model=use_model,
                    )
                    for chunk in stream:
                        delta = _inference_client.extract_delta(chunk, use_model)
                        if delta:
                            response_text += delta
                            event = json.dumps({"token": delta})
                            self.wfile.write(f"data: {event}\n\n".encode())
                            self.wfile.flush()

                    # Done event with session_id
                    done_event = json.dumps({
                        "done": True,
                        "session_id": session.session_id
                    })
                    self.wfile.write(f"data: {done_event}\n\n".encode())
                    self.wfile.flush()

                    # Update session with assistant response
                    if response_text:
                        session.add_assistant_message(response_text)

                        # Store turn if configured
                        from .config import get_config as _gc
                        cfg = _gc()
                        store_turns = cfg.get("chat_store_turns", "true")
                        if store_turns and store_turns != "false":
                            try:
                                mem.store(
                                    f"User: {message}\n\nAssistant: {response_text}",
                                    {
                                        "collection": "conversation",
                                        "category": "chat-turn",
                                        "source": f"chat-http-{username}",
                                    },
                                    defer_embedding=True,
                                )
                            except Exception as e:
                                logger.warning(f"Turn storage failed: {e}")

                    # Cleanup expired sessions in background
                    import threading
                    threading.Thread(
                        target=cleanup_expired_sessions,
                        args=(_memory, _inference_client, _get_memory),
                        daemon=True
                    ).start()

                except Exception as e:
                    error_event = json.dumps({"error": str(e)})
                    self.wfile.write(f"data: {error_event}\n\n".encode())
                    self.wfile.flush()

                self._perform_rotation()

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
            
            elif self.path == '/housekeeping':
                # Housekeeping: summarize idle sessions + clean expired conversations.
                # Designed to be called by cron every ~10 minutes.
                # Collects all known user DB paths from the per-user cache.
                user_db_paths = set()
                for um in _user_memories.values():
                    if um._user_backend and hasattr(um._user_backend, 'db_path'):
                        user_db_paths.add(um._user_backend.db_path)

                results = run_housekeeping(
                    memory=_memory,
                    inference_client=_inference_client,
                    memory_resolver=_get_memory,
                    user_db_paths=list(user_db_paths),
                )
                _touch_timelapse()
                self._respond(200, results)

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
                mem = _get_memory(user.get('username'))
                deleted = mem.delete(memory_id)
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
        # Static web UI files are public (auth happens in the JS/API layer)
        web_root = self._get_web_root()
        if os.path.isdir(web_root):
            path_clean = self.path.split("?")[0]
            check_path = "/index.html" if path_clean == "/" else path_clean
            safe = os.path.normpath(check_path.lstrip("/"))
            if not safe.startswith("..") and os.path.isfile(os.path.join(web_root, safe)):
                self._serve_static()
                return
        user = self._check_auth()
        if not user:
            self._respond(401, {"error": "unauthorized"})
            return
        if self.path == '/count':
            mem = _get_memory(user.get('username'))
            response = {"count": mem.count()}
            if mem.is_multi_db:
                response["user"] = mem._user_backend.count()
                response["common"] = mem._backend.count()
            self._respond(200, response)
        elif self.path == '/user/model':
            # GET /user/model — get preferred model for authenticated user
            username = user["username"]
            backend = _memory._backend
            model = backend.get_user_preferred_model(username)
            self._respond(200, {"model": model, "username": username})
        else:
            # Try serving static web UI files
            self._serve_static()
    
    def log_message(self, format, *args):
        """Route HTTP logs through our logger instead of stderr"""
        http_logger.info(f"{self.address_string()} {format % args}")


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """Start the HTTP server"""
    import socket
    import sys

    # Check if port is available before loading model
    # SO_REUSEADDR allows binding to TIME_WAIT sockets (normal after restart)
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        test_sock.bind((host, port))
    except OSError:
        print(lang.ERR_PORT_IN_USE.format(port=port), file=sys.stderr)
        sys.exit(1)
    finally:
        test_sock.close()

    global _memory, _server_token, _inference_client
    
    from .config import get_config
    cfg = get_config()
    single_user = cfg.get("single_user", True)
    
    # Initialize inference client if configured
    if cfg.get("inference_api_url") or cfg.get("inference_endpoints"):
        try:
            _inference_client = InferenceClient.from_config(cfg)
            print(f"Inference: enabled (default model: {_inference_client.model})")
        except Exception as e:
            logger.warning(f"Inference client initialization failed: {e}")
            print("Inference: disabled (no valid configuration)")
    
    if single_user:
        # Single-user mode: ensure token, create default user if needed
        _server_token = ensure_token()
        print(f"Auth: bearer token required (token: {token_path()})")
        _memory = RaggerMemory()
        
        if _server_token and hasattr(_memory, '_backend'):
            backend = _memory._backend
            hashed = hash_token(_server_token)
            existing = backend.get_user_by_token_hash(hashed)
            if not existing:
                import getpass
                username = getpass.getuser()
                user_id = backend.create_user(username, hashed, is_admin=True)
                print(f"Created user: {username} (id={user_id})")
            else:
                print(f"User: {existing['username']} (id={existing['id']})")
    else:
        # Multi-user mode: don't create tokens or default users.
        # Users are provisioned via install.sh / add-user.
        # Common DB only at init; per-user DBs opened on demand via _get_memory().
        import os
        common_path = os.path.expanduser(cfg["common_db_path"])
        _memory = RaggerMemory(uri=common_path)
        print(f"Multi-user mode: common={common_path}")
        print("Auth: via provisioned user tokens")
        print("User DBs: resolved per-user from ~username/.ragger/memories.db")
    
    server = HTTPServer((host, port), RaggerHandler)

    # TLS support
    tls_cert = cfg.get("tls_cert", "")
    tls_key = cfg.get("tls_key", "")
    if tls_cert and tls_key:
        cert_path = os.path.expanduser(tls_cert)
        key_path = os.path.expanduser(tls_key)
        if os.path.exists(cert_path) and os.path.exists(key_path):
            import ssl
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert_path, key_path)
            server.socket = ctx.wrap_socket(server.socket, server_side=True)
            print(f"TLS enabled: {cert_path}")
        else:
            print("WARNING: TLS certificates not found — starting without encryption")
            if not os.path.exists(cert_path):
                print(f"  Missing: {cert_path}")
            if not os.path.exists(key_path):
                print(f"  Missing: {key_path}")

    print(lang.MSG_SERVER_RUNNING.format(host=host, port=port))
    print(lang.MSG_SERVER_ENDPOINTS)
    print(f"  POST   /store               - {{\"text\": \"...\", \"metadata\": {{...}}}}")
    print(f"  POST   /search              - {{\"query\": \"...\", \"limit\": 5}}")
    print(f"  POST   /search_by_metadata  - {{\"metadata\": {{\"category\": \"...\"}}, \"limit\": 10}}")
    print(f"  POST   /delete_batch        - {{\"ids\": [\"1\", \"2\"]}}")
    print(f"  DELETE /memory/<id>         - delete by ID")
    print(f"  GET    /count               - memory count")
    print(f"  POST   /register            - {{\"username\": \"...\"}}")
    print(f"  POST   /chat                - memory-augmented chat (SSE streaming)")
    print(f"  POST   /v1/chat/completions - inference proxy (respects user model)")
    print(f"  PUT    /user/model          - set preferred model")
    print(f"  GET    /user/model          - get preferred model")
    print(f"  DELETE /user/model          - clear preferred model")
    print(f"  GET    /health              - health check")
    print(f"\n  Health check: curl http://{host}:{port}/health")
    print(lang.MSG_SERVER_STOP)

    # Warmup: pre-load embedding cache
    try:
        _memory.search("warmup", 1, 0.0)
        print(f"Warmup: embedding cache loaded ({_memory.count()} memories)")
    except Exception as e:
        print(f"Warmup: {e}")

    # Preload default model on local inference engines
    if _inference_client and _inference_client.model:
        _preload_local_model(_inference_client.model)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
        _memory.close()
