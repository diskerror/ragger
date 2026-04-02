"""
Tests for /auth/login endpoint and web session management.
"""
import json
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer
from unittest.mock import MagicMock, patch

import pytest

from ragger_memory import server as server_module
from ragger_memory.server import RaggerHandler
from ragger_memory.auth import hash_password, hash_token


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def auth_server(mock_embedder, tmp_db):
    """Start server with a real backend that has a user with a password."""
    from ragger_memory.sqlite_backend import SqliteBackend
    from ragger_memory.memory import RaggerMemory
    
    import sqlite3
    backend = SqliteBackend(mock_embedder, tmp_db)
    # Allow cross-thread access for test (backend created here, used in server thread)
    backend.conn.close()
    backend.conn = sqlite3.connect(tmp_db, check_same_thread=False)
    
    # Create a test user with password
    token_hash = hash_token("test-bearer-token")
    backend.create_user("testuser", token_hash)
    backend.set_user_password("testuser", hash_password("testpass123"))
    
    # Create admin user
    admin_hash = hash_token("admin-bearer-token")
    backend.create_user("admin", admin_hash)
    backend.set_user_password("admin", hash_password("adminpass"))
    
    # Build a minimal RaggerMemory mock that exposes the real backend
    mem = MagicMock()
    mem._backend = backend
    mem.count.return_value = 0
    mem.is_multi_db = False
    
    original_memory = server_module._memory
    original_token = server_module._server_token
    server_module._memory = mem
    server_module._server_token = None
    
    from ragger_memory.config import get_config
    cfg = get_config()
    orig_single_user = cfg.get("single_user", True)
    cfg["single_user"] = True
    
    port = _find_free_port()
    httpd = HTTPServer(('127.0.0.1', port), RaggerHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    
    yield port
    
    httpd.shutdown()
    backend.close()
    server_module._memory = original_memory
    server_module._server_token = original_token
    cfg["single_user"] = orig_single_user


def _post(port, path, body, headers=None):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read()), dict(resp.headers)


def _post_error(port, path, body, expected_code):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == expected_code
    return json.loads(exc_info.value.read())


class TestAuthLogin:
    def test_login_success(self, auth_server):
        """Valid credentials return a session token."""
        status, data, headers = _post(auth_server, "/auth/login", {
            "username": "testuser",
            "password": "testpass123"
        })
        assert status == 200
        assert "token" in data
        assert data["username"] == "testuser"
        assert "expires_in" in data

    def test_login_wrong_password(self, auth_server):
        data = _post_error(auth_server, "/auth/login", {
            "username": "testuser",
            "password": "wrongpass"
        }, 401)
        assert "error" in data

    def test_login_nonexistent_user(self, auth_server):
        data = _post_error(auth_server, "/auth/login", {
            "username": "nobody",
            "password": "anything"
        }, 401)
        assert "error" in data

    def test_login_missing_fields(self, auth_server):
        data = _post_error(auth_server, "/auth/login", {}, 400)
        assert "error" in data

    def test_session_token_authenticates(self, auth_server):
        """Session token from login can authenticate API requests."""
        # Login first
        _, login_data, _ = _post(auth_server, "/auth/login", {
            "username": "testuser",
            "password": "testpass123"
        })
        token = login_data["token"]

        # Use token as cookie to access authenticated endpoint
        url = f"http://127.0.0.1:{auth_server}/count"
        req = urllib.request.Request(url, headers={
            "Content-Type": "application/json",
            "Cookie": f"ragger_token={token}"
        })
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
