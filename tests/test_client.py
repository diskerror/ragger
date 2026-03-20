"""
Tests for the thin HTTP client and auth module
"""
import json
import os
import stat
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import pytest

from ragger_memory.client import RaggerClient, is_daemon_running
from ragger_memory.auth import (
    generate_token, ensure_token, load_token, validate_token
)


# -----------------------------------------------------------------------
# Auth tests
# -----------------------------------------------------------------------

class TestAuth:
    def test_generate_token_length(self):
        token = generate_token()
        assert len(token) > 20  # urlsafe_b64 of 32 bytes

    def test_generate_token_unique(self):
        t1 = generate_token()
        t2 = generate_token()
        assert t1 != t2

    def test_validate_token_match(self):
        token = generate_token()
        assert validate_token(token, token)

    def test_validate_token_mismatch(self):
        assert not validate_token("abc", "def")

    def test_ensure_token_creates_file(self, tmp_path, monkeypatch):
        token_file = tmp_path / "token"
        monkeypatch.setattr("ragger_memory.auth.token_path", lambda: str(token_file))

        token = ensure_token()
        assert token
        assert token_file.exists()
        assert token_file.read_text().strip() == token

    def test_ensure_token_permissions(self, tmp_path, monkeypatch):
        token_file = tmp_path / "token"
        monkeypatch.setattr("ragger_memory.auth.token_path", lambda: str(token_file))

        ensure_token()
        mode = os.stat(str(token_file)).st_mode
        # Should be 0660 (owner+group read/write)
        assert mode & stat.S_IRUSR
        assert mode & stat.S_IWUSR
        assert mode & stat.S_IRGRP
        assert mode & stat.S_IWGRP
        assert not (mode & stat.S_IROTH)
        assert not (mode & stat.S_IWOTH)

    def test_ensure_token_idempotent(self, tmp_path, monkeypatch):
        token_file = tmp_path / "token"
        monkeypatch.setattr("ragger_memory.auth.token_path", lambda: str(token_file))

        t1 = ensure_token()
        t2 = ensure_token()
        assert t1 == t2  # Same token on second call

    def test_load_token_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ragger_memory.auth.token_path",
                            lambda: str(tmp_path / "nonexistent"))
        assert load_token() is None


# -----------------------------------------------------------------------
# Mock server for client tests
# -----------------------------------------------------------------------

class MockHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self._respond(200, {"status": "ok", "memories": 42})
        elif self.path == '/count':
            self._respond(200, {"count": 42})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == '/store':
            self._respond(200, {"id": "test-123", "status": "stored"})
        elif self.path == '/search':
            self._respond(200, {
                "results": [{"id": 1, "text": "test result",
                             "score": 0.95, "metadata": {},
                             "timestamp": "2026-01-01"}],
                "timing": {"total_ms": 10}
            })
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # Suppress output


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock server on a random port"""
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


# -----------------------------------------------------------------------
# Client tests
# -----------------------------------------------------------------------

class TestClient:
    def test_health(self, mock_server):
        client = RaggerClient("127.0.0.1", mock_server)
        result = client.health()
        assert result["status"] == "ok"
        assert result["memories"] == 42

    def test_count(self, mock_server):
        client = RaggerClient("127.0.0.1", mock_server)
        assert client.count() == 42

    def test_store(self, mock_server):
        client = RaggerClient("127.0.0.1", mock_server)
        result = client.store("test memory")
        assert result == "test-123"

    def test_search(self, mock_server):
        client = RaggerClient("127.0.0.1", mock_server)
        result = client.search("test query")
        assert len(result["results"]) == 1
        assert result["results"][0]["score"] == 0.95

    def test_close_noop(self, mock_server):
        client = RaggerClient("127.0.0.1", mock_server)
        client.close()  # Should not raise

    def test_context_manager(self, mock_server):
        with RaggerClient("127.0.0.1", mock_server) as client:
            assert client.count() == 42


class TestDaemonDetection:
    def test_daemon_running(self, mock_server):
        assert is_daemon_running("127.0.0.1", mock_server)

    def test_daemon_not_running(self):
        assert not is_daemon_running("127.0.0.1", 19999)
