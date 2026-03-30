"""
Tests for the HTTP server.

Starts an actual HTTP server on a random port with a mock backend,
then makes real HTTP requests against it.
"""
import json
import threading
import time
from http.server import HTTPServer
from unittest.mock import MagicMock

import pytest

from ragger_memory import server as server_module
from ragger_memory.server import RaggerHandler


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_memory():
    """Create a mock RaggerMemory instance."""
    mem = MagicMock()
    mem.count.return_value = 42
    mem.is_multi_db = False
    mem.store.return_value = "123"
    mem.search.return_value = {
        "results": [
            {
                "id": "1",
                "text": "test result",
                "score": 0.85,
                "metadata": {"collection": "memory"},
                "timestamp": "2026-01-01T00:00:00"
            }
        ],
        "timing": {"total_ms": 5.0}
    }
    return mem


@pytest.fixture
def test_server(mock_memory):
    """Start a test HTTP server with mock backend, auth disabled."""
    original_memory = server_module._memory
    original_token = server_module._server_token
    server_module._memory = mock_memory
    server_module._server_token = None  # disable auth (single_user + no token)
    
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
    server_module._memory = original_memory
    server_module._server_token = original_token
    cfg["single_user"] = orig_single_user


def _get(port, path):
    import urllib.request
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def _post(port, path, data):
    import urllib.request
    url = f"http://127.0.0.1:{port}{path}"
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


class TestHealthEndpoint:
    def test_health_returns_ok(self, test_server):
        status, data = _get(test_server, "/health")
        assert status == 200
        assert data["status"] == "ok"
        assert data["memories"] == 42


class TestCountEndpoint:
    def test_get_count(self, test_server):
        status, data = _get(test_server, "/count")
        assert status == 200
        assert data["count"] == 42
    
    def test_post_count(self, test_server):
        status, data = _post(test_server, "/count", {})
        assert status == 200
        assert data["count"] == 42


class TestStoreEndpoint:
    def test_store_memory(self, test_server, mock_memory):
        status, data = _post(test_server, "/store", {
            "text": "test memory",
            "metadata": {"source": "test"}
        })
        assert status == 200
        assert data["status"] == "stored"
        assert data["id"] == "123"
        mock_memory.store.assert_called_once()
    
    def test_store_without_text_returns_400(self, test_server):
        try:
            _post(test_server, "/store", {"metadata": {}})
            assert False, "Should have raised"
        except Exception as e:
            assert "400" in str(e)
    
    def test_store_defaults_collection_to_memory(self, test_server, mock_memory):
        _post(test_server, "/store", {"text": "no collection"})
        call_args = mock_memory.store.call_args
        metadata = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("metadata", {})
        assert metadata.get("collection") == "memory"


class TestSearchEndpoint:
    def test_search(self, test_server, mock_memory):
        status, data = _post(test_server, "/search", {
            "query": "test query",
            "limit": 3
        })
        assert status == 200
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "test result"
        assert "timing" in data
    
    def test_search_without_query_returns_400(self, test_server):
        try:
            _post(test_server, "/search", {"limit": 5})
            assert False, "Should have raised"
        except Exception as e:
            assert "400" in str(e)


class TestUnknownEndpoint:
    def test_unknown_get_returns_404(self, test_server):
        try:
            _get(test_server, "/nonexistent")
            assert False, "Should have raised"
        except Exception as e:
            assert "404" in str(e)
    
    def test_unknown_post_returns_404(self, test_server):
        try:
            _post(test_server, "/nonexistent", {})
            assert False, "Should have raised"
        except Exception as e:
            assert "404" in str(e)


class TestPortCollision:
    """Tests for port collision handling."""
    
    def test_port_already_in_use(self, mock_memory):
        """Starting server on occupied port should raise error."""
        import socket
        
        # Bind to a port first
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind(('127.0.0.1', 0))
        blocker.listen(1)
        blocked_port = blocker.getsockname()[1]
        
        try:
            # Try to start server on same port
            original = server_module._memory
            server_module._memory = mock_memory
            
            with pytest.raises(OSError) as exc_info:
                httpd = HTTPServer(('127.0.0.1', blocked_port), RaggerHandler)
            
            # Should indicate address already in use
            assert exc_info.value.errno in (48, 98)  # EADDRINUSE (macOS=48, Linux=98)
            
            server_module._memory = original
        finally:
            blocker.close()
    
    def test_find_free_port_returns_available(self):
        """_find_free_port should return an available port."""
        port = _find_free_port()
        
        # Should be valid port number
        assert 1024 <= port <= 65535
        
        # Should be bindable
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            # If we got here, port was available
    
    def test_sequential_servers_use_different_ports(self, mock_memory):
        """Multiple server instances should use different ports."""
        original = server_module._memory
        server_module._memory = mock_memory
        
        port1 = _find_free_port()
        httpd1 = HTTPServer(('127.0.0.1', port1), RaggerHandler)
        thread1 = threading.Thread(target=httpd1.serve_forever, daemon=True)
        thread1.start()
        
        # Second server should get different port
        port2 = _find_free_port()
        assert port2 != port1
        
        httpd2 = HTTPServer(('127.0.0.1', port2), RaggerHandler)
        thread2 = threading.Thread(target=httpd2.serve_forever, daemon=True)
        thread2.start()
        
        # Both should respond
        status1, _ = _get(port1, "/health")
        status2, _ = _get(port2, "/health")
        assert status1 == 200
        assert status2 == 200
        
        httpd1.shutdown()
        httpd2.shutdown()
        server_module._memory = original
