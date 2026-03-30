"""
Tests for /v1/chat/completions inference proxy endpoint.
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


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def inference_server(mock_embedder, tmp_db):
    """Server with mock inference client."""
    import sqlite3
    from ragger_memory.sqlite_backend import SqliteBackend
    from ragger_memory.auth import hash_token

    backend = SqliteBackend(mock_embedder, tmp_db)
    backend.conn.close()
    backend.conn = sqlite3.connect(tmp_db, check_same_thread=False)

    token_hash = hash_token("test-token")
    backend.create_user("testuser", token_hash)

    mem = MagicMock()
    mem._backend = backend
    mem.count.return_value = 0
    mem.is_multi_db = False

    mock_inference = MagicMock()
    mock_inference.model = "test-model"
    mock_inference.max_tokens = 4096
    mock_inference.chat.return_value = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        "model": "test-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }

    original_memory = server_module._memory
    original_token = server_module._server_token
    original_inference = server_module._inference_client
    server_module._memory = mem
    server_module._server_token = None
    server_module._inference_client = mock_inference

    from ragger_memory.config import get_config
    cfg = get_config()
    orig_single_user = cfg.get("single_user", True)
    cfg["single_user"] = True

    port = _find_free_port()
    httpd = HTTPServer(('127.0.0.1', port), RaggerHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield port, mock_inference

    httpd.shutdown()
    backend.close()
    server_module._memory = original_memory
    server_module._server_token = original_token
    server_module._inference_client = original_inference
    cfg["single_user"] = orig_single_user


class TestChatCompletions:
    def test_basic_completion(self, inference_server):
        port, mock_inf = inference_server
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        body = json.dumps({
            "messages": [{"role": "user", "content": "Hello"}]
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert "choices" in data
        mock_inf.chat.assert_called_once()

    def test_requires_messages(self, inference_server):
        port, _ = inference_server
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        body = json.dumps({"model": "test"}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400

    def test_passes_model_to_inference(self, inference_server):
        port, mock_inf = inference_server
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        body = json.dumps({
            "messages": [{"role": "user", "content": "test"}],
            "model": "custom-model"
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
        # Should have used custom-model (no preferred model set)
        call_kwargs = mock_inf.chat.call_args
        assert "custom-model" in str(call_kwargs)

    def test_max_tokens_passthrough(self, inference_server):
        port, mock_inf = inference_server
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        body = json.dumps({
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 100
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
        call_kwargs = mock_inf.chat.call_args
        assert "100" in str(call_kwargs) or call_kwargs[1].get("max_tokens") == 100
