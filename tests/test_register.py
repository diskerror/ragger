"""
Tests for the POST /register endpoint.
"""
import json
import threading
from http.server import HTTPServer
from unittest.mock import MagicMock, patch, mock_open
import urllib.request
import urllib.error

import pytest

from ragger_memory import server as server_module
from ragger_memory.server import RaggerHandler
from ragger_memory.auth import hash_token


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _post(port, path, data, headers=None):
    url = f"http://127.0.0.1:{port}{path}"
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _fake_pwnam(username):
    """Return a mock pwd entry."""
    pw = MagicMock()
    pw.pw_dir = f"/home/{username}"
    pw.pw_uid = 1001
    pw.pw_gid = 1001
    return pw


@pytest.fixture
def register_server():
    """Start a test server with mock backend that has user management methods."""
    mock_mem = MagicMock()
    mock_mem.count.return_value = 0
    mock_mem.is_multi_db = False
    mock_mem._backend = MagicMock()
    mock_mem._backend.get_user_by_username.return_value = None
    mock_mem._backend.get_user_by_token_hash.return_value = None
    mock_mem._backend.create_user.return_value = 42
    
    original_mem = server_module._memory
    original_token = server_module._server_token
    # Set a server token so auth is enabled but _check_auth won't block /register
    # Actually _check_auth is called first — we need it to pass.
    # The /register endpoint re-checks the Authorization header itself.
    # Set _server_token to None so _check_auth returns the anonymous user.
    server_module._memory = mock_mem
    server_module._server_token = None  # auth disabled = anonymous access
    
    port = _find_free_port()
    httpd = HTTPServer(('127.0.0.1', port), RaggerHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    
    yield port, mock_mem
    
    httpd.shutdown()
    server_module._memory = original_mem
    server_module._server_token = original_token


class TestRegisterEndpoint:
    
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', mock_open(read_data='test-token-123\n'))
    @patch('pwd.getpwnam', side_effect=_fake_pwnam)
    def test_register_creates_user(self, mock_pwnam, mock_exists, register_server):
        port, mock_mem = register_server
        mock_mem._backend.get_user_by_username.return_value = None
        mock_mem._backend.create_user.return_value = 42
        
        status, data = _post(port, '/register',
                             {'username': 'testuser'},
                             {'Authorization': 'Bearer test-token-123'})
        assert status == 200
        assert data['status'] == 'created'
        assert data['user_id'] == 42

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', mock_open(read_data='test-token-123\n'))
    @patch('pwd.getpwnam', side_effect=_fake_pwnam)
    def test_register_existing_user(self, mock_pwnam, mock_exists, register_server):
        port, mock_mem = register_server
        hashed = hash_token('test-token-123')
        mock_mem._backend.get_user_by_username.return_value = {
            'id': 7, 'username': 'testuser', 'token_hash': hashed
        }
        
        status, data = _post(port, '/register',
                             {'username': 'testuser'},
                             {'Authorization': 'Bearer test-token-123'})
        assert status == 200
        assert data['status'] == 'exists'
        assert data['user_id'] == 7

    def test_register_missing_username(self, register_server):
        port, _ = register_server
        status, data = _post(port, '/register', {},
                             {'Authorization': 'Bearer sometoken'})
        assert status == 400
        assert 'username' in data['error']

    def test_register_no_auth_header(self, register_server):
        port, _ = register_server
        status, data = _post(port, '/register', {'username': 'testuser'})
        assert status == 401

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', mock_open(read_data='correct-token\n'))
    @patch('pwd.getpwnam', side_effect=_fake_pwnam)
    def test_register_token_mismatch(self, mock_pwnam, mock_exists, register_server):
        port, _ = register_server
        
        status, data = _post(port, '/register',
                             {'username': 'testuser'},
                             {'Authorization': 'Bearer wrong-token'})
        assert status == 403

    @patch('pwd.getpwnam', side_effect=KeyError('no such user'))
    def test_register_unknown_system_user(self, mock_pwnam, register_server):
        port, _ = register_server
        
        status, data = _post(port, '/register',
                             {'username': 'nouser'},
                             {'Authorization': 'Bearer sometoken'})
        assert status == 400
        assert 'unknown system user' in data['error']

    @patch('os.path.exists', return_value=False)
    @patch('pwd.getpwnam', side_effect=_fake_pwnam)
    def test_register_no_token_file(self, mock_pwnam, mock_exists, register_server):
        port, _ = register_server
        
        status, data = _post(port, '/register',
                             {'username': 'testuser'},
                             {'Authorization': 'Bearer sometoken'})
        assert status == 400
        assert 'no token file' in data['error']

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', mock_open(read_data='new-token\n'))
    @patch('pwd.getpwnam', side_effect=_fake_pwnam)
    def test_register_updates_token_hash(self, mock_pwnam, mock_exists, register_server):
        port, mock_mem = register_server
        mock_mem._backend.get_user_by_username.return_value = {
            'id': 7, 'username': 'testuser', 'token_hash': 'old-hash'
        }
        
        status, data = _post(port, '/register',
                             {'username': 'testuser'},
                             {'Authorization': 'Bearer new-token'})
        assert status == 200
        assert data['status'] == 'exists'
        mock_mem._backend.update_user_token.assert_called_once_with(
            'testuser', hash_token('new-token'))
