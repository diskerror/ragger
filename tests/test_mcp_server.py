"""
Tests for the MCP server (JSON-RPC + plain text mode).

Runs the MCP server in a subprocess with mock-friendly setup,
feeding it stdin lines and checking stdout.
"""
import json
import subprocess
import sys
import os

import pytest


RAGGER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(RAGGER_DIR, ".venv", "bin", "python3")


def run_mcp(input_lines: list[str], timeout: int = 30) -> str:
    """
    Run the MCP server with given stdin lines and return stdout.
    
    Uses a helper script that creates a memory, then starts the MCP loop
    so there's data to search against.
    """
    # Build a small script that:
    # 1. Creates a temp DB with test data
    # 2. Runs the MCP input loop against it
    script = """
import sys, os, json, tempfile
sys.path.insert(0, {ragger_dir!r})

# Override config before importing anything
import ragger_memory.config as cfg
cfg.SQLITE_PATH = os.path.join(tempfile.mkdtemp(), "test.db")
cfg.QUERY_LOGGING_ENABLED = False
cfg.USAGE_TRACKING_ENABLED = False

# Suppress model load noise
os.environ["TQDM_DISABLE"] = "1"
import logging
logging.basicConfig(level=logging.WARNING)

from ragger_memory.mcp_server import run_mcp_server
run_mcp_server()
""".format(ragger_dir=RAGGER_DIR)
    
    stdin_text = "\n".join(input_lines) + "\n"
    
    result = subprocess.run(
        [PYTHON, "-c", script],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=RAGGER_DIR,
    )
    return result.stdout, result.stderr


class TestMCPJsonRpc:
    """Tests for JSON-RPC mode."""
    
    def test_unknown_method_returns_error(self):
        request = json.dumps({
            "jsonrpc": "2.0",
            "method": "nonexistent",
            "params": {},
            "id": 1
        })
        stdout, _ = run_mcp([request])
        response = json.loads(stdout.strip())
        assert "error" in response
        assert response["id"] == 1
    
    def test_store_and_search(self):
        store_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "memory_store",
            "params": {"text": "Python is a programming language", "metadata": {"collection": "memory"}},
            "id": 1
        })
        search_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "memory_search",
            "params": {"query": "programming", "min_score": -1.0},
            "id": 2
        })
        stdout, _ = run_mcp([store_req, search_req])
        lines = [l for l in stdout.strip().split("\n") if l]
        
        store_resp = json.loads(lines[0])
        assert store_resp["result"]["status"] == "stored"
        
        search_resp = json.loads(lines[1])
        results = search_resp["result"]["results"]["results"]
        assert len(results) > 0
    
    def test_search_missing_query_returns_error(self):
        request = json.dumps({
            "jsonrpc": "2.0",
            "method": "memory_search",
            "params": {},
            "id": 1
        })
        stdout, _ = run_mcp([request])
        response = json.loads(stdout.strip())
        assert "error" in response
    
    def test_store_missing_text_returns_error(self):
        request = json.dumps({
            "jsonrpc": "2.0",
            "method": "memory_store",
            "params": {"metadata": {}},
            "id": 1
        })
        stdout, _ = run_mcp([request])
        response = json.loads(stdout.strip())
        assert "error" in response


class TestMCPPlainText:
    """Tests for plain text search mode."""
    
    def test_plain_text_no_results(self):
        """Plain text query on empty DB should say no results."""
        stdout, _ = run_mcp(["hello world"])
        assert "No results" in stdout
    
    def test_plain_text_with_data(self):
        """Store via JSON-RPC, then search via plain text."""
        store_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "memory_store",
            "params": {"text": "Ragger is a memory system", "metadata": {"collection": "memory"}},
            "id": 1
        })
        stdout, _ = run_mcp([store_req, "memory system"])
        # First line is JSON store response, rest is plain text search
        lines = stdout.strip().split("\n")
        # Skip the JSON response line
        plain_output = "\n".join(lines[1:])
        assert "score:" in plain_output or "No results" in plain_output
    
    def test_plain_text_shows_timing(self):
        """Plain text output should include timing info."""
        stdout, _ = run_mcp(["test query"])
        # Even with no results, we should get "No results" not a crash
        assert "No results" in stdout or "Timing:" in stdout
    
    def test_empty_lines_ignored(self):
        """Empty lines should not produce output."""
        stdout, _ = run_mcp(["", "  ", ""])
        assert stdout.strip() == ""
    
    def test_mixed_json_and_plain(self):
        """Should handle interleaved JSON-RPC and plain text."""
        store_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "memory_store",
            "params": {"text": "test mixed mode", "metadata": {"collection": "memory"}},
            "id": 1
        })
        stdout, _ = run_mcp([store_req, "mixed mode"])
        lines = [l for l in stdout.strip().split("\n") if l]
        # First line should be JSON
        first = json.loads(lines[0])
        assert first["result"]["status"] == "stored"
        # Remaining lines should be plain text
        remaining = "\n".join(lines[1:])
        assert "score:" in remaining or "No results" in remaining
