"""
Tests for the MCP server (MCP-compliant JSON-RPC + plain text mode).

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


def run_mcp(input_lines: list[str], timeout: int = 30) -> tuple[str, str]:
    """
    Run the MCP server with given stdin lines and return (stdout, stderr).
    """
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


def mcp_request(method: str, params: dict = None, req_id: int = 1) -> str:
    """Build a JSON-RPC request string."""
    req = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        req["params"] = params
    return json.dumps(req)


def mcp_notification(method: str, params: dict = None) -> str:
    """Build a JSON-RPC notification (no id field)."""
    req = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        req["params"] = params
    return json.dumps(req)


class TestMCPInitialize:
    """Tests for MCP initialize handshake."""

    def test_initialize_returns_server_info(self):
        stdout, _ = run_mcp([mcp_request("initialize")])
        response = json.loads(stdout.strip())
        assert response["id"] == 1
        result = response["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "ragger-memory"
        assert result["serverInfo"]["version"] == "0.7.0"
        assert "tools" in result["capabilities"]

    def test_initialized_notification_no_response(self):
        """notifications/initialized should produce no JSON-RPC response."""
        stdout, _ = run_mcp([mcp_notification("notifications/initialized")])
        # Should be empty — no response for notifications
        assert stdout.strip() == ""


class TestMCPToolsList:
    """Tests for tools/list."""

    def test_tools_list_returns_both_tools(self):
        stdout, _ = run_mcp([mcp_request("tools/list")])
        response = json.loads(stdout.strip())
        tools = response["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {"store", "search"}

    def test_tools_have_input_schemas(self):
        stdout, _ = run_mcp([mcp_request("tools/list")])
        response = json.loads(stdout.strip())
        tools = response["result"]["tools"]
        for tool in tools:
            assert "inputSchema" in tool
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema


class TestMCPToolsCall:
    """Tests for tools/call."""

    def test_store_and_search(self):
        store = mcp_request("tools/call", {
            "name": "store",
            "arguments": {"text": "Python is great", "metadata": {"collection": "memory"}}
        }, req_id=1)
        search = mcp_request("tools/call", {
            "name": "search",
            "arguments": {"query": "programming", "min_score": -1.0}
        }, req_id=2)
        stdout, _ = run_mcp([store, search])
        lines = [l for l in stdout.strip().split("\n") if l]

        store_resp = json.loads(lines[0])
        content = store_resp["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        stored = json.loads(content[0]["text"])
        assert stored["status"] == "stored"
        assert "id" in stored

        search_resp = json.loads(lines[1])
        content = search_resp["result"]["content"]
        assert len(content) == 1
        results = json.loads(content[0]["text"])
        # results could be a dict with "results" key or a list
        if isinstance(results, dict):
            results = results.get("results", [])
        assert len(results) > 0

    def test_store_missing_text_returns_error(self):
        req = mcp_request("tools/call", {
            "name": "store",
            "arguments": {"metadata": {}}
        })
        stdout, _ = run_mcp([req])
        response = json.loads(stdout.strip())
        assert response["result"]["isError"] is True

    def test_search_missing_query_returns_error(self):
        req = mcp_request("tools/call", {
            "name": "search",
            "arguments": {}
        })
        stdout, _ = run_mcp([req])
        response = json.loads(stdout.strip())
        assert response["result"]["isError"] is True

    def test_unknown_tool_returns_error(self):
        req = mcp_request("tools/call", {
            "name": "nonexistent",
            "arguments": {}
        })
        stdout, _ = run_mcp([req])
        response = json.loads(stdout.strip())
        result = response["result"]
        assert result["isError"] is True
        assert "Unknown tool" in result["content"][0]["text"]


class TestMCPErrors:
    """Tests for error handling."""

    def test_unknown_method_returns_32601(self):
        req = mcp_request("nonexistent")
        stdout, _ = run_mcp([req])
        response = json.loads(stdout.strip())
        assert "error" in response
        assert response["error"]["code"] == -32601


class TestMCPPlainText:
    """Tests for plain text search mode."""

    def test_plain_text_no_results(self):
        stdout, _ = run_mcp(["hello world"])
        assert "No results" in stdout

    def test_plain_text_with_data(self):
        store = mcp_request("tools/call", {
            "name": "store",
            "arguments": {"text": "Ragger is a memory system", "metadata": {"collection": "memory"}}
        })
        stdout, _ = run_mcp([store, "memory system"])
        lines = stdout.strip().split("\n")
        plain_output = "\n".join(lines[1:])
        assert "score:" in plain_output or "No results" in plain_output

    def test_empty_lines_ignored(self):
        stdout, _ = run_mcp(["", "  ", ""])
        assert stdout.strip() == ""

    def test_mixed_json_and_plain(self):
        store = mcp_request("tools/call", {
            "name": "store",
            "arguments": {"text": "test mixed mode", "metadata": {"collection": "memory"}}
        })
        stdout, _ = run_mcp([store, "mixed mode"])
        lines = [l for l in stdout.strip().split("\n") if l]
        first = json.loads(lines[0])
        assert "result" in first
        remaining = "\n".join(lines[1:])
        assert "score:" in remaining or "No results" in remaining
