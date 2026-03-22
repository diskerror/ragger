"""
Thin HTTP client for Ragger Memory

Used by the CLI when a daemon is already running.
Avoids loading the embedding model — sub-100ms queries.
"""

import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List


class RaggerClient:
    """HTTP client that mirrors the RaggerMemory interface"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8432,
                 token: Optional[str] = None):
        self.base_url = f"http://{host}:{port}"
        self.token = token

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _post(self, path: str, data: dict) -> dict:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers=self._headers()
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    
    def _delete(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers=self._headers(),
            method="DELETE"
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory via the daemon"""
        data = {"text": text}
        if metadata:
            data["metadata"] = metadata
        result = self._post("/store", data)
        return result.get("id", "")

    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
        collections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search memories via the daemon"""
        data = {"query": query, "limit": limit, "min_score": min_score}
        if collections:
            data["collections"] = collections
        return self._post("/search", data)

    def count(self) -> int:
        """Get memory count via the daemon"""
        result = self._get("/count")
        return result.get("count", 0)

    def health(self) -> dict:
        """Check daemon health"""
        return self._get("/health")
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID via the daemon"""
        result = self._delete(f"/memory/{memory_id}")
        return result.get("status") == "deleted"
    
    def delete_batch(self, memory_ids: list) -> int:
        """Delete multiple memories by ID via the daemon"""
        data = {"ids": memory_ids}
        result = self._post("/delete_batch", data)
        return result.get("deleted", 0)
    
    def search_by_metadata(self, metadata_filter: dict, limit: int = None) -> list:
        """Search memories by metadata fields via the daemon"""
        data = {"metadata": metadata_filter}
        if limit:
            data["limit"] = limit
        result = self._post("/search_by_metadata", data)
        return result.get("results", [])

    def close(self):
        """No-op — HTTP is stateless"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def is_daemon_running(host: str = "127.0.0.1", port: int = 8432) -> bool:
    """Check if a ragger daemon is responding on the given port"""
    try:
        req = urllib.request.Request(f"http://{host}:{port}/health")
        with urllib.request.urlopen(req, timeout=1) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError):
        return False
