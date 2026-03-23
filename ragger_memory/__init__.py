"""
Ragger Memory - Backend-agnostic RAG memory store

Stores text with local embeddings and performs hybrid vector + BM25 search.
SQLite backend included; see backend.py for implementing custom backends.

Usage:
    from ragger_memory import RaggerMemory
    
    memory = RaggerMemory()
    memory.store("Some important fact")
    results = memory.search("important")
"""

from .memory import RaggerMemory
from .config import (
    STORAGE_ENGINE,
    SQLITE_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS
)

__all__ = ['RaggerMemory']
__version__ = '0.7.0'


def build_version() -> str:
    """Version with date suffix if not on a tagged release."""
    import subprocess
    import os
    try:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=pkg_dir, capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return __version__
    except Exception:
        pass
    from datetime import datetime
    return f"{__version__}-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
