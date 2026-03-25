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
__version__ = '0.8.0'


def build_version() -> str:
    """
    Version with git commit hash and build date.
    
    Format:
        ragger 0.8.0
        commit <7-char hash>
        built  <date>
    """
    import subprocess
    import os
    from datetime import datetime
    
    lines = [f"ragger {__version__}"]
    
    # Git commit hash (7 chars)
    try:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=pkg_dir, capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            lines.append(f"commit {commit}")
        else:
            lines.append("commit unknown")
    except Exception:
        lines.append("commit unknown")
    
    # Build date from __init__.py mtime
    try:
        init_file = os.path.join(os.path.dirname(__file__), "__init__.py")
        mtime = os.path.getmtime(init_file)
        build_date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
        lines.append(f"built  {build_date}")
    except Exception:
        lines.append(f"built  {datetime.now().strftime('%Y-%m-%d')}")
    
    return "\n".join(lines)
