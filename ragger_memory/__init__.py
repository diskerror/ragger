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
__version__ = '0.5.1'
