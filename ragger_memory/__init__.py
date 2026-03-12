"""
Ragger Memory - Backend-agnostic RAG memory store for OpenClaw

Supports multiple storage backends (MongoDB, SQLite) via inheritance.

Usage:
    from ragger_memory import RaggerMemory
    
    # Use default backend from config
    memory = RaggerMemory()
    
    # Or specify backend explicitly
    memory = RaggerMemory(engine="sqlite")
    memory = RaggerMemory(engine="mongodb", uri="mongodb://localhost:27017/")
    
    memory.store("Some important fact")
    results = memory.search("important")
"""

from .memory import RaggerMemory
from .config import (
    STORAGE_ENGINE,
    MONGODB_URI,
    SQLITE_PATH,
    MONGODB_DB_NAME,
    MONGODB_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS
)

__all__ = ['RaggerMemory']
__version__ = '0.3.1'
