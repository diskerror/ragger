"""
Ragger Memory - MongoDB-based RAG memory store for OpenClaw

Usage:
    from ragger_memory import RaggerMemory
    
    memory = RaggerMemory()
    memory.store("Some important fact")
    results = memory.search("important")
"""

from .memory import RaggerMemory
from .config import (
    MONGODB_URI, DB_NAME, COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
)

__all__ = ['RaggerMemory']
__version__ = '2.0.0'
