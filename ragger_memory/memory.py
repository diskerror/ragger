"""
RaggerMemory - Factory facade for backend-agnostic memory storage

Stores text with embeddings. Performs vector similarity search.
Backends inherit from MemoryBackend (backend.py).
"""

import logging
from typing import Optional, Dict, Any, List

from .embedding import Embedder
from . import lang

logger = logging.getLogger(__name__)


class RaggerMemory:
    """Factory facade — creates the right backend based on config"""
    
    def __init__(self, uri: Optional[str] = None, engine: Optional[str] = None):
        """
        Initialize memory store with appropriate backend
        
        Args:
            uri: File path for SQLite database.
                 Defaults to the value in config.
            engine: Storage engine (currently "sqlite").
                    Defaults to config.STORAGE_ENGINE.
        """
        from .config import STORAGE_ENGINE
        self.engine = engine or STORAGE_ENGINE
        
        # Create embedder (shared across all backends)
        embedder = Embedder()
        
        if self.engine == "sqlite":
            from .sqlite_backend import SqliteBackend
            self._backend = SqliteBackend(embedder, uri)
            logger.info(lang.MSG_USING_SQLITE)
        else:
            raise ValueError(
                lang.ERR_UNKNOWN_ENGINE.format(engine=self.engine)
            )
    
    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a memory with vector embedding
        
        Args:
            text: The memory text
            metadata: Optional metadata (source, tags, etc.)
        
        Returns:
            Memory ID (str)
        """
        return self._backend.store(text, metadata)
    
    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
        collections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Vector search for relevant memories using cosine similarity.
        
        Args:
            query: Search query text
            limit: Maximum results to return
            min_score: Minimum similarity score (0.0-1.0)
            collections: Collections to search. None = all collections.
                         Use explicit list to narrow, e.g. ["memory"] or ["sibelius"].
        
        Returns:
            Dict with 'results' list and 'timing' dict
        """
        return self._backend.search(query, limit, min_score, collections)
    
    def count(self) -> int:
        """Return number of stored memories"""
        return self._backend.count()
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID
        
        Args:
            memory_id: Memory ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        return self._backend.delete(memory_id)
    
    def delete_batch(self, memory_ids: list) -> int:
        """
        Delete multiple memories by ID
        
        Args:
            memory_ids: List of memory IDs to delete
        
        Returns:
            Number of memories deleted
        """
        return self._backend.delete_batch(memory_ids)
    
    def search_by_metadata(self, metadata_filter: dict, limit: int = None) -> list:
        """
        Search memories by metadata fields
        
        Args:
            metadata_filter: Dict of metadata fields to match
            limit: Maximum results to return (None = all)
        
        Returns:
            List of dicts with id, text, metadata, timestamp
        """
        return self._backend.search_by_metadata(metadata_filter, limit)
    
    def close(self):
        """Close backend connection"""
        self._backend.close()
    
    @staticmethod
    def download_model():
        """Download or update the embedding model from HuggingFace"""
        return Embedder.download_model()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
