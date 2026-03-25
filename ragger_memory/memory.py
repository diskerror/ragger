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
    """Factory facade — creates the right backend based on config.
    
    Single-user: one backend (user's DB).
    Multi-user: two backends — common DB (shared) + user DB (private).
    Search merges both. Store routes by collection/flag.
    """
    
    def __init__(self, uri: Optional[str] = None, engine: Optional[str] = None,
                 user_db_path: Optional[str] = None):
        """
        Initialize memory store with appropriate backend(s).
        
        Args:
            uri: File path for primary database. Defaults to config.
            engine: Storage engine (currently "sqlite").
            user_db_path: Path to user's private DB (multi-user mode).
                          If provided, uri becomes the common DB path
                          and user_db_path becomes the user's private DB.
        """
        from .config import STORAGE_ENGINE
        self.engine = engine or STORAGE_ENGINE
        
        # Create embedder (shared across all backends)
        embedder = Embedder()
        
        if self.engine == "sqlite":
            from .sqlite_backend import SqliteBackend
            self._backend = SqliteBackend(embedder, uri)
            logger.info(lang.MSG_USING_SQLITE)
            
            # Multi-user: open a second backend for the user's private DB
            self._user_backend = None
            if user_db_path:
                self._user_backend = SqliteBackend(embedder, user_db_path)
                logger.info(f"User DB: {user_db_path}")
        else:
            raise ValueError(
                lang.ERR_UNKNOWN_ENGINE.format(engine=self.engine)
            )
    
    @property
    def is_multi_db(self) -> bool:
        """True if operating with separate common + user databases."""
        return self._user_backend is not None
    
    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None,
              common: bool = False) -> str:
        """
        Store a memory with vector embedding.
        
        In multi-DB mode, stores to user's private DB by default.
        Set common=True to store to the shared common DB.
        In single-DB mode, common flag is ignored.
        
        When common=True, automatically adds "keep" to tags
        to prevent deletion.
        
        Args:
            text: The memory text
            metadata: Optional metadata (source, tags, etc.)
            common: Store to common DB instead of user DB (multi-DB only)
        
        Returns:
            Memory ID (str)
        """
        # Auto-set "keep" tag when storing to common DB
        if common:
            if metadata is None:
                metadata = {}
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            if "keep" not in tags:
                tags.append("keep")
            metadata["tags"] = tags
        
        if self._user_backend and not common:
            return self._user_backend.store(text, metadata)
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
        
        In multi-DB mode, searches both common and user DBs,
        merges results by score, and returns the top `limit`.
        
        Args:
            query: Search query text
            limit: Maximum results to return
            min_score: Minimum similarity score (0.0-1.0)
            collections: Collections to search. None = all collections.
        
        Returns:
            Dict with 'results' list and 'timing' dict
        """
        if not self._user_backend:
            return self._backend.search(query, limit, min_score, collections)
        
        # Multi-DB: query both, merge by score
        common_result = self._backend.search(query, limit, min_score, collections)
        user_result = self._user_backend.search(query, limit, min_score, collections)
        
        # Merge results, sort by score descending, take top limit
        all_results = common_result.get("results", []) + user_result.get("results", [])
        all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
        
        # Merge timing info
        ct = common_result.get("timing", {})
        ut = user_result.get("timing", {})
        merged_timing = {
            "embedding_ms": ct.get("embedding_ms", 0),  # same query, same embedding
            "search_ms": (ct.get("search_ms", 0) or 0) + (ut.get("search_ms", 0) or 0),
            "total_ms": (ct.get("total_ms", 0) or 0) + (ut.get("total_ms", 0) or 0),
            "corpus_size": (ct.get("corpus_size", 0) or 0) + (ut.get("corpus_size", 0) or 0),
        }
        
        return {
            "results": all_results[:limit],
            "timing": merged_timing,
        }
    
    def count(self) -> int:
        """Return number of stored memories (across all DBs)"""
        total = self._backend.count()
        if self._user_backend:
            total += self._user_backend.count()
        return total
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID. Tries user DB first, then common.
        
        Args:
            memory_id: Memory ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        if self._user_backend:
            if self._user_backend.delete(memory_id):
                return True
        return self._backend.delete(memory_id)
    
    def delete_batch(self, memory_ids: list) -> int:
        """
        Delete multiple memories by ID. Tries both DBs.
        
        Args:
            memory_ids: List of memory IDs to delete
        
        Returns:
            Number of memories deleted
        """
        deleted = self._backend.delete_batch(memory_ids)
        if self._user_backend:
            deleted += self._user_backend.delete_batch(memory_ids)
        return deleted
    
    def search_by_metadata(self, metadata_filter: dict, limit: int = None) -> list:
        """
        Search memories by metadata fields (across all DBs).
        
        Args:
            metadata_filter: Dict of metadata fields to match
            limit: Maximum results to return (None = all)
        
        Returns:
            List of dicts with id, text, metadata, timestamp
        """
        results = self._backend.search_by_metadata(metadata_filter, limit)
        if self._user_backend:
            user_results = self._user_backend.search_by_metadata(metadata_filter, limit)
            results = results + user_results
            if limit:
                results = results[:limit]
        return results
    
    def rebuild_embeddings(self):
        """
        Rebuild embeddings for all documents with the current embedding model.
        In multi-DB mode, rebuilds both common and user databases.
        
        Returns:
            Number of documents re-embedded
        """
        from .embedding import Embedder
        embedder = Embedder()
        
        count = self._backend.rebuild_embeddings(embedder)
        if self._user_backend:
            count += self._user_backend.rebuild_embeddings(embedder)
        
        return count
    
    def close(self):
        """Close backend connection(s)"""
        self._backend.close()
        if self._user_backend:
            self._user_backend.close()
    
    @staticmethod
    def download_model():
        """Download or update the embedding model from HuggingFace"""
        return Embedder.download_model()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
