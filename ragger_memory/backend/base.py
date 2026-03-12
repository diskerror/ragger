"""
Abstract base class for memory backends
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import QUERY_LOGGING_ENABLED

logger = logging.getLogger(__name__)


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends"""
    
    def __init__(self, embedder):
        """
        Initialize backend
        
        Args:
            embedder: Embedder instance for generating embeddings
        """
        self.embedder = embedder
        
        # Cache for embeddings (avoids re-reading from storage on every search)
        self._embedding_cache: Optional[tuple] = None
        self._cache_count: int = 0
    
    @abstractmethod
    def store_raw(
        self,
        text: str,
        embedding: list,
        metadata: dict,
        timestamp: datetime
    ) -> str:
        """
        Store text with pre-computed embedding
        
        Args:
            text: The memory text
            embedding: Pre-computed embedding (as list)
            metadata: Metadata dict
            timestamp: UTC timestamp
        
        Returns:
            Memory ID (str)
        """
        pass
    
    @abstractmethod
    def load_all_embeddings(self) -> tuple[list, list, np.ndarray, list, list]:
        """
        Load all embeddings from storage
        
        Returns:
            (ids, texts, embeddings_matrix, metadata_list, timestamps)
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return number of stored memories"""
        pass
    
    @abstractmethod
    def close(self):
        """Close connection/cleanup"""
        pass
    
    def _invalidate_cache(self):
        """Invalidate the embedding cache after writes"""
        self._embedding_cache = None
        self._cache_count = 0
    
    def _load_embeddings_cached(self) -> tuple:
        """
        Load embeddings with caching.
        Subclasses should call load_all_embeddings() for actual loading.
        
        Returns:
            (ids, texts, embeddings_matrix, metadata_list, timestamps)
        """
        doc_count = self.count()
        
        if self._embedding_cache is not None and self._cache_count == doc_count:
            return self._embedding_cache
        
        logger.info(f"Loading {doc_count} embeddings from storage...")
        
        result = self.load_all_embeddings()
        
        if result[0]:  # if ids list is not empty
            embeddings = result[2]
            logger.info(f"Loaded {len(result[0])} embeddings ({embeddings.nbytes / 1024:.0f} KB)")
        
        self._embedding_cache = result
        self._cache_count = doc_count
        
        return result
    
    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Encode and store a memory
        
        Args:
            text: The memory text
            metadata: Optional metadata
        
        Returns:
            Memory ID (str)
        """
        try:
            embedding = self.embedder.encode(text).tolist()
            timestamp = datetime.now(timezone.utc)
            
            memory_id = self.store_raw(text, embedding, metadata or {}, timestamp)
            self._invalidate_cache()
            
            logger.info(f"Stored memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
        collections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Default NumPy brute-force vector search using cosine similarity.
        Override in backends that support native vector search.
        
        Args:
            query: Search query text
            limit: Maximum results to return
            min_score: Minimum similarity score (0.0-1.0)
            collections: List of collections to search. None = ["memory"] (default).
                         Use ["*"] or ["all"] to search everything.
        
        Returns:
            Dict with 'results' list and 'timing' dict
        """
        try:
            t_start = time.perf_counter()
            
            ids, texts, embeddings, metadata, timestamps = self._load_embeddings_cached()
            
            if len(ids) == 0:
                logger.info("No memories stored yet")
                return {"results": [], "timing": {}}
            
            # Filter by collection
            if collections and "*" not in collections and "all" not in collections:
                mask = np.array([
                    m.get("collection", "memory") in collections
                    for m in metadata
                ], dtype=bool)
            else:
                # Default: only "memory" collection
                if collections is None:
                    mask = np.array([
                        m.get("collection", "memory") == "memory"
                        for m in metadata
                    ], dtype=bool)
                else:
                    mask = np.ones(len(ids), dtype=bool)
            
            if not mask.any():
                logger.info(f"No memories in collections {collections}")
                return {"results": [], "timing": {"corpus_size": len(ids), "filtered_size": 0}}
            
            # Apply mask to get filtered indices
            filtered_indices = np.where(mask)[0]
            filtered_embeddings = embeddings[filtered_indices]
            
            # Generate query embedding
            t_embed_start = time.perf_counter()
            query_embedding = self.embedder.encode(query).astype(np.float32)
            t_embed_end = time.perf_counter()
            
            # Cosine similarity: dot(q, E) / (|q| * |E|)
            t_search_start = time.perf_counter()
            
            # Normalize query vector
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            # Normalize filtered embeddings
            norms = np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # avoid division by zero
            embeddings_norm = filtered_embeddings / norms
            
            # Compute similarities (matrix-vector multiply)
            similarities = embeddings_norm @ query_norm
            
            # Rank and filter — get top results from filtered set
            top_k = min(limit, len(similarities))
            ranked_local = np.argsort(similarities)[::-1][:top_k]
            # Map back to original indices
            ranked_indices = filtered_indices[ranked_local]
            ranked_scores = similarities[ranked_local]
            t_search_end = time.perf_counter()
            
            results = []
            for i, idx in enumerate(ranked_indices):
                score = float(ranked_scores[i])
                if score < min_score:
                    continue
                
                # Normalize timestamp to ISO string if it's a datetime
                ts = timestamps[idx]
                if isinstance(ts, datetime):
                    ts = ts.isoformat()
                
                results.append({
                    "id": ids[idx],
                    "text": texts[idx],
                    "score": score,
                    "metadata": metadata[idx],
                    "timestamp": ts
                })
            
            t_total = time.perf_counter()
            
            # Build timing info
            embedding_ms = round((t_embed_end - t_embed_start) * 1000, 1)
            search_ms = round((t_search_end - t_search_start) * 1000, 1)
            total_ms = round((t_total - t_start) * 1000, 1)
            
            timing = {
                "embedding_ms": embedding_ms,
                "search_ms": search_ms,
                "total_ms": total_ms,
                "corpus_size": len(ids),
                "filtered_size": int(mask.sum())
            }
            
            # Log the query (if enabled)
            if QUERY_LOGGING_ENABLED:
                scores = [r["score"] for r in results]
                top_score = scores[0] if scores else 0.0
                score_gap = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0
                
                self._log_query(
                    query=query,
                    results=results,
                    timing=timing,
                    top_score=top_score,
                    score_gap=score_gap,
                    limit=limit,
                    min_score=min_score,
                )
            
            logger.info(
                f"Search returned {len(results)} results "
                f"(embed: {embedding_ms}ms, search: {search_ms}ms, total: {total_ms}ms)"
            )
            return {"results": results, "timing": timing}
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def _log_query(
        self,
        query: str,
        results: List[Dict],
        timing: Dict,
        top_score: float,
        score_gap: float,
        limit: int,
        min_score: float
    ):
        """
        Optional query logging. Override in backends that support it, or no-op.
        """
        pass
