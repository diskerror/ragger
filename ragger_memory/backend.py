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

import os

from .bm25 import BM25Index
from .config import QUERY_LOG_ENABLED, BM25_ENABLED, BM25_WEIGHT, VECTOR_WEIGHT, BM25_K1, BM25_B, NORMALIZE_HOME_PATH, DEFAULT_COLLECTION
from . import lang

# Resolve actual home path once at import time (e.g. "/Volumes/WDBlack2")
_HOME_DIR = os.path.expanduser("~")
# Ensure it ends with / for clean replacement
_HOME_PREFIX = _HOME_DIR if _HOME_DIR.endswith("/") else _HOME_DIR + "/"

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
        
        # BM25 index (built alongside embedding cache)
        self._bm25: Optional[BM25Index] = None
    
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
        self._bm25 = None
    
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
        
        logger.info(lang.MSG_LOADING_EMBEDDINGS.format(count=doc_count))
        
        result = self.load_all_embeddings()
        
        if result[0]:  # if ids list is not empty
            embeddings = result[2]
            logger.info(lang.MSG_LOADED_EMBEDDINGS.format(count=len(result[0]), size_kb=embeddings.nbytes / 1024))
        
        self._embedding_cache = result
        self._cache_count = doc_count
        
        # Build BM25 index — from persisted table if available, else from texts
        if BM25_ENABLED and result[1]:
            self._bm25 = BM25Index(k1=BM25_K1, b=BM25_B)
            if not self._load_bm25_from_storage(result[0]):
                self._bm25.build(result[1])  # result[1] = texts
        
        return result
    
    @staticmethod
    def _normalize_paths(text: str) -> str:
        """Replace home directory path with ~ in text."""
        if NORMALIZE_HOME_PATH and _HOME_PREFIX:
            text = text.replace(_HOME_PREFIX, "~/")
        return text
    
    @staticmethod
    def _normalize_metadata(metadata: dict) -> dict:
        """Replace home directory paths with ~ in metadata string values."""
        if not NORMALIZE_HOME_PATH or not _HOME_PREFIX:
            return metadata
        
        normalized = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                normalized[key] = value.replace(_HOME_PREFIX, "~/")
            elif isinstance(value, list):
                normalized[key] = [
                    v.replace(_HOME_PREFIX, "~/") if isinstance(v, str) else v
                    for v in value
                ]
            else:
                normalized[key] = value
        return normalized
    
    # Dimensions must match the embedding model (MiniLM-L6 = 384)
    EMBEDDING_DIMS = 384

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None,
              defer_embedding: bool = False) -> str:
        """
        Encode and store a memory.

        Args:
            text: The memory text
            metadata: Optional metadata
            defer_embedding: If True, store with a zero-vector and
                             compute the real embedding asynchronously.
                             The record is immediately searchable by BM25
                             (keywords) but not by vector similarity until
                             the embedding is backfilled.

        Returns:
            Memory ID (str)
        """
        try:
            text = self._normalize_paths(text)
            meta = self._normalize_metadata(metadata or {})

            # Ensure every record has a collection
            if 'collection' not in meta:
                meta['collection'] = DEFAULT_COLLECTION

            timestamp = datetime.now(timezone.utc)

            if defer_embedding:
                # Zero-vector placeholder — backfilled by _embed_deferred()
                embedding = [0.0] * self.EMBEDDING_DIMS
                memory_id = self.store_raw(text, embedding, meta, timestamp)
                self._invalidate_cache()
                logger.info(f"Stored memory {memory_id} (embedding deferred)")
                # Kick off background embedding
                self._embed_deferred(int(memory_id), text)
            else:
                embedding = self.embedder.encode(text).tolist()
                memory_id = self.store_raw(text, embedding, meta, timestamp)
                self._invalidate_cache()
                logger.info(lang.MSG_STORED_MEMORY.format(id=memory_id))

            return memory_id

        except Exception as e:
            logger.error(lang.ERR_STORE_FAILED.format(error=e))
            raise

    def _embed_deferred(self, memory_id: int, text: str):
        """Compute embedding in a background thread and update the record."""
        import threading

        # Capture the DB path — the thread opens its own connection
        db_path = self._get_db_path()

        def _do_embed():
            try:
                embedding = self.embedder.encode(text)
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                self._update_embedding_by_path(db_path, memory_id, embedding_blob)
                self._invalidate_cache()
                logger.info(f"Deferred embedding complete for memory {memory_id}")
            except Exception as e:
                logger.warning(f"Deferred embedding failed for {memory_id}: {e}")

        threading.Thread(target=_do_embed, daemon=True).start()

    @abstractmethod
    def _get_db_path(self) -> str:
        """Return the database file path (for thread-safe deferred operations)."""
        ...

    @staticmethod
    def _update_embedding_by_path(db_path: str, memory_id: int, embedding_blob: bytes):
        """Update embedding using a fresh connection (thread-safe)."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE memories SET embedding = ? WHERE id = ?",
                     (embedding_blob, memory_id))
        conn.commit()
        conn.close()

    @abstractmethod
    def _update_embedding(self, memory_id: int, embedding_blob: bytes):
        """Update the embedding blob for an existing record."""
        ...
    
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
            collections: List of collections to search. None = all collections.
                         Use explicit list to narrow, e.g. ["memory"] or ["sibelius"].
        
        Returns:
            Dict with 'results' list and 'timing' dict
        """
        try:
            t_start = time.perf_counter()
            
            ids, texts, embeddings, metadata, timestamps = self._load_embeddings_cached()
            
            if len(ids) == 0:
                logger.info(lang.MSG_NO_MEMORIES)
                return {"results": [], "timing": {}}
            
            # Filter by collection
            # Default (None) = search all collections.
            # Pass explicit list to narrow, e.g. ["memory"] or ["sibelius"].
            if collections and "*" not in collections and "all" not in collections:
                mask = np.array([
                    m.get("collection", DEFAULT_COLLECTION) in collections
                    for m in metadata
                ], dtype=bool)
            else:
                # None or ["*"]/["all"] → search everything
                mask = np.ones(len(ids), dtype=bool)
            
            if not mask.any():
                logger.info(lang.MSG_NO_MEMORIES_IN_COLLECTIONS.format(collections=collections))
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
            
            # Hybrid ranking: combine vector cosine + BM25
            if BM25_ENABLED and self._bm25 is not None and self._bm25.is_built:
                bm25_scores = self._bm25.score(query, filtered_indices)
                
                # Normalize both score arrays to [0, 1] for fair blending
                vec_min, vec_max = similarities.min(), similarities.max()
                if vec_max > vec_min:
                    vec_norm = (similarities - vec_min) / (vec_max - vec_min)
                else:
                    vec_norm = similarities
                
                bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
                if bm25_max > bm25_min:
                    bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
                else:
                    bm25_norm = bm25_scores
                
                # Normalize weights: A/(A+B) pattern — any values work, don't need to sum to 1.0
                w_total = VECTOR_WEIGHT + BM25_WEIGHT
                w_vec = VECTOR_WEIGHT / w_total if w_total > 0 else 0.5
                w_bm25 = BM25_WEIGHT / w_total if w_total > 0 else 0.5
                combined = w_vec * vec_norm + w_bm25 * bm25_norm
            else:
                combined = similarities
            
            # Rank and filter — get top results from filtered set
            top_k = min(limit, len(combined))
            ranked_local = np.argsort(combined)[::-1][:top_k]
            # Map back to original indices
            ranked_indices = filtered_indices[ranked_local]
            ranked_scores = similarities[ranked_local]  # Report raw cosine as the score
            t_search_end = time.perf_counter()
            
            results = []
            for i, idx in enumerate(ranked_indices):
                score = float(ranked_scores[i])
                if score < min_score:
                    continue
                
                # Normalize timestamp to ISO string if it's a datetime
                ts = timestamps[idx]
                if isinstance(ts, datetime):
                    ts = ts.strftime('%Y-%m-%dT%H:%M:%SZ')
                
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
            if QUERY_LOG_ENABLED:
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
            
            # Track usage of returned results
            self._track_search_usage(results)
            
            logger.info(
                lang.MSG_SEARCH_RESULTS.format(count=len(results), ms=total_ms)
            )
            return {"results": results, "timing": timing}
            
        except Exception as e:
            logger.error(lang.ERR_SEARCH_FAILED.format(error=e))
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
    
    def _load_bm25_from_storage(self, ids: list) -> bool:
        """
        Optional: load BM25 index from persistent storage.
        Override in backends that support it.
        
        Args:
            ids: List of document IDs (same order as embeddings)
        
        Returns:
            True if index was loaded from storage, False to fall back to build()
        """
        return False
    
    def _track_search_usage(self, results: List[Dict]):
        """
        Optional usage tracking. Override in backends that support it, or no-op.
        Called after every search with the returned results.
        """
        pass
