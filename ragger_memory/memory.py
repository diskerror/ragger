"""
RaggerMemory - MongoDB-based vector memory store

Stores text with embeddings in MongoDB. Performs vector similarity
search in Python using NumPy (no mongot/Atlas Search required).
Works with any MongoDB version (including MacPorts 6.0).
"""

import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError
from sentence_transformers import SentenceTransformer

from .config import (
    MONGODB_URI, DB_NAME, COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    MODEL_CACHE_DIR, MODEL_LOCAL_PATH
)

logger = logging.getLogger(__name__)


class RaggerMemory:
    """MongoDB-based memory store with Python vector search"""
    
    def __init__(self, uri: Optional[str] = None):
        """
        Initialize memory store
        
        Args:
            uri: MongoDB connection URI (defaults to config.MONGODB_URI)
        """
        self.uri = uri or MONGODB_URI
        self.client = None
        self.db = None
        self.collection = None
        self.model = None
        
        # Cache for embeddings (avoids re-reading from MongoDB on every search)
        self._embedding_cache = None
        self._cache_count = 0
        
        self._connect()
        self._load_model()
        self._ensure_indexes()
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            logger.info(f"Connected to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        except PyMongoError as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    def _load_model(self):
        """Load sentence transformer model from local snapshot only (no network)"""
        if not MODEL_LOCAL_PATH:
            raise FileNotFoundError(
                f"Model '{EMBEDDING_MODEL}' not found in {MODEL_CACHE_DIR}\n"
                f"Run './ragger.py --update-model' to download it."
            )
        try:
            # Force CPU — MPS (Metal) can throw I/O errors in background
            # processes. CPU is fast enough for 384-dim embeddings.
            self.model = SentenceTransformer(MODEL_LOCAL_PATH, device="cpu")
            logger.info(f"Embedding model loaded: {EMBEDDING_MODEL} (cpu)")
        except Exception as e:
            logger.error(f"Failed to load embedding model from {MODEL_LOCAL_PATH}: {e}")
            raise
    
    @staticmethod
    def download_model():
        """Download or update the embedding model from HuggingFace"""
        logger.info(f"Downloading model: {EMBEDDING_MODEL}")
        logger.info(f"Cache directory: {MODEL_CACHE_DIR}")
        model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_DIR)
        logger.info(f"Model ready: {EMBEDDING_MODEL}")
        return model
    
    def _ensure_indexes(self):
        """Create MongoDB indexes for efficient querying"""
        try:
            self.collection.create_index("timestamp")
            self.collection.create_index("metadata.source")
            self.collection.create_index("metadata.category")
            logger.info("MongoDB indexes ensured")
        except PyMongoError as e:
            logger.warning(f"Could not create indexes: {e}")
    
    def _invalidate_cache(self):
        """Invalidate the embedding cache after writes"""
        self._embedding_cache = None
        self._cache_count = 0
    
    def _load_embeddings(self) -> tuple:
        """
        Load all embeddings from MongoDB into NumPy arrays for search.
        Caches results until invalidated by a write.
        
        Returns:
            (ids, texts, embeddings_matrix, metadata_list, timestamps)
        """
        doc_count = self.collection.estimated_document_count()
        
        if self._embedding_cache is not None and self._cache_count == doc_count:
            return self._embedding_cache
        
        logger.info(f"Loading {doc_count} embeddings from MongoDB...")
        
        docs = list(self.collection.find(
            {},
            {"_id": 1, "text": 1, "embedding": 1, "metadata": 1, "timestamp": 1}
        ))
        
        if not docs:
            empty = ([], [], np.array([]), [], [])
            self._embedding_cache = empty
            self._cache_count = 0
            return empty
        
        ids = [str(d["_id"]) for d in docs]
        texts = [d["text"] for d in docs]
        embeddings = np.array([d["embedding"] for d in docs], dtype=np.float32)
        metadata = [d.get("metadata", {}) for d in docs]
        timestamps = [d.get("timestamp") for d in docs]
        
        result = (ids, texts, embeddings, metadata, timestamps)
        self._embedding_cache = result
        self._cache_count = doc_count
        
        logger.info(f"Loaded {len(docs)} embeddings ({embeddings.nbytes / 1024:.0f} KB)")
        return result
    
    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a memory with vector embedding
        
        Args:
            text: The memory text
            metadata: Optional metadata (source, tags, etc.)
        
        Returns:
            Memory ID (str)
        """
        try:
            embedding = self.model.encode(text).tolist()
            
            doc = {
                "text": text,
                "embedding": embedding,
                "timestamp": datetime.now(timezone.utc),
                "metadata": metadata or {}
            }
            
            result = self.collection.insert_one(doc)
            self._invalidate_cache()
            
            memory_id = str(result.inserted_id)
            logger.info(f"Stored memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Vector search for relevant memories using cosine similarity.
        
        Args:
            query: Search query text
            limit: Maximum results to return
            min_score: Minimum similarity score (0.0-1.0)
        
        Returns:
            List of matching memories with scores, sorted by relevance
        """
        try:
            ids, texts, embeddings, metadata, timestamps = self._load_embeddings()
            
            if len(ids) == 0:
                logger.info("No memories stored yet")
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode(query).astype(np.float32)
            
            # Cosine similarity: dot(q, E) / (|q| * |E|)
            # Normalize query vector
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            # Normalize all stored embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # avoid division by zero
            embeddings_norm = embeddings / norms
            
            # Compute similarities (matrix-vector multiply)
            similarities = embeddings_norm @ query_norm
            
            # Rank and filter
            ranked_indices = np.argsort(similarities)[::-1][:limit]
            
            results = []
            for idx in ranked_indices:
                score = float(similarities[idx])
                if score < min_score:
                    continue
                results.append({
                    "text": texts[idx],
                    "score": score,
                    "metadata": metadata[idx],
                    "timestamp": timestamps[idx]
                })
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def count(self) -> int:
        """Return number of stored memories"""
        return self.collection.estimated_document_count()
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
