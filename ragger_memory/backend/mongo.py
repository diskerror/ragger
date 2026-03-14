"""
MongoDB backend for Ragger Memory
"""
import logging
from datetime import datetime
from typing import List, Dict

import numpy as np
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError

from .base import MemoryBackend
from ..config import MONGODB_DB_NAME, MONGODB_COLLECTION, MONGODB_QUERY_LOG_COLLECTION, MONGODB_USAGE_COLLECTION, MONGODB_URI, USAGE_TRACKING_ENABLED

logger = logging.getLogger(__name__)


class MongoBackend(MemoryBackend):
    """MongoDB storage backend with Python vector search"""
    
    def __init__(self, embedder, uri: str = None):
        """
        Initialize MongoDB backend
        
        Args:
            embedder: Embedder instance
            uri: MongoDB connection URI (defaults to config.MONGODB_URI)
        """
        super().__init__(embedder)
        
        self.uri = uri or MONGODB_URI
        self.client = None
        self.db = None
        self.collection = None
        self.query_log = None
        self.usage_col = None
        
        self._connect()
        self._ensure_indexes()
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[MONGODB_DB_NAME]
            self.collection = self.db[MONGODB_COLLECTION]
            self.query_log = self.db[MONGODB_QUERY_LOG_COLLECTION]
            self.usage_col = self.db[MONGODB_USAGE_COLLECTION]
            logger.info(f"Connected to MongoDB: {MONGODB_DB_NAME}.{MONGODB_COLLECTION}")
        except PyMongoError as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    def _ensure_indexes(self):
        """Create MongoDB indexes for efficient querying"""
        try:
            self.collection.create_index("timestamp")
            self.collection.create_index("metadata.source")
            self.collection.create_index("metadata.category")
            self.query_log.create_index("timestamp")
            self.usage_col.create_index("memory_id")
            self.usage_col.create_index("timestamp")
            logger.info("MongoDB indexes ensured")
        except PyMongoError as e:
            logger.warning(f"Could not create indexes: {e}")
    
    def store_raw(
        self,
        text: str,
        embedding: list,
        metadata: dict,
        timestamp: datetime
    ) -> str:
        """Store text with pre-computed embedding in MongoDB"""
        try:
            doc = {
                "text": text,
                "embedding": embedding,
                "timestamp": timestamp,
                "metadata": metadata
            }
            
            result = self.collection.insert_one(doc)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store in MongoDB: {e}")
            raise
    
    def load_all_embeddings(self) -> tuple[list, list, np.ndarray, list, list]:
        """Load all embeddings from MongoDB"""
        docs = list(self.collection.find(
            {},
            {"_id": 1, "text": 1, "embedding": 1, "metadata": 1, "timestamp": 1}
        ))
        
        if not docs:
            return ([], [], np.array([]), [], [])
        
        ids = [str(d["_id"]) for d in docs]
        texts = [d["text"] for d in docs]
        embeddings = np.array([d["embedding"] for d in docs], dtype=np.float32)
        metadata = [d.get("metadata", {}) for d in docs]
        timestamps = [d.get("timestamp") for d in docs]
        
        return (ids, texts, embeddings, metadata, timestamps)
    
    def count(self) -> int:
        """Return number of stored memories"""
        return self.collection.estimated_document_count()
    
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
        """Log a search query to the MongoDB query_log collection"""
        try:
            from datetime import timezone
            log_entry = {
                "timestamp": datetime.now(timezone.utc),
                "query": query,
                "limit": limit,
                "min_score": min_score,
                "num_results": len(results),
                "top_score": round(top_score, 4),
                "score_gap": round(score_gap, 4),
                "below_threshold": top_score < 0.4,
                "results": [
                    {
                        "chunk_id": r["id"],
                        "score": round(r["score"], 4),
                        "source": r["metadata"].get("source", ""),
                        "chunk_size": len(r["text"])
                    }
                    for r in results
                ],
                "timing": timing,
                "feedback": None
            }
            self.query_log.insert_one(log_entry)
        except Exception as e:
            # Don't let logging failures break search
            logger.warning(f"Failed to log query: {e}")
    
    def _track_search_usage(self, results: List[Dict]):
        """Track usage of memories returned by search"""
        if not USAGE_TRACKING_ENABLED or not results:
            return
        try:
            from datetime import timezone
            timestamp = datetime.now(timezone.utc)
            docs = [
                {"memory_id": r["id"], "timestamp": timestamp}
                for r in results
            ]
            self.usage_col.insert_many(docs)
        except Exception as e:
            logger.warning(f"Failed to track usage: {e}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
