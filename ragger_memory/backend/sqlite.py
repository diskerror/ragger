"""
SQLite backend for Ragger Memory
"""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import numpy as np

from .base import MemoryBackend
from ..config import SQLITE_PATH, SQLITE_MEMORIES_TABLE, SQLITE_QUERY_LOG_TABLE, SQLITE_USAGE_TABLE, USAGE_TRACKING_ENABLED

logger = logging.getLogger(__name__)


class SqliteBackend(MemoryBackend):
    """SQLite storage backend with Python vector search"""
    
    def __init__(self, embedder, db_path: str = None):
        """
        Initialize SQLite backend
        
        Args:
            embedder: Embedder instance
            db_path: Path to SQLite database file (defaults to config.SQLITE_PATH)
        """
        super().__init__(embedder)
        
        # Expand ~ in path
        path_str = db_path or SQLITE_PATH
        self.db_path = Path(path_str).expanduser()
        self._memories_table = SQLITE_MEMORIES_TABLE
        self._query_log_table = SQLITE_QUERY_LOG_TABLE
        self._usage_table = SQLITE_USAGE_TABLE
        
        # Auto-create parent directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self._connect()
        self._create_schema()
    
    def _connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info(f"Connected to SQLite: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"SQLite connection failed: {e}")
            raise
    
    def _create_schema(self):
        """Create SQLite schema if it doesn't exist"""
        try:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._memories_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._query_log_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    results TEXT,
                    timing TEXT
                )
            """)
            
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._usage_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL
                        REFERENCES {self._memories_table}(id)
                        ON DELETE CASCADE
                        ON UPDATE CASCADE,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Index for fast lookups by memory_id
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._usage_table}_memory_id
                ON {self._usage_table} (memory_id)
            """)
            
            # Migrate: if existing table lacks FK, rebuild it
            self._migrate_usage_fk()
            
            self.conn.commit()
            logger.info("SQLite schema ensured")
        except sqlite3.Error as e:
            logger.warning(f"Could not create schema: {e}")
    
    def _migrate_usage_fk(self):
        """Rebuild memory_usage table with FK constraint if it lacks one."""
        try:
            # Check if the table has a FK already
            fk_info = self.conn.execute(
                f"PRAGMA foreign_key_list({self._usage_table})"
            ).fetchall()
            if fk_info:
                return  # FK already exists
            
            # Check if the table has any rows worth migrating
            count = self.conn.execute(
                f"SELECT COUNT(*) FROM {self._usage_table}"
            ).fetchone()[0]
            
            tmp = f"{self._usage_table}_old"
            self.conn.execute(f"DROP INDEX IF EXISTS idx_{self._usage_table}_memory_id")
            self.conn.execute(f"ALTER TABLE {self._usage_table} RENAME TO {tmp}")
            self.conn.execute(f"""
                CREATE TABLE {self._usage_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL
                        REFERENCES {self._memories_table}(id)
                        ON DELETE CASCADE
                        ON UPDATE CASCADE,
                    timestamp TEXT NOT NULL
                )
            """)
            self.conn.execute(f"""
                CREATE INDEX idx_{self._usage_table}_memory_id
                ON {self._usage_table} (memory_id)
            """)
            if count > 0:
                self.conn.execute(f"""
                    INSERT INTO {self._usage_table} (id, memory_id, timestamp)
                    SELECT id, memory_id, timestamp FROM {tmp}
                    WHERE memory_id IN (SELECT id FROM {self._memories_table})
                """)
            self.conn.execute(f"DROP TABLE {tmp}")
            self.conn.commit()
            logger.info(f"Migrated {self._usage_table}: added FK constraint ({count} rows)")
        except sqlite3.Error as e:
            logger.warning(f"FK migration skipped: {e}")
    
    def store_raw(
        self,
        text: str,
        embedding: list,
        metadata: dict,
        timestamp: datetime
    ) -> str:
        """Store text with pre-computed embedding in SQLite"""
        try:
            embedding_array = np.array(embedding, dtype=np.float32)
            embedding_blob = embedding_array.tobytes()
            metadata_json = json.dumps(metadata)
            timestamp_str = timestamp.isoformat()
            
            cursor = self.conn.execute(
                f"INSERT INTO {self._memories_table} (text, embedding, metadata, timestamp) "
                f"VALUES (?, ?, ?, ?)",
                (text, embedding_blob, metadata_json, timestamp_str)
            )
            
            self.conn.commit()
            return str(cursor.lastrowid)
            
        except Exception as e:
            logger.error(f"Failed to store in SQLite: {e}")
            raise
    
    def load_all_embeddings(self) -> tuple[list, list, np.ndarray, list, list]:
        """Load all embeddings from SQLite"""
        try:
            cursor = self.conn.execute(
                f"SELECT id, text, embedding, metadata, timestamp FROM {self._memories_table}"
            )
            rows = cursor.fetchall()
            
            if not rows:
                return ([], [], np.array([]), [], [])
            
            ids = [str(row[0]) for row in rows]
            texts = [row[1] for row in rows]
            
            embeddings = []
            for row in rows:
                embedding = np.frombuffer(row[2], dtype=np.float32)
                embeddings.append(embedding)
            embeddings = np.array(embeddings, dtype=np.float32)
            
            metadata = []
            for row in rows:
                meta = json.loads(row[3]) if row[3] else {}
                metadata.append(meta)
            
            timestamps = [row[4] for row in rows]
            
            return (ids, texts, embeddings, metadata, timestamps)
            
        except Exception as e:
            logger.error(f"Failed to load from SQLite: {e}")
            raise
    
    def count(self) -> int:
        """Return number of stored memories"""
        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {self._memories_table}")
        return cursor.fetchone()[0]
    
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
        """Log a search query to the SQLite query_log table"""
        try:
            log_entry = {
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
                "feedback": None
            }
            
            timestamp = datetime.now(timezone.utc).isoformat()
            results_json = json.dumps(log_entry)
            timing_json = json.dumps(timing)
            
            self.conn.execute(
                f"INSERT INTO {self._query_log_table} (timestamp, query, results, timing) "
                f"VALUES (?, ?, ?, ?)",
                (timestamp, query, results_json, timing_json)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")
    
    def _track_search_usage(self, results: List[Dict]):
        """Track usage of memories returned by search"""
        if results:
            self._track_usage([r["id"] for r in results])
    
    def _track_usage(self, memory_ids: list):
        """Record usage of returned memories"""
        if not USAGE_TRACKING_ENABLED or not memory_ids:
            return
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            self.conn.executemany(
                f"INSERT INTO {self._usage_table} (memory_id, timestamp) VALUES (?, ?)",
                [(int(mid), timestamp) for mid in memory_ids]
            )
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Failed to track usage: {e}")
    
    def close(self):
        """Close SQLite connection"""
        if self.conn:
            self.conn.close()
            logger.info("SQLite connection closed")
