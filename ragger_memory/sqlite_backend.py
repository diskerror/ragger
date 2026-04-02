"""
SQLite backend for Ragger Memory
"""
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict

# Dedicated query logger (search queries, scores, timing → query.log)
query_logger = logging.getLogger('ragger_memory.query')

import numpy as np

from .backend import MemoryBackend
from .bm25 import tokenize
from .config import SQLITE_PATH, SQLITE_MEMORIES_TABLE, SQLITE_USAGE_TABLE, USAGE_TRACKING_ENABLED

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
            # Users table — token-based auth, maps token → user
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    token_hash TEXT NOT NULL,
                    -- is_admin removed: sudo is the admin gate
                    created TEXT NOT NULL,
                    modified TEXT NOT NULL
                )
            """)

            # Trigger: update modified timestamp on users
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS users_modified
                AFTER UPDATE ON users
                BEGIN
                    UPDATE users SET modified = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                    WHERE id = NEW.id;
                END
            """)

            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._memories_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL,
                    user_id INTEGER REFERENCES users(id)
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
            
            # BM25 index table — persisted token frequencies per document
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS bm25_index (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL
                        REFERENCES memories(id)
                        ON DELETE CASCADE
                        ON UPDATE CASCADE,
                    token     TEXT NOT NULL,
                    term_freq INTEGER NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bm25_memory_id
                ON bm25_index (memory_id)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bm25_token
                ON bm25_index (token)
            """)
            
            # Migrations
            self._migrate_usage_fk()
            self._migrate_add_user_id()
            self._migrate_dedicated_columns()
            self._migrate_add_token_rotated_at()
            self._migrate_add_preferred_model()
            self._migrate_add_password_hash()
            self._migrate_add_web_sessions()
            self._migrate_add_chat_sessions()
            
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
    
    def _migrate_add_user_id(self):
        """Add user_id column to memories table if it doesn't exist."""
        try:
            cols = [row[1] for row in self.conn.execute(
                f"PRAGMA table_info({self._memories_table})"
            ).fetchall()]
            if "user_id" not in cols:
                self.conn.execute(
                    f"ALTER TABLE {self._memories_table} ADD COLUMN "
                    f"user_id INTEGER REFERENCES users(id)"
                )
                self.conn.commit()
                logger.info(f"Migrated {self._memories_table}: added user_id column")
        except sqlite3.Error as e:
            logger.warning(f"user_id migration skipped: {e}")

    def _migrate_dedicated_columns(self):
        """Add collection, category, tags columns and backfill from JSON metadata."""
        from .migrations import migrate_add_dedicated_columns
        try:
            migrate_add_dedicated_columns(self.conn, self._memories_table)
        except sqlite3.Error as e:
            logger.warning(f"dedicated columns migration skipped: {e}")

    def _migrate_add_token_rotated_at(self):
        """Add token_rotated_at column to users table if it doesn't exist."""
        try:
            cols = [row[1] for row in self.conn.execute(
                "PRAGMA table_info(users)"
            ).fetchall()]
            if "token_rotated_at" not in cols:
                # Add column with NULL default (will be set on first rotation)
                self.conn.execute(
                    "ALTER TABLE users ADD COLUMN token_rotated_at TEXT"
                )
                # Initialize existing users to current time
                now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                self.conn.execute(
                    "UPDATE users SET token_rotated_at = ?",
                    (now,)
                )
                self.conn.commit()
                logger.info("Migrated users: added token_rotated_at column")
        except sqlite3.Error as e:
            logger.warning(f"token_rotated_at migration skipped: {e}")

    def _migrate_add_preferred_model(self):
        """Add preferred_model column to users table if it doesn't exist."""
        try:
            cols = [row[1] for row in self.conn.execute(
                "PRAGMA table_info(users)"
            ).fetchall()]
            if "preferred_model" not in cols:
                self.conn.execute(
                    "ALTER TABLE users ADD COLUMN preferred_model TEXT"
                )
                self.conn.commit()
                logger.info("Migrated users: added preferred_model column")
        except sqlite3.Error as e:
            logger.warning(f"preferred_model migration skipped: {e}")

    def _migrate_add_password_hash(self):
        """Add password_hash column to users table if it doesn't exist."""
        try:
            cols = [row[1] for row in self.conn.execute(
                "PRAGMA table_info(users)"
            ).fetchall()]
            if "password_hash" not in cols:
                self.conn.execute(
                    "ALTER TABLE users ADD COLUMN password_hash TEXT"
                )
                self.conn.commit()
                logger.info("Migrated users: added password_hash column")
        except sqlite3.Error as e:
            logger.warning(f"password_hash migration skipped: {e}")

    def _migrate_add_web_sessions(self):
        """Create web_sessions table if it doesn't exist."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS web_sessions (
                    token TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    created TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    expires TEXT NOT NULL
                )
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"web_sessions migration skipped: {e}")

    def _migrate_add_chat_sessions(self):
        """Create chat_sessions table if it doesn't exist."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    web_token TEXT,
                    username TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    created TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    FOREIGN KEY (web_token) REFERENCES web_sessions(token) ON DELETE SET NULL
                )
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"chat_sessions migration skipped: {e}")

    # --- Web sessions ---

    def create_web_session(self, token: str, username: str, user_id: int,
                           ttl_seconds: int = 86400) -> None:
        """Store a web session token."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=ttl_seconds)
        self.conn.execute(
            "INSERT OR REPLACE INTO web_sessions (token, username, user_id, created, expires) "
            "VALUES (?, ?, ?, ?, ?)",
            (token, username, user_id,
             now.strftime('%Y-%m-%dT%H:%M:%SZ'),
             expires.strftime('%Y-%m-%dT%H:%M:%SZ'))
        )
        self.conn.commit()

    def get_web_session(self, token: str) -> dict | None:
        """Look up a web session. Returns {username, user_id} or None if expired/missing."""
        row = self.conn.execute(
            "SELECT username, user_id, expires FROM web_sessions WHERE token = ?",
            (token,)
        ).fetchone()
        if not row:
            return None
        expires = datetime.strptime(row[2], '%Y-%m-%dT%H:%M:%SZ').replace(
            tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires:
            self.conn.execute("DELETE FROM web_sessions WHERE token = ?", (token,))
            self.conn.commit()
            return None
        return {"username": row[0], "user_id": row[1]}

    def delete_web_session(self, token: str) -> None:
        """Remove a web session."""
        self.conn.execute("DELETE FROM web_sessions WHERE token = ?", (token,))
        self.conn.commit()

    def cleanup_web_sessions(self) -> int:
        """Remove all expired web sessions. Returns count deleted."""
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        cursor = self.conn.execute(
            "DELETE FROM web_sessions WHERE expires < ?", (now,))
        self.conn.commit()
        return cursor.rowcount

    # --- Chat sessions ---

    def save_chat_session(self, session_id: str, username: str, messages: list,
                         web_token: str = None) -> None:
        """Save or update a chat session."""
        import json as json_lib
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        messages_json = json_lib.dumps(messages)
        
        # Check if session exists
        existing = self.conn.execute(
            "SELECT session_id FROM chat_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        
        if existing:
            # Update existing
            self.conn.execute(
                "UPDATE chat_sessions SET messages = ?, updated = ?, web_token = ? "
                "WHERE session_id = ?",
                (messages_json, now, web_token, session_id)
            )
        else:
            # Insert new
            self.conn.execute(
                "INSERT INTO chat_sessions (session_id, username, messages, web_token, created, updated) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, username, messages_json, web_token, now, now)
            )
        self.conn.commit()

    def get_chat_session(self, session_id: str) -> dict | None:
        """Retrieve a chat session. Returns {session_id, username, messages, created, updated} or None."""
        import json as json_lib
        row = self.conn.execute(
            "SELECT session_id, username, messages, created, updated FROM chat_sessions "
            "WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "session_id": row[0],
            "username": row[1],
            "messages": json_lib.loads(row[2]),
            "created": row[3],
            "updated": row[4]
        }

    def delete_chat_session(self, session_id: str) -> None:
        """Delete a chat session."""
        self.conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        self.conn.commit()

    def list_user_chat_sessions(self, username: str, limit: int = 10) -> list:
        """List recent chat sessions for a user."""
        import json as json_lib
        rows = self.conn.execute(
            "SELECT session_id, messages, created, updated FROM chat_sessions "
            "WHERE username = ? ORDER BY updated DESC LIMIT ?",
            (username, limit)
        ).fetchall()
        return [
            {
                "session_id": row[0],
                "messages": json_lib.loads(row[1]),
                "created": row[2],
                "updated": row[3]
            }
            for row in rows
        ]

    # --- User management ---

    def create_user(self, username: str, token_hash: str) -> int:
        """Create a user. Returns user id."""
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        cursor = self.conn.execute(
            "INSERT INTO users (username, token_hash, created, modified, token_rotated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, token_hash, now, now, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_user_by_token_hash(self, token_hash: str) -> dict | None:
        """Look up user by token hash. Returns dict or None."""
        row = self.conn.execute(
            "SELECT id, username FROM users WHERE token_hash = ?",
            (token_hash,)
        ).fetchone()
        if row:
            return {"id": row[0], "username": row[1]}
        return None

    def get_user_by_username(self, username: str) -> dict | None:
        """Look up user by username. Returns dict or None."""
        row = self.conn.execute(
            "SELECT id, username, token_hash FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        if row:
            return {"id": row[0], "username": row[1], "token_hash": row[2]}
        return None

    def update_user_token(self, username: str, token_hash: str):
        """Update a user's token hash."""
        self.conn.execute(
            "UPDATE users SET token_hash = ? WHERE username = ?",
            (token_hash, username)
        )
        self.conn.commit()

    def update_user_token_rotated_at(self, username: str, timestamp: str):
        """Update a user's token_rotated_at timestamp."""
        self.conn.execute(
            "UPDATE users SET token_rotated_at = ? WHERE username = ?",
            (timestamp, username)
        )
        self.conn.commit()

    def get_user_token_rotated_at(self, username: str) -> str | None:
        """Get a user's token_rotated_at timestamp."""
        row = self.conn.execute(
            "SELECT token_rotated_at FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        return row[0] if row else None

    def update_user_preferred_model(self, username: str, model: str | None):
        """Update a user's preferred model."""
        self.conn.execute(
            "UPDATE users SET preferred_model = ? WHERE username = ?",
            (model, username)
        )
        self.conn.commit()

    def get_user_preferred_model(self, username: str) -> str | None:
        """Get a user's preferred model."""
        row = self.conn.execute(
            "SELECT preferred_model FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        return row[0] if row else None

    def set_user_password(self, username: str, password_hash: str | None):
        """Set a user's password hash. None or empty string clears it."""
        self.conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (password_hash if password_hash else None, username)
        )
        self.conn.commit()

    def get_user_password(self, username: str) -> str | None:
        """Get a user's password hash. Returns None if not set."""
        row = self.conn.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        return row[0] if row else None

    def get_user_count(self) -> int:
        """Return total number of users in the users table."""
        row = self.conn.execute("SELECT count(*) FROM users").fetchone()
        return row[0] if row else 0

    def delete_user(self, username: str):
        """Remove a user from the users table."""
        self.conn.execute(
            "DELETE FROM users WHERE username = ?",
            (username,)
        )
        self.conn.commit()

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
            
            # Extract dedicated columns from metadata
            collection = metadata.pop("collection", "memory")
            category = metadata.pop("category", "")
            tags_val = metadata.pop("tags", "")
            if isinstance(tags_val, list):
                tag_list = [str(t) for t in tags_val]
            elif isinstance(tags_val, str) and tags_val:
                tag_list = [t.strip() for t in tags_val.split(",") if t.strip()]
            else:
                tag_list = []
            
            # Convert boolean flags to tags
            if metadata.pop("keep", False):
                if "keep" not in tag_list:
                    tag_list.append("keep")
            if metadata.pop("bad", False):
                if "bad" not in tag_list:
                    tag_list.append("bad")
            
            tags_str = ",".join(tag_list)
            
            metadata_json = json.dumps(metadata) if metadata else None
            timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            cursor = self.conn.execute(
                f"INSERT INTO {self._memories_table} "
                f"(text, embedding, metadata, timestamp, collection, category, tags) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?)",
                (text, embedding_blob, metadata_json, timestamp_str,
                 collection, category, tags_str)
            )
            
            memory_id = cursor.lastrowid
            
            # Index BM25 tokens for this document
            self._index_bm25_tokens(memory_id, text)
            
            self.conn.commit()
            return str(memory_id)
            
        except Exception as e:
            logger.error(f"Failed to store in SQLite: {e}")
            raise
    
    def _get_db_path(self) -> str:
        """Return the database file path."""
        return str(self.db_path)

    def _update_embedding(self, memory_id: int, embedding_blob: bytes):
        """Update the embedding blob for an existing record."""
        self.conn.execute(
            f"UPDATE {self._memories_table} SET embedding = ? WHERE id = ?",
            (embedding_blob, memory_id)
        )
        self.conn.commit()

    def _index_bm25_tokens(self, memory_id: int, text: str):
        """Tokenize text and store term frequencies in bm25_index table."""
        tokens = tokenize(text)
        if not tokens:
            return
        
        # Count term frequencies
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        self.conn.executemany(
            "INSERT INTO bm25_index (memory_id, token, term_freq) VALUES (?, ?, ?)",
            [(memory_id, token, freq) for token, freq in tf.items()]
        )
    
    def _reconstruct_metadata(self, row_meta_json: str,
                              collection: str, category: str, tags: str) -> dict:
        """Reconstruct full metadata dict from dedicated columns + JSON remainder."""
        meta = json.loads(row_meta_json) if row_meta_json else {}
        meta["collection"] = collection or "memory"
        if category:
            meta["category"] = category
        if tags:
            meta["tags"] = tags.split(",")
        return meta

    def load_all_embeddings(self) -> tuple[list, list, np.ndarray, list, list]:
        """Load all embeddings from SQLite"""
        try:
            cursor = self.conn.execute(
                f"SELECT id, text, embedding, metadata, timestamp, "
                f"collection, category, tags FROM {self._memories_table}"
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
                meta = self._reconstruct_metadata(row[3], row[5], row[6], row[7])
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
    
    def search_by_metadata(self, metadata_filter: dict, limit: int = None,
                          after: datetime = None, before: datetime = None) -> list:
        """
        Search memories by metadata fields with optional temporal filtering.
        
        Uses SQL WHERE for dedicated columns (collection, category, tags, timestamp)
        and Python-side filtering for remaining JSON metadata fields.
        
        Args:
            metadata_filter: Dict of metadata fields to match
            limit: Maximum results to return (None = all)
            after: Only return memories after this timestamp (inclusive)
            before: Only return memories before this timestamp (exclusive)
        
        Returns:
            List of dicts with id, text, metadata, timestamp
        """
        try:
            # Build SQL WHERE for dedicated columns
            dedicated = {"collection", "category", "tags"}
            sql_conditions = []
            sql_params = []
            json_filter = {}
            
            for k, v in metadata_filter.items():
                if k in dedicated:
                    if k == "tags":
                        # Tags is comma-separated; check if value is in the list
                        sql_conditions.append(f"(',' || tags || ',' LIKE ?)")
                        sql_params.append(f"%,{v},%")
                    else:
                        sql_conditions.append(f"{k} = ?")
                        sql_params.append(v)
                else:
                    json_filter[k] = v
            
            # Add temporal filtering
            if after:
                sql_conditions.append("timestamp >= ?")
                sql_params.append(after.strftime('%Y-%m-%dT%H:%M:%SZ'))
            if before:
                sql_conditions.append("timestamp < ?")
                sql_params.append(before.strftime('%Y-%m-%dT%H:%M:%SZ'))
            
            sql = (f"SELECT id, text, metadata, timestamp, collection, category, tags "
                   f"FROM {self._memories_table}")
            if sql_conditions:
                sql += " WHERE " + " AND ".join(sql_conditions)
            sql += " ORDER BY timestamp DESC"
            
            cursor = self.conn.execute(sql, sql_params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                meta = self._reconstruct_metadata(row[2], row[4], row[5], row[6])
                # Check remaining JSON fields
                if json_filter and not all(meta.get(k) == v for k, v in json_filter.items()):
                    continue
                results.append({
                    "id": str(row[0]),
                    "text": row[1],
                    "metadata": meta,
                    "timestamp": row[3]
                })
                if limit and len(results) >= limit:
                    break
            
            return results
        except Exception as e:
            logger.error(f"Failed to search by metadata: {e}")
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
        """Log a search query to query.log as single-line JSON"""
        try:
            log_entry = {
                "ts": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "query": query,
                "limit": limit,
                "min_score": min_score,
                "num_results": len(results),
                "top_score": round(top_score, 4),
                "score_gap": round(score_gap, 4),
                "below_threshold": top_score < 0.4,
                "timing": timing,
                "results": [
                    {
                        "id": r["id"],
                        "score": round(r["score"], 4),
                        "source": r["metadata"].get("source", ""),
                        "size": len(r["text"])
                    }
                    for r in results
                ]
            }
            
            query_logger.info(json.dumps(log_entry, separators=(',', ':')))
            
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
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            self.conn.executemany(
                f"INSERT INTO {self._usage_table} (memory_id, timestamp) VALUES (?, ?)",
                [(int(mid), timestamp) for mid in memory_ids]
            )
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Failed to track usage: {e}")
    
    def _load_bm25_from_storage(self, ids: list) -> bool:
        """
        Load BM25 index from the persistent bm25_index table.
        Populates self._bm25 internal state directly from stored token data.
        
        Returns True if loaded successfully, False to fall back to build().
        """
        import math
        from .config import BM25_ENABLED
        
        if not BM25_ENABLED or self._bm25 is None:
            return False
        
        index_count = self.bm25_index_count()
        doc_count = len(ids)
        
        if index_count == 0:
            return False
        
        if index_count != doc_count:
            logger.info(
                f"BM25 index stale ({index_count} indexed vs {doc_count} docs), rebuilding..."
            )
            self.rebuild_bm25_index()
        
        # Load token data from table
        data = self.load_bm25_index()
        if not data:
            return False
        
        # Build id -> position mapping
        id_to_pos = {int(doc_id): pos for pos, doc_id in enumerate(ids)}
        
        # Reconstruct BM25Index internals from persisted data
        bm25 = self._bm25
        bm25._doc_count = doc_count
        bm25._doc_freqs = [{} for _ in range(doc_count)]
        doc_lens = np.zeros(doc_count, dtype=np.float32)
        df = {}  # term -> document frequency
        
        for memory_id, tf_dict in data['doc_freqs'].items():
            pos = id_to_pos.get(int(memory_id))
            if pos is None:
                continue
            bm25._doc_freqs[pos] = tf_dict
            doc_len = sum(tf_dict.values())
            doc_lens[pos] = doc_len
            for token in tf_dict:
                df[token] = df.get(token, 0) + 1
        
        bm25._doc_lens = doc_lens
        bm25._avgdl = float(doc_lens.mean()) if doc_count > 0 else 0.0
        
        # Compute IDF from document frequencies
        bm25._idf = {}
        for term, freq in df.items():
            bm25._idf[term] = math.log(
                (doc_count - freq + 0.5) / (freq + 0.5) + 1.0
            )
        
        logger.info(
            f"BM25 index loaded from storage: {doc_count} docs, "
            f"{len(bm25._idf)} unique terms"
        )
        return True
    
    def load_bm25_index(self) -> dict:
        """
        Load BM25 token data from the persistent bm25_index table.
        
        Returns:
            Dict with 'doc_freqs' (list of {token: freq} dicts, indexed by memory id),
            'id_map' (memory_id -> list index), and 'count' (number of indexed docs).
            Returns empty dict if table is empty.
        """
        try:
            rows = self.conn.execute(
                "SELECT memory_id, token, term_freq FROM bm25_index ORDER BY memory_id"
            ).fetchall()
            
            if not rows:
                return {}
            
            # Build per-document term frequency dicts
            doc_freqs = {}  # memory_id -> {token: freq}
            for memory_id, token, freq in rows:
                if memory_id not in doc_freqs:
                    doc_freqs[memory_id] = {}
                doc_freqs[memory_id][token] = freq
            
            return {
                'doc_freqs': doc_freqs,
                'count': len(doc_freqs)
            }
            
        except sqlite3.Error as e:
            logger.warning(f"Failed to load BM25 index: {e}")
            return {}
    
    def bm25_index_count(self) -> int:
        """Return number of documents in the BM25 index."""
        try:
            row = self.conn.execute(
                "SELECT COUNT(DISTINCT memory_id) FROM bm25_index"
            ).fetchone()
            return row[0] if row else 0
        except sqlite3.Error:
            return 0
    
    def rebuild_bm25_index(self):
        """
        Rebuild the bm25_index table from all documents in memories.
        Use after migration or if the index gets out of sync.
        """
        try:
            self.conn.execute("DELETE FROM bm25_index")
            
            cursor = self.conn.execute(
                f"SELECT id, text FROM {self._memories_table}"
            )
            
            count = 0
            for row in cursor:
                self._index_bm25_tokens(row[0], row[1])
                count += 1
                if count % 1000 == 0:
                    logger.info(f"BM25 index rebuild: {count} docs...")
            
            self.conn.commit()
            logger.info(f"BM25 index rebuilt: {count} documents indexed")
            return count
            
        except sqlite3.Error as e:
            logger.error(f"BM25 index rebuild failed: {e}")
            raise
    
    def rebuild_embeddings(self, embedder):
        """
        Rebuild embeddings for all documents with the current embedding model.
        Use after switching models or to update embeddings.
        
        Args:
            embedder: Embedder instance to use for re-embedding
        
        Returns:
            Number of documents re-embedded
        """
        try:
            # Get total count first
            total = self.conn.execute(
                f"SELECT COUNT(*) FROM {self._memories_table}"
            ).fetchone()[0]
            
            cursor = self.conn.execute(
                f"SELECT id, text FROM {self._memories_table}"
            )
            
            count = 0
            for row in cursor:
                memory_id, text = row[0], row[1]
                
                # Generate new embedding
                embedding = embedder.encode(text)
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                
                # Update embedding in database
                self.conn.execute(
                    f"UPDATE {self._memories_table} SET embedding = ? WHERE id = ?",
                    (embedding_blob, memory_id)
                )
                
                count += 1
                # Print in-place progress update
                print(f"\rRebuilding embeddings: {count}/{total}", end="", flush=True)
            
            print()  # Final newline
            self.conn.commit()
            logger.info(f"Embeddings rebuilt: {count} documents")
            return count
            
        except Exception as e:
            print()  # Ensure newline on error
            logger.error(f"Embeddings rebuild failed: {e}")
            raise
    
    @staticmethod
    def _has_tag(tags: str, tag: str) -> bool:
        """Check if a comma-separated tags string contains a specific tag."""
        return tag in (t.strip() for t in (tags or "").split(",") if t.strip())

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID. Respects "keep" tag.
        
        Args:
            memory_id: Memory ID to delete
        
        Returns:
            True if deleted, False if not found or protected by "keep" tag
        """
        try:
            row = self.conn.execute(
                f"SELECT tags FROM {self._memories_table} WHERE id = ?",
                (int(memory_id),)
            ).fetchone()
            
            if not row:
                return False  # Not found
            
            if self._has_tag(row[0], "keep"):
                logger.info(f"Skipped deletion of memory {memory_id} (keep tag)")
                return False
            
            cursor = self.conn.execute(
                f"DELETE FROM {self._memories_table} WHERE id = ?",
                (int(memory_id),)
            )
            self.conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted memory {memory_id}")
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise
    
    def delete_batch(self, memory_ids: list) -> int:
        """
        Delete multiple memories by ID. Respects "keep" tag.
        
        Args:
            memory_ids: List of memory IDs to delete
        
        Returns:
            Number of memories deleted
        """
        if not memory_ids:
            return 0
        try:
            placeholders = ",".join("?" * len(memory_ids))
            rows = self.conn.execute(
                f"SELECT id, tags FROM {self._memories_table} WHERE id IN ({placeholders})",
                [int(mid) for mid in memory_ids]
            ).fetchall()
            
            deletable = []
            kept = []
            for row in rows:
                memory_id = row[0]
                if self._has_tag(row[1], "keep"):
                    kept.append(memory_id)
                else:
                    deletable.append(memory_id)
            
            if kept:
                logger.info(f"Skipped deletion of {len(kept)} memories with keep tag")
            
            if not deletable:
                return 0
            
            placeholders = ",".join("?" * len(deletable))
            cursor = self.conn.execute(
                f"DELETE FROM {self._memories_table} WHERE id IN ({placeholders})",
                deletable
            )
            self.conn.commit()
            count = cursor.rowcount
            logger.info(f"Deleted {count} memories")
            return count
        except Exception as e:
            logger.error(f"Failed to delete memories: {e}")
            raise
    
    def close(self):
        """Close SQLite connection"""
        if self.conn:
            self.conn.close()
            logger.info("SQLite connection closed")
