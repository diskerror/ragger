"""
Configuration for Ragger Memory
"""
import os

# --- Engine selection ---
STORAGE_ENGINE = "sqlite"  # "mongodb" or "sqlite"

# --- MongoDB backend ---
MONGODB_URI = "mongodb://localhost:27017/"
MONGODB_DB_NAME = "ragger"
MONGODB_COLLECTION = "memories"
MONGODB_QUERY_LOG_COLLECTION = "query_log"

# --- SQLite backend ---
SQLITE_PATH = "~/.local/share/ragger/memories.db"
SQLITE_MEMORIES_TABLE = "memories"
SQLITE_QUERY_LOG_TABLE = "query_log"

# --- Query logging (all backends) ---
QUERY_LOGGING_ENABLED = True

# --- Usage tracking ---
USAGE_TRACKING_ENABLED = True
SQLITE_USAGE_TABLE = "memory_usage"

# --- Embedding model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, ~90MB
EMBEDDING_DIMENSIONS = 384
MODEL_CACHE_DIR = os.environ.get(
    'SENTENCE_TRANSFORMERS_HOME',
    os.path.expanduser('~/.cache/huggingface')
)

# --- Search defaults ---
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_MIN_SCORE = 0.4
DEFAULT_CHUNK_SIZE = 500  # characters per chunk
