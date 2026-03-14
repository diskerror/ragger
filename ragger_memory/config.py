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
MONGODB_USAGE_COLLECTION = "memory_usage"

# --- SQLite backend ---
SQLITE_PATH = "~/.local/share/ragger/memories.db"
SQLITE_MEMORIES_TABLE = "memories"
SQLITE_QUERY_LOG_TABLE = "query_log"

# --- Query logging (all backends) ---
QUERY_LOGGING_ENABLED = True

# --- Usage tracking ---
USAGE_TRACKING_ENABLED = True
SQLITE_USAGE_TABLE = "memory_usage"

# --- Path normalization ---
NORMALIZE_HOME_PATH = True  # Replace $HOME with ~ in stored text and metadata

# --- Hybrid search (BM25 + vector) ---
BM25_ENABLED = True
BM25_WEIGHT = 0.3       # Weight for BM25 scores in hybrid merge (0.0–1.0)
VECTOR_WEIGHT = 0.7     # Weight for vector cosine scores (should sum to 1.0 with BM25_WEIGHT)
BM25_K1 = 1.5           # Term frequency saturation
BM25_B = 0.75           # Document length normalization

# --- Embedding model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, ~90MB
EMBEDDING_DIMENSIONS = 384
MODEL_CACHE_DIR = os.environ.get(
    'SENTENCE_TRANSFORMERS_HOME',
    os.path.expanduser('~/.cache/huggingface')
)

# --- Server ---
DEFAULT_PORT = 8432

# --- Search defaults ---
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_MIN_SCORE = 0.4
DEFAULT_COLLECTION = "memory"  # default collection for search and untagged memories
MINIMUM_CHUNK_SIZE = 300  # merge short paragraphs until at least this size
