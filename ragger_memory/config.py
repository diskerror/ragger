"""
Configuration for Ragger Memory
"""
import os

# --- Server ---
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8432

# --- Engine selection ---
STORAGE_ENGINE = "sqlite"

# --- SQLite backend ---
SQLITE_PATH = "~/.ragger/memories.db"
SQLITE_MEMORIES_TABLE = "memories"
SQLITE_USAGE_TABLE = "memory_usage"

# --- Embedding model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, ~90MB
EMBEDDING_DIMENSIONS = 384
MODEL_CACHE_DIR = os.environ.get(
    'SENTENCE_TRANSFORMERS_HOME',
    os.path.expanduser('~/.cache/huggingface')
)

# --- Logging ---
LOG_DIR = os.path.expanduser("~/.ragger")
QUERY_LOG_ENABLED = True       # query.log — search queries, scores, timing
HTTP_LOG_ENABLED = True        # http.log — HTTP server requests/responses
MCP_LOG_ENABLED = True         # mcp.log — MCP JSON-RPC interactions
# error.log is always on

# --- Usage tracking ---
USAGE_TRACKING_ENABLED = True

# --- Path normalization ---
NORMALIZE_HOME_PATH = True  # Replace $HOME with ~ in stored text and metadata

# --- Hybrid search (BM25 + vector) ---
BM25_ENABLED = True
BM25_WEIGHT = 0.3       # Weight for BM25 scores in hybrid merge (0.0–1.0)
VECTOR_WEIGHT = 0.7     # Weight for vector cosine scores (should sum to 1.0 with BM25_WEIGHT)
BM25_K1 = 1.5           # Term frequency saturation
BM25_B = 0.75           # Document length normalization

# --- Search defaults ---
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_MIN_SCORE = 0.4
DEFAULT_COLLECTION = "memory"  # default collection for search and untagged memories
MINIMUM_CHUNK_SIZE = 300  # merge short paragraphs until at least this size
