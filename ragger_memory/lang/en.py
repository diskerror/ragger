"""
English language strings for Ragger Memory.

To add a new language: copy this file to xx.py, translate all values,
and import it in __init__.py instead.
"""

# --- Config ---
ERR_CONFIG_NOT_FOUND = (
    "No config file found.\n"
    "Searched: /etc/ragger.conf, ~/.ragger/ragger.conf\n"
    "Use --config-file=<path> to specify one."
)
ERR_CONFIG_FILE_MISSING = "Config file not found: {path}"

# --- Server ---
MSG_SERVER_RUNNING = "Ragger Memory server running on http://{host}:{port}"
MSG_SERVER_ENDPOINTS = "Endpoints:"
MSG_SERVER_STOP = "Press Ctrl+C to stop."
ERR_PORT_IN_USE = (
    "Error: port {port} is already in use.\n"
    "Ragger daemon is already running. Use 'ragger search', 'ragger store',\n"
    "or 'ragger chat' to connect.\n"
    "To run a second instance, set a different port in your config file\n"
    "(~/.ragger/ragger.conf) and update the OpenClaw plugin's serverUrl to match."
)
ERR_REQUEST = "Request error: {error}"

# --- Backend ---
MSG_LOADING_EMBEDDINGS = "Loading {count} embeddings from storage..."
MSG_LOADED_EMBEDDINGS = "Loaded {count} embeddings ({size_kb:.0f} KB)"
MSG_STORED_MEMORY = "Stored memory: {id}"
ERR_STORE_FAILED = "Failed to store memory: {error}"
MSG_NO_MEMORIES = "No memories stored yet"
MSG_NO_MEMORIES_IN_COLLECTIONS = "No memories in collections {collections}"
MSG_SEARCH_RESULTS = "Search returned {count} results in {ms:.1f}ms"
ERR_SEARCH_FAILED = "Search failed: {error}"

# --- BM25 ---
MSG_BM25_BUILT = "BM25 index built: {doc_count} docs, {vocab_size} vocab"

# --- Memory factory ---
MSG_USING_SQLITE = "Using SQLite backend"
ERR_UNKNOWN_ENGINE = (
    "Unknown storage engine: {engine}. "
    "See backend.py for how to implement a custom backend."
)

# --- Embedding ---
ERR_MODEL_CACHE_BROKEN_SYMLINK = (
    "Model cache directory is a broken symlink:\n{path}"
)
ERR_MODEL_CACHE_NOT_FOUND = (
    "Model cache directory not found: {path}\n"
    "Run './ragger.py --update-model' to download the embedding model."
)
ERR_MODEL_NO_SNAPSHOTS = (
    "Model '{model}' has no snapshots in {path}\n"
    "Run './ragger.py --update-model' to download it."
)
ERR_MODEL_NOT_FOUND = (
    "Model '{model}' not found in {path}\n"
    "Run './ragger.py --update-model' to download it."
)

