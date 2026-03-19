"""
Configuration for Ragger Memory

Loaded from ragger.conf at runtime.
Search order: --config-file= → ~/.ragger/ragger.conf → bootstrap new config
First file found wins. Created automatically on first run.
"""
import configparser
import os
import sys


def expand_path(path: str) -> str:
    """Expand ~ to $HOME."""
    return os.path.expanduser(path)


DEFAULT_CONFIG = """\
# ragger.conf — Ragger Memory configuration
#
# Search order:
#   1. --config-file=<path>  (explicit override)
#   2. ~/.ragger/ragger.conf (per-user default)
#
# First file found wins. Created automatically on first run.

[server]
host = 127.0.0.1
port = 8432

[storage]
db_path = ~/.ragger/memories.db
default_collection = memory

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384
# model_dir: path to ONNX model files (default: ~/.ragger/models)
# model_dir = ~/.ragger/models

[search]
default_limit = 5
default_min_score = 0.4
bm25_enabled = true
bm25_weight = 0.3
vector_weight = 0.7
bm25_k1 = 1.5
bm25_b = 0.75

[logging]
log_dir = ~/.ragger
query_log = true
http_log = true
mcp_log = true

[paths]
normalize_home = true

[import]
minimum_chunk_size = 300
"""


def _bootstrap_user_config() -> str:
    """Create ~/.ragger/ and default config on first run."""
    ragger_dir = expand_path("~/.ragger")
    conf_path = os.path.join(ragger_dir, "ragger.conf")
    os.makedirs(ragger_dir, exist_ok=True)
    with open(conf_path, "w") as f:
        f.write(DEFAULT_CONFIG)
    print(f"Created default config: {conf_path}", file=sys.stderr)
    return conf_path


def find_config_file(cli_path: str = "") -> str:
    """
    Find config file using search order. Returns path or bootstraps new one.

    Args:
        cli_path: Path from --config-file (empty if not given)
    """
    # 1. Explicit --config-file= takes highest priority
    if cli_path:
        resolved = expand_path(cli_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Config file not found: {resolved}")
        return resolved

    # 2. ~/.ragger/ragger.conf
    user_conf = expand_path("~/.ragger/ragger.conf")
    if os.path.exists(user_conf):
        return user_conf

    # 3. First run — bootstrap default config
    return _bootstrap_user_config()


def load_config(path: str) -> dict:
    """Load config from an INI file. Returns dict of values."""
    parser = configparser.ConfigParser()
    parser.read(path)

    def get(section, key, fallback):
        return parser.get(section, key, fallback=fallback)

    def getint(section, key, fallback):
        return parser.getint(section, key, fallback=fallback)

    def getfloat(section, key, fallback):
        return parser.getfloat(section, key, fallback=fallback)

    def getbool(section, key, fallback):
        return parser.getboolean(section, key, fallback=fallback)

    return {
        # Server
        "host": get("server", "host", "127.0.0.1"),
        "port": getint("server", "port", 8432),

        # Storage
        "db_path": get("storage", "db_path", "~/.ragger/memories.db"),
        "default_collection": get("storage", "default_collection", "memory"),

        # Embedding
        "embedding_model": get("embedding", "model", "all-MiniLM-L6-v2"),
        "embedding_dimensions": getint("embedding", "dimensions", 384),
        "model_dir": get("embedding", "model_dir", ""),

        # Search
        "default_search_limit": getint("search", "default_limit", 5),
        "default_min_score": getfloat("search", "default_min_score", 0.4),
        "bm25_enabled": getbool("search", "bm25_enabled", True),
        "bm25_weight": getfloat("search", "bm25_weight", 0.3),
        "vector_weight": getfloat("search", "vector_weight", 0.7),
        "bm25_k1": getfloat("search", "bm25_k1", 1.5),
        "bm25_b": getfloat("search", "bm25_b", 0.75),

        # Logging
        "log_dir": get("logging", "log_dir", "~/.ragger"),
        "query_log_enabled": getbool("logging", "query_log", True),
        "http_log_enabled": getbool("logging", "http_log", True),
        "mcp_log_enabled": getbool("logging", "mcp_log", True),

        # Paths
        "normalize_home_path": getbool("paths", "normalize_home", True),

        # Import
        "minimum_chunk_size": getint("import", "minimum_chunk_size", 300),
    }


# ---------------------------------------------------------------------------
# Module-level config — initialized on first access
# ---------------------------------------------------------------------------
_config: dict | None = None
_config_path: str | None = None


def init_config(cli_path: str = "") -> dict:
    """Initialize config from file. Call once at startup."""
    global _config, _config_path
    _config_path = find_config_file(cli_path)
    _config = load_config(_config_path)
    return _config


def get_config() -> dict:
    """Get loaded config. Auto-initializes if not yet loaded."""
    global _config
    if _config is None:
        init_config()
    return _config


def get_config_path() -> str | None:
    """Return the path of the loaded config file."""
    return _config_path


# ---------------------------------------------------------------------------
# Convenience accessors (backward compatibility)
# ---------------------------------------------------------------------------
# These are properties that lazily read from the loaded config.
# Code that was doing `from config import DEFAULT_PORT` will need to switch
# to `from config import get_config; cfg = get_config(); cfg["port"]`
# or use these module-level accessors.

def __getattr__(name: str):
    """Allow old-style access like config.DEFAULT_PORT via module __getattr__."""
    _map = {
        "DEFAULT_HOST": "host",
        "DEFAULT_PORT": "port",
        "STORAGE_ENGINE": None,
        "SQLITE_PATH": "db_path",
        "SQLITE_MEMORIES_TABLE": None,
        "SQLITE_USAGE_TABLE": None,
        "EMBEDDING_MODEL": "embedding_model",
        "EMBEDDING_DIMENSIONS": "embedding_dimensions",
        "MODEL_CACHE_DIR": None,
        "LOG_DIR": "log_dir",
        "QUERY_LOG_ENABLED": "query_log_enabled",
        "HTTP_LOG_ENABLED": "http_log_enabled",
        "MCP_LOG_ENABLED": "mcp_log_enabled",
        "USAGE_TRACKING_ENABLED": None,
        "NORMALIZE_HOME_PATH": "normalize_home_path",
        "BM25_ENABLED": "bm25_enabled",
        "BM25_WEIGHT": "bm25_weight",
        "VECTOR_WEIGHT": "vector_weight",
        "BM25_K1": "bm25_k1",
        "BM25_B": "bm25_b",
        "DEFAULT_SEARCH_LIMIT": "default_search_limit",
        "DEFAULT_MIN_SCORE": "default_min_score",
        "DEFAULT_COLLECTION": "default_collection",
        "MINIMUM_CHUNK_SIZE": "minimum_chunk_size",
    }

    if name in _map:
        cfg_key = _map[name]
        if cfg_key is None:
            # Static values that don't come from config file
            static = {
                "STORAGE_ENGINE": "sqlite",
                "SQLITE_MEMORIES_TABLE": "memories",
                "SQLITE_USAGE_TABLE": "memory_usage",
                "USAGE_TRACKING_ENABLED": True,
                "MODEL_CACHE_DIR": os.environ.get(
                    'SENTENCE_TRANSFORMERS_HOME',
                    os.path.expanduser('~/.cache/huggingface')
                ),
            }
            return static[name]
        cfg = get_config()
        val = cfg[cfg_key]
        # Expand paths for path-like values
        if name in ("SQLITE_PATH", "LOG_DIR"):
            return expand_path(val)
        return val

    raise AttributeError(f"module 'config' has no attribute {name}")
