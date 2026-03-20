"""
Configuration for Ragger Memory

Layered config:
  1. System config (/etc/ragger.conf or platform equivalent) — full settings + user defaults
  2. User config (~/.ragger/ragger.conf) — personal overrides (subset of settings)
  3. --config-file= — explicit override (replaces both, for development/testing)

System config is the authoritative source. User config can only override
user-level settings (search preferences, default collection, query logging).
"""
import configparser
import os
import platform
import sys


def expand_path(path: str) -> str:
    """Expand ~ to $HOME."""
    return os.path.expanduser(path)


def system_config_path() -> str:
    """Platform-specific system config path."""
    if platform.system() == "Windows":
        appdata = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        return os.path.join(appdata, "ragger", "ragger.conf")
    else:
        # macOS and Linux: /etc/ragger.conf
        return "/etc/ragger.conf"


def system_data_dir() -> str:
    """Platform-specific system data directory."""
    if platform.system() == "Windows":
        appdata = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        return os.path.join(appdata, "ragger")
    elif platform.system() == "Darwin":
        return "/var/ragger"
    else:
        # Linux FHS
        return "/var/lib/ragger"


def system_log_dir() -> str:
    """Platform-specific system log directory."""
    if platform.system() == "Windows":
        appdata = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        return os.path.join(appdata, "ragger", "logs")
    else:
        return "/var/log/ragger"


def system_model_dir() -> str:
    """Platform-specific system model directory."""
    return os.path.join(system_data_dir(), "models")


# ---------------------------------------------------------------------------
# Default configs (embedded)
# ---------------------------------------------------------------------------

SYSTEM_DEFAULT_CONFIG = """\
# ragger.conf — Ragger Memory system configuration
#
# This file controls the daemon and sets defaults for all users.
# Location: {system_path}
#
# User overrides: ~/.ragger/ragger.conf (limited settings only)

[server]
host = 127.0.0.1
port = 8432

[storage]
# System-wide shared memory database
db_path = {data_dir}/memories.db
default_collection = memory

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384
model_dir = {model_dir}

[search]
default_limit = 5
default_min_score = 0.4
bm25_enabled = true
bm25_weight = 0.3
vector_weight = 0.7
bm25_k1 = 1.5
bm25_b = 0.75

[logging]
log_dir = {log_dir}
query_log = true
http_log = true
mcp_log = true

[paths]
normalize_home = true

[import]
minimum_chunk_size = 300
""".format(
    system_path=system_config_path(),
    data_dir=system_data_dir(),
    model_dir=system_model_dir(),
    log_dir=system_log_dir()
)

USER_DEFAULT_CONFIG = """\
# ragger.conf — User configuration
#
# Personal overrides. System config provides all defaults.
# Only settings listed here can be changed by the user.

[user]
# mode = memory-only   # or "full" for chat access

[search]
# default_limit = 5
# default_min_score = 0.4

[storage]
# default_collection = memory

[logging]
# query_log = true     # set to false to opt out of query logging
"""

# Settings the user is allowed to override
USER_OVERRIDABLE = {
    ("user", "mode"),
    ("search", "default_limit"),
    ("search", "default_min_score"),
    ("storage", "default_collection"),
    ("logging", "query_log"),
}


def _bootstrap_user_config() -> str:
    """Create ~/.ragger/ and default user config on first run."""
    ragger_dir = expand_path("~/.ragger")
    conf_path = os.path.join(ragger_dir, "ragger.conf")
    os.makedirs(ragger_dir, exist_ok=True)
    with open(conf_path, "w") as f:
        f.write(USER_DEFAULT_CONFIG)
    print(f"Created user config: {conf_path}", file=sys.stderr)
    return conf_path


def _bootstrap_single_user_config() -> str:
    """
    Bootstrap for single-user mode (no system config).
    Creates a full config in ~/.ragger/ so everything works standalone.
    """
    ragger_dir = expand_path("~/.ragger")
    conf_path = os.path.join(ragger_dir, "ragger.conf")
    os.makedirs(ragger_dir, exist_ok=True)

    # Single-user gets a full config with user-local paths
    single_user_config = """\
# ragger.conf — Ragger Memory configuration (single-user)
#
# All settings in one file. When a system config exists at
# {system_path}, this file becomes a user-override file.

[server]
host = 127.0.0.1
port = 8432

[storage]
db_path = ~/.ragger/memories.db
default_collection = memory

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384
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
""".format(system_path=system_config_path())

    with open(conf_path, "w") as f:
        f.write(single_user_config)
    print(f"Created default config: {conf_path}", file=sys.stderr)
    return conf_path


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

        # User
        "user_mode": get("user", "mode", "memory-only"),
    }


def load_layered_config(system_path: str | None, user_path: str | None) -> dict:
    """
    Load config with layering: system first, user overrides on top.
    User can only override settings in USER_OVERRIDABLE.
    """
    # Start with defaults
    if system_path and os.path.exists(system_path):
        cfg = load_config(system_path)
    elif user_path and os.path.exists(user_path):
        # Single-user mode: user config has everything
        cfg = load_config(user_path)
        return cfg
    else:
        cfg = load_config("")  # all defaults

    # Apply user overrides (limited set)
    if user_path and os.path.exists(user_path):
        user_parser = configparser.ConfigParser()
        user_parser.read(user_path)

        for section, key in USER_OVERRIDABLE:
            if user_parser.has_option(section, key):
                val = user_parser.get(section, key)
                # Map to config dict keys
                key_map = {
                    ("user", "mode"): "user_mode",
                    ("search", "default_limit"): "default_search_limit",
                    ("search", "default_min_score"): "default_min_score",
                    ("storage", "default_collection"): "default_collection",
                    ("logging", "query_log"): "query_log_enabled",
                }
                cfg_key = key_map.get((section, key))
                if cfg_key:
                    # Type coercion
                    if cfg_key in ("default_search_limit",):
                        cfg[cfg_key] = int(val)
                    elif cfg_key in ("default_min_score",):
                        cfg[cfg_key] = float(val)
                    elif cfg_key in ("query_log_enabled",):
                        cfg[cfg_key] = val.lower() in ("true", "yes", "1")
                    else:
                        cfg[cfg_key] = val

    return cfg


def find_config_files(cli_path: str = "") -> tuple[str | None, str | None]:
    """
    Find system and user config files.

    Returns (system_path, user_path) — either may be None.

    With --config-file: that file is used as the sole config (no layering).
    Without: system config loaded first, user config overlays.
    """
    # Explicit --config-file overrides everything
    if cli_path:
        resolved = expand_path(cli_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Config file not found: {resolved}")
        return (resolved, None)

    sys_path = system_config_path()
    user_path = expand_path("~/.ragger/ragger.conf")

    has_system = os.path.exists(sys_path)
    has_user = os.path.exists(user_path)

    if has_system and not has_user:
        # System config exists, bootstrap user config for overrides
        _bootstrap_user_config()
        return (sys_path, user_path)

    if has_system and has_user:
        return (sys_path, user_path)

    if has_user:
        # Single-user mode: user config has everything
        return (None, user_path)

    # First run, no system config — single-user bootstrap
    _bootstrap_single_user_config()
    return (None, user_path)


# Keep backward compat
def find_config_file(cli_path: str = "") -> str:
    """Legacy: returns a single config path."""
    sys_path, user_path = find_config_files(cli_path)
    if cli_path:
        return sys_path  # explicit override
    return sys_path or user_path or ""


# ---------------------------------------------------------------------------
# Module-level config — initialized on first access
# ---------------------------------------------------------------------------
_config: dict | None = None
_config_path: str | None = None
_is_multi_user: bool = False


def init_config(cli_path: str = "") -> dict:
    """Initialize config from file(s). Call once at startup."""
    global _config, _config_path, _is_multi_user

    sys_path, user_path = find_config_files(cli_path)

    if cli_path:
        # Explicit override — single file, no layering
        _config_path = sys_path
        _config = load_config(sys_path)
    else:
        _config_path = sys_path or user_path
        _config = load_layered_config(sys_path, user_path)
        _is_multi_user = sys_path is not None

    return _config


def get_config() -> dict:
    """Get loaded config. Auto-initializes if not yet loaded."""
    global _config
    if _config is None:
        init_config()
    return _config


def is_multi_user() -> bool:
    """Whether system config was found (multi-user mode)."""
    return _is_multi_user


def get_config_path() -> str | None:
    """Return the path of the loaded config file."""
    return _config_path


# ---------------------------------------------------------------------------
# Convenience accessors (backward compatibility)
# ---------------------------------------------------------------------------

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
        if name in ("SQLITE_PATH", "LOG_DIR"):
            return expand_path(val)
        return val

    raise AttributeError(f"module 'config' has no attribute {name}")
