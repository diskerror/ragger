"""
Configuration for Ragger Memory

Config search order:
  1. --config=<path>       (explicit override, replaces all layering)
  2. /etc/ragger.ini       (system config, installed by install-system.sh)
  3. ~/.ragger/ragger.ini  (per-user overrides, auto-created on first run)

System config is the authoritative source. User config can only override
user-level settings (search preferences, default collection, query logging).

If no system config exists, ~/.ragger/ragger.ini is bootstrapped with
full settings for standalone single-user operation.
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
        return os.path.join(appdata, "ragger", "ragger.ini")
    else:
        # macOS and Linux: /etc/ragger.ini
        return "/etc/ragger.ini"


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
# ragger.ini — Ragger Memory system configuration
#
# This file controls the daemon and sets defaults for all users.
# Location: {system_path}
#
# User overrides: ~/.ragger/ragger.ini (limited settings only)

[server]
host = 127.0.0.1
port = 8432
# Single-user mode: no system DB is created.
# Each user's memory lives only in ~/.ragger/memories.db.
# Set to false for shared/collaborative document storage.
single_user = true

[storage]
default_collection = memory

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384
model_dir = {model_dir}

[search]
default_limit = 5
default_min_score = 0.4
bm25_enabled = true
; Weights are ratios — don't need to sum to 1.0. "3 and 7" = "0.3 and 0.7"
bm25_weight = 3
vector_weight = 7
bm25_k1 = 1.5
bm25_b = 0.75

[inference]
# provider = openai-compatible
# api_url = https://api.anthropic.com/v1
# api_key = sk-ant-...
# max_tokens = 4096

[logging]
log_dir = {log_dir}
query_log = true
http_log = true
mcp_log = true

[paths]
normalize_home = true

[import]
minimum_chunk_size = 300

[chat]
# Conversation persistence
store_turns = true        # "true" (per-turn), "session" (one growing entry), or "false" (summaries only)
summarize_on_pause = true
pause_minutes = 10
summarize_on_quit = true
# System hard limits (user can't override)
max_turn_retention_minutes = 60
max_turns_stored = 100
""".format(
    system_path=system_config_path(),
    model_dir=system_model_dir(),
    log_dir=system_log_dir()
)

USER_DEFAULT_CONFIG = """\
# ragger.ini — User configuration
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

[inference]
# model = claude-sonnet-4-5

[logging]
# query_log = true     # set to false to opt out of query logging
"""

# Server infrastructure keys — system config always wins (blacklist)
SERVER_LOCKED = {
    ("server", "host"),
    ("server", "port"),
    ("storage", "db_path"),
    ("storage", "formats_dir"),
    ("logging", "log_dir"),
    ("embedding", "model"),
    ("embedding", "dimensions"),
    ("embedding", "model_dir"),
    # Chat hard limits (system controls the ceiling)
    ("chat", "max_turn_retention_minutes"),
    ("chat", "max_turns_stored"),
}


def _bootstrap_user_config() -> str:
    """Create ~/.ragger/ and default user config on first run."""
    ragger_dir = expand_path("~/.ragger")
    conf_path = os.path.join(ragger_dir, "ragger.ini")
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
    conf_path = os.path.join(ragger_dir, "ragger.ini")
    os.makedirs(ragger_dir, exist_ok=True)

    # Single-user gets a full config with user-local paths
    single_user_config = """\
# ragger.ini — Ragger Memory configuration (single-user)
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
; Weights are ratios — don't need to sum to 1.0. "3 and 7" = "0.3 and 0.7"
bm25_weight = 3
vector_weight = 7
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

[chat]
# Conversation persistence
store_turns = true        # "true" (per-turn), "session" (one growing entry), or "false" (summaries only)
summarize_on_pause = true
pause_minutes = 10
summarize_on_quit = true
# System hard limits (single-user can modify these)
max_turn_retention_minutes = 60
max_turns_stored = 100
""".format(system_path=system_config_path())

    with open(conf_path, "w") as f:
        f.write(single_user_config)
    print(f"Created default config: {conf_path}", file=sys.stderr)
    return conf_path


def _parse_inference_endpoints(parser) -> list:
    """
    Parse [inference.*] sections into a list of endpoint dicts.

    Example config:
        [inference.local]
        api_url = http://localhost:1234/v1
        api_key = lmstudio-local
        models = qwen/*, llama/*

        [inference.anthropic]
        api_url = https://api.anthropic.com/v1
        api_key = sk-ant-...
        models = claude-*
    """
    endpoints = []
    for section in parser.sections():
        if section.startswith("inference."):
            name = section.split(".", 1)[1]
            ep = {
                "name": name,
                "api_url": parser.get(section, "api_url", fallback=""),
                "api_key": parser.get(section, "api_key", fallback=""),
                "models": parser.get(section, "models", fallback="*"),
                "format": parser.get(section, "format", fallback=""),
            }
            if ep["api_url"]:
                endpoints.append(ep)
    return endpoints


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
        "single_user": getbool("server", "single_user", True),

        # Storage
        "db_path": get("storage", "db_path", "~/.ragger/memories.db"),
        "default_collection": get("storage", "default_collection", "memory"),
        "formats_dir": get("storage", "formats_dir", "/var/ragger/formats"),

        # Embedding
        "embedding_model": get("embedding", "model", "all-MiniLM-L6-v2"),
        "embedding_dimensions": getint("embedding", "dimensions", 384),
        "model_dir": get("embedding", "model_dir", ""),

        # Search
        "default_search_limit": getint("search", "default_limit", 5),
        "default_min_score": getfloat("search", "default_min_score", 0.4),
        "bm25_enabled": getbool("search", "bm25_enabled", True),
        "bm25_weight": getfloat("search", "bm25_weight", 3.0),
        "vector_weight": getfloat("search", "vector_weight", 7.0),
        "bm25_k1": getfloat("search", "bm25_k1", 1.5),
        "bm25_b": getfloat("search", "bm25_b", 0.75),

        # Inference — single endpoint (backward compat)
        "inference_provider": get("inference", "provider", ""),
        "inference_api_url": get("inference", "api_url", ""),
        "inference_api_key": get("inference", "api_key", ""),
        "inference_model": get("inference", "model", "claude-sonnet-4-5"),
        "inference_max_tokens": getint("inference", "max_tokens", 4096),
        "inference_default": get("inference", "default", ""),

        # Inference — multi-endpoint from [inference.*] sections
        "inference_endpoints": _parse_inference_endpoints(parser),

        # Logging
        "log_dir": get("logging", "log_dir", "~/.ragger"),
        "query_log_enabled": getbool("logging", "query_log", True),
        "http_log_enabled": getbool("logging", "http_log", True),
        "mcp_log_enabled": getbool("logging", "mcp_log", True),

        # Paths
        "normalize_home_path": getbool("paths", "normalize_home", True),

        # Import
        "minimum_chunk_size": getint("import", "minimum_chunk_size", 300),

        # Chat persistence
        "chat_store_turns": get("chat", "store_turns", "true"),  # "true", "session", or "false"
        "chat_summarize_on_pause": getbool("chat", "summarize_on_pause", True),
        "chat_pause_minutes": getint("chat", "pause_minutes", 10),
        "chat_summarize_on_quit": getbool("chat", "summarize_on_quit", True),
        "chat_max_turn_retention_minutes": getint("chat", "max_turn_retention_minutes", 60),
        "chat_max_turns_stored": getint("chat", "max_turns_stored", 100),
        "chat_max_persona_chars": getint("chat", "max_persona_chars", 0),  # 0 = unlimited
        "chat_max_memory_results": getint("chat", "max_memory_results", 3),

        # User
        "user_mode": get("user", "mode", "memory-only"),
    }


def load_layered_config(system_path: str | None, user_path: str | None) -> dict:
    """
    Load config with layering: system first, user overrides on top.
    
    SERVER_LOCKED keys (infrastructure) are always taken from system config.
    Everything else: user config wins if set.
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

    # Apply user overrides (everything except SERVER_LOCKED)
    if user_path and os.path.exists(user_path):
        user_parser = configparser.ConfigParser()
        user_parser.read(user_path)

        # Complete mapping of (section, key) → config dict key
        key_map = {
            # Server
            ("server", "host"): "host",
            ("server", "port"): "port",
            ("server", "single_user"): "single_user",
            # Storage
            ("storage", "db_path"): "db_path",
            ("storage", "default_collection"): "default_collection",
            ("storage", "formats_dir"): "formats_dir",
            # Embedding
            ("embedding", "model"): "embedding_model",
            ("embedding", "dimensions"): "embedding_dimensions",
            ("embedding", "model_dir"): "model_dir",
            # Search
            ("search", "default_limit"): "default_search_limit",
            ("search", "default_min_score"): "default_min_score",
            ("search", "bm25_enabled"): "bm25_enabled",
            ("search", "bm25_weight"): "bm25_weight",
            ("search", "vector_weight"): "vector_weight",
            ("search", "bm25_k1"): "bm25_k1",
            ("search", "bm25_b"): "bm25_b",
            # Inference
            ("inference", "provider"): "inference_provider",
            ("inference", "api_url"): "inference_api_url",
            ("inference", "api_key"): "inference_api_key",
            ("inference", "model"): "inference_model",
            ("inference", "max_tokens"): "inference_max_tokens",
            ("inference", "default"): "inference_default",
            # Logging
            ("logging", "log_dir"): "log_dir",
            ("logging", "query_log"): "query_log_enabled",
            ("logging", "http_log"): "http_log_enabled",
            ("logging", "mcp_log"): "mcp_log_enabled",
            # Paths
            ("paths", "normalize_home"): "normalize_home_path",
            # Import
            ("import", "minimum_chunk_size"): "minimum_chunk_size",
            # Chat
            ("chat", "store_turns"): "chat_store_turns",
            ("chat", "summarize_on_pause"): "chat_summarize_on_pause",
            ("chat", "pause_minutes"): "chat_pause_minutes",
            ("chat", "summarize_on_quit"): "chat_summarize_on_quit",
            ("chat", "max_turn_retention_minutes"): "chat_max_turn_retention_minutes",
            ("chat", "max_turns_stored"): "chat_max_turns_stored",
            ("chat", "max_persona_chars"): "chat_max_persona_chars",
            ("chat", "max_memory_results"): "chat_max_memory_results",
            # User
            ("user", "mode"): "user_mode",
        }

        # Type information for coercion
        int_keys = {
            "port", "embedding_dimensions", "default_search_limit",
            "inference_max_tokens", "minimum_chunk_size", "chat_pause_minutes",
            "chat_max_turn_retention_minutes", "chat_max_turns_stored",
            "chat_max_persona_chars", "chat_max_memory_results"
        }
        float_keys = {
            "default_min_score", "bm25_weight", "vector_weight", "bm25_k1", "bm25_b"
        }
        bool_keys = {
            "single_user", "bm25_enabled", "query_log_enabled",
            "http_log_enabled", "mcp_log_enabled", "normalize_home_path",
            "chat_summarize_on_pause", "chat_summarize_on_quit"
        }
        # chat_store_turns is now a string, not bool

        # Overlay user config for non-locked keys
        for section in user_parser.sections():
            for key in user_parser.options(section):
                section_key = (section, key)
                
                # Skip server-locked keys
                if section_key in SERVER_LOCKED:
                    continue
                
                # Skip [inference.*] sections — handled by _parse_inference_endpoints
                if section.startswith("inference."):
                    continue
                
                cfg_key = key_map.get(section_key)
                if cfg_key:
                    val = user_parser.get(section, key)
                    
                    # Type coercion
                    if cfg_key in int_keys:
                        cfg[cfg_key] = int(val)
                    elif cfg_key in float_keys:
                        cfg[cfg_key] = float(val)
                    elif cfg_key in bool_keys:
                        cfg[cfg_key] = val.lower() in ("true", "yes", "1")
                    else:
                        cfg[cfg_key] = val
        
        # Re-parse inference endpoints from user config (not locked)
        user_endpoints = _parse_inference_endpoints(user_parser)
        if user_endpoints:
            cfg["inference_endpoints"] = user_endpoints

    return cfg


def find_config_files(cli_path: str = "") -> tuple[str | None, str | None]:
    """
    Find system and user config files.

    Returns (system_path, user_path) — either may be None.

    --config=<path> replaces /etc/ragger.ini as the system config.
    ~/.ragger/ragger.ini is ALWAYS read for user-personal settings.
    """
    user_path = expand_path("~/.ragger/ragger.ini")
    has_user = os.path.exists(user_path)

    # Explicit --config replaces /etc as system config
    if cli_path:
        resolved = expand_path(cli_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Config file not found: {resolved}")
        if not has_user:
            _bootstrap_user_config()
        return (resolved, user_path)

    sys_path = system_config_path()
    has_system = os.path.exists(sys_path)

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
