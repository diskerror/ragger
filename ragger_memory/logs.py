"""
Logging setup for Ragger Memory

Four log files, all in ~/.ragger/:
- query.log   — search queries, scores, timing (toggleable)
- http.log    — HTTP server requests/responses (toggleable)
- mcp.log     — MCP JSON-RPC interactions (toggleable)
- error.log   — errors from all components (always on)
"""

import logging
import os
from pathlib import Path

from .config import LOG_DIR, QUERY_LOG_ENABLED, HTTP_LOG_ENABLED, MCP_LOG_ENABLED

_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
_initialized = False


def _make_handler(filepath: str, level=logging.DEBUG) -> logging.FileHandler:
    """Create a file handler, ensuring parent directory exists."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(path))
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


def setup_logging(verbose: bool = False, server_mode: bool = False):
    """
    Configure all Ragger loggers.
    
    Args:
        verbose: Enable DEBUG level on stderr
        server_mode: Enable INFO level on stderr (for --server)
    """
    global _initialized
    if _initialized:
        return
    _initialized = True
    
    log_dir = LOG_DIR
    
    # Stderr handler
    if verbose:
        stderr_level = logging.DEBUG
    elif server_mode:
        stderr_level = logging.INFO
    else:
        stderr_level = logging.WARNING
    
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(stderr_level)
    stderr_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    
    # Error log — always on, WARNING and above from all ragger_memory loggers
    error_handler = _make_handler(os.path.join(log_dir, 'error.log'), level=logging.WARNING)
    
    # Root ragger_memory logger
    root = logging.getLogger('ragger_memory')
    root.setLevel(logging.DEBUG)
    root.addHandler(stderr_handler)
    root.addHandler(error_handler)
    
    # Query logger
    query_logger = logging.getLogger('ragger_memory.query')
    if QUERY_LOG_ENABLED:
        query_logger.addHandler(_make_handler(os.path.join(log_dir, 'query.log')))
    
    # HTTP logger
    http_logger = logging.getLogger('ragger_memory.http')
    if HTTP_LOG_ENABLED:
        http_logger.addHandler(_make_handler(os.path.join(log_dir, 'http.log')))
    
    # MCP logger
    mcp_logger = logging.getLogger('ragger_memory.mcp')
    if MCP_LOG_ENABLED:
        mcp_logger.addHandler(_make_handler(os.path.join(log_dir, 'mcp.log')))


def get_query_logger() -> logging.Logger:
    """Logger for search queries, scores, timing."""
    return logging.getLogger('ragger_memory.query')


def get_http_logger() -> logging.Logger:
    """Logger for HTTP server requests."""
    return logging.getLogger('ragger_memory.http')


def get_mcp_logger() -> logging.Logger:
    """Logger for MCP JSON-RPC interactions."""
    return logging.getLogger('ragger_memory.mcp')


def get_error_logger() -> logging.Logger:
    """Logger for errors (always on)."""
    return logging.getLogger('ragger_memory')
