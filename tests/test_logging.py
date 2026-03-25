"""
Tests for the logging module.
"""
import logging
import os
from unittest.mock import patch

import pytest

from ragger_memory.logs import setup_logging, get_query_logger, get_http_logger


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset the logging module's _initialized flag between tests."""
    import ragger_memory.logs as logs_mod
    logs_mod._initialized = False
    # Clean up handlers added by previous tests
    for name in ('ragger_memory', 'ragger_memory.query', 'ragger_memory.http', 'ragger_memory.mcp'):
        logger = logging.getLogger(name)
        logger.handlers.clear()
    yield
    logs_mod._initialized = False
    for name in ('ragger_memory', 'ragger_memory.query', 'ragger_memory.http', 'ragger_memory.mcp'):
        logger = logging.getLogger(name)
        logger.handlers.clear()


class TestSetupLogging:
    @patch('ragger_memory.logs.LOG_DIR')
    @patch('ragger_memory.logs.QUERY_LOG_ENABLED', True)
    @patch('ragger_memory.logs.HTTP_LOG_ENABLED', True)
    @patch('ragger_memory.logs.MCP_LOG_ENABLED', True)
    def test_setup_logging_creates_files(self, mock_log_dir, tmp_path):
        mock_log_dir.__str__ = lambda s: str(tmp_path)
        # Patch LOG_DIR to be the tmp_path string
        with patch('ragger_memory.logs.LOG_DIR', str(tmp_path)):
            setup_logging(server_mode=True)
        
        assert (tmp_path / 'error.log').exists()
        assert (tmp_path / 'query.log').exists()
        assert (tmp_path / 'http.log').exists()

    @patch('ragger_memory.logs.QUERY_LOG_ENABLED', True)
    def test_query_log_writes(self, tmp_path):
        with patch('ragger_memory.logs.LOG_DIR', str(tmp_path)), \
             patch('ragger_memory.logs.expand_path', return_value=str(tmp_path)):
            setup_logging(server_mode=True)
        
        query_logger = get_query_logger()
        query_logger.info("search query: test embedding lookup")
        
        # Flush handlers
        for h in query_logger.handlers:
            h.flush()
        
        content = (tmp_path / 'query.log').read_text()
        assert "test embedding lookup" in content

    @patch('ragger_memory.logs.HTTP_LOG_ENABLED', True)
    def test_http_log_writes(self, tmp_path):
        with patch('ragger_memory.logs.LOG_DIR', str(tmp_path)), \
             patch('ragger_memory.logs.expand_path', return_value=str(tmp_path)):
            setup_logging(server_mode=True)
        
        http_logger = get_http_logger()
        http_logger.info("POST /store 200")
        
        for h in http_logger.handlers:
            h.flush()
        
        content = (tmp_path / 'http.log').read_text()
        assert "POST /store 200" in content

    @patch('ragger_memory.logs.QUERY_LOG_ENABLED', False)
    @patch('ragger_memory.logs.HTTP_LOG_ENABLED', False)
    @patch('ragger_memory.logs.MCP_LOG_ENABLED', False)
    def test_logging_disabled(self, tmp_path):
        with patch('ragger_memory.logs.LOG_DIR', str(tmp_path)), \
             patch('ragger_memory.logs.expand_path', return_value=str(tmp_path)):
            setup_logging(server_mode=True)
        
        get_query_logger().info("should not appear")
        get_http_logger().info("should not appear")
        
        # query.log and http.log should not exist
        assert not (tmp_path / 'query.log').exists()
        assert not (tmp_path / 'http.log').exists()
