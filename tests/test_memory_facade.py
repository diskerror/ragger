"""
Tests for the RaggerMemory facade/factory.

Tests the public API that users interact with, using SQLite backend
with mock embedder.
"""
import pytest

from unittest.mock import patch, MagicMock
from ragger_memory.memory import RaggerMemory


class TestRaggerMemoryInit:
    """Tests for factory initialization."""
    
    @patch('ragger_memory.memory.Embedder')
    def test_default_engine_is_sqlite(self, mock_embedder_cls):
        mock_embedder_cls.return_value = MagicMock()
        with patch('ragger_memory.config.STORAGE_ENGINE', 'sqlite'):
            with patch('ragger_memory.sqlite_backend.SqliteBackend') as mock_backend:
                mem = RaggerMemory()
                assert mem.engine == "sqlite"
    
    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown storage engine"):
            RaggerMemory(engine="flatfile")
    
    def test_context_manager(self, tmp_path, mock_embedder):
        """RaggerMemory should work as a context manager."""
        db = str(tmp_path / "ctx.db")
        with patch('ragger_memory.memory.Embedder', return_value=mock_embedder):
            with RaggerMemory(uri=db, engine="sqlite") as mem:
                mem.store("context manager test")
                assert mem.count() == 1


class TestRaggerMemoryAPI:
    """Tests for the public store/search/count API."""
    
    @pytest.fixture
    def memory(self, tmp_path, mock_embedder):
        db = str(tmp_path / "api.db")
        with patch('ragger_memory.memory.Embedder', return_value=mock_embedder):
            mem = RaggerMemory(uri=db, engine="sqlite")
            yield mem
            mem.close()
    
    def test_store_and_count(self, memory):
        assert memory.count() == 0
        memory.store("first memory")
        assert memory.count() == 1
    
    def test_store_returns_id(self, memory):
        mid = memory.store("test")
        assert mid is not None
    
    def test_search_returns_dict(self, memory):
        memory.store("searchable content", {"collection": "memory"})
        result = memory.search("searchable")
        assert isinstance(result, dict)
        assert "results" in result
        assert "timing" in result
    
    def test_store_with_metadata(self, memory):
        memory.store("with meta", {"source": "test.md", "collection": "docs"})
        result = memory.search("with meta", collections=["docs"])
        assert len(result["results"]) > 0
        assert result["results"][0]["metadata"]["source"] == "test.md"
