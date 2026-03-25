"""
Tests for rebuild-embeddings functionality.
"""
import numpy as np
import pytest
from unittest.mock import patch

from ragger_memory.memory import RaggerMemory


class TestRebuildEmbeddings:
    """Tests for the rebuild_embeddings method."""
    
    @pytest.fixture
    def memory(self, tmp_path, mock_embedder):
        """Provide a RaggerMemory instance with mock embedder."""
        db = str(tmp_path / "rebuild_test.db")
        with patch('ragger_memory.memory.Embedder', return_value=mock_embedder):
            mem = RaggerMemory(uri=db, engine="sqlite")
            yield mem
            mem.close()
    
    def test_rebuild_empty_db(self, memory):
        """Rebuild should work on empty database."""
        count = memory.rebuild_embeddings()
        assert count == 0
    
    def test_rebuild_updates_all_embeddings(self, memory):
        """Rebuild should update embeddings for all documents."""
        # Store some test documents
        memory.store("first document", {"collection": "test"})
        memory.store("second document", {"collection": "test"})
        memory.store("third document", {"collection": "test"})
        
        assert memory.count() == 3
        
        # Rebuild embeddings
        count = memory.rebuild_embeddings()
        
        # Should return count of updated documents
        assert count == 3
    
    def test_rebuild_count_matches_records(self, memory):
        """Rebuild should return count matching number of records."""
        # Add various documents
        for i in range(10):
            memory.store(f"document number {i}", {"index": i})
        
        assert memory.count() == 10
        
        count = memory.rebuild_embeddings()
        assert count == 10
    
    def test_embeddings_match_model_dimensions(self, tmp_path, mock_embedder):
        """Rebuilt embeddings should have correct dimensions."""
        db = str(tmp_path / "dims_test.db")
        
        with patch('ragger_memory.memory.Embedder', return_value=mock_embedder):
            mem = RaggerMemory(uri=db, engine="sqlite")
            
            # Store a document
            memory_id = mem.store("test document")
            
            # Get the backend to access raw embeddings
            backend = mem._backend
            
            # Load embeddings directly from database
            ids, texts, embeddings, metadata, timestamps = backend.load_all_embeddings()
            
            # Check original embedding dimensions
            assert embeddings.shape == (1, mock_embedder.DIMS)
            
            # Rebuild embeddings
            count = mem.rebuild_embeddings()
            assert count == 1
            
            # Load updated embeddings
            ids, texts, embeddings, metadata, timestamps = backend.load_all_embeddings()
            
            # Check dimensions match
            assert embeddings.shape == (1, mock_embedder.DIMS)
            assert embeddings.dtype == np.float32
            
            mem.close()
    
    def test_rebuild_preserves_metadata(self, memory):
        """Rebuild should not affect metadata."""
        # Store with metadata
        meta = {"source": "test.md", "category": "documentation", "tags": ["important"]}
        memory_id = memory.store("test content", meta)
        
        # Rebuild
        memory.rebuild_embeddings()
        
        # Search and verify metadata preserved
        results = memory.search_by_metadata({"category": "documentation"})
        assert len(results) == 1
        assert results[0]["metadata"]["source"] == "test.md"
        assert results[0]["metadata"]["category"] == "documentation"
    
    def test_rebuild_with_different_model(self, tmp_path):
        """Simulate rebuilding with a different embedding model."""
        from tests.conftest import MockEmbedder
        
        db = str(tmp_path / "model_change.db")
        
        # First embedder (original)
        embedder1 = MockEmbedder()
        
        with patch('ragger_memory.memory.Embedder', return_value=embedder1):
            mem = RaggerMemory(uri=db, engine="sqlite")
            mem.store("document to re-embed")
            
            # Get original embedding
            backend = mem._backend
            ids1, texts1, embeddings1, meta1, ts1 = backend.load_all_embeddings()
            original_embedding = embeddings1[0].copy()
            
            mem.close()
        
        # Second embedder (simulates model update)
        class DifferentMockEmbedder(MockEmbedder):
            """Mock embedder that produces different vectors."""
            def encode(self, text: str, **kwargs) -> np.ndarray:
                # Add offset to make vectors different
                vec = super().encode(text, **kwargs)
                return vec + 0.1  # Shift all values
        
        embedder2 = DifferentMockEmbedder()
        
        with patch('ragger_memory.memory.Embedder', return_value=embedder2):
            mem = RaggerMemory(uri=db, engine="sqlite")
            
            # Rebuild with new model
            count = mem.rebuild_embeddings()
            assert count == 1
            
            # Get new embedding
            backend = mem._backend
            ids2, texts2, embeddings2, meta2, ts2 = backend.load_all_embeddings()
            new_embedding = embeddings2[0]
            
            # Embeddings should be different
            assert not np.allclose(original_embedding, new_embedding)
            
            mem.close()
    
    def test_rebuild_with_multi_db(self, tmp_path, mock_embedder):
        """Rebuild should work with multi-DB setup."""
        common_db = str(tmp_path / "common.db")
        user_db = str(tmp_path / "user.db")
        
        with patch('ragger_memory.memory.Embedder', return_value=mock_embedder):
            mem = RaggerMemory(uri=common_db, engine="sqlite", user_db_path=user_db)
            
            # Store to both databases
            mem.store("common document", common=True)
            mem.store("user document", common=False)
            
            assert mem.count() == 2
            
            # Rebuild should update both
            count = mem.rebuild_embeddings()
            assert count == 2
            
            mem.close()
