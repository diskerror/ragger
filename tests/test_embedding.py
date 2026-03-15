"""
Tests for the mock embedder (and embedder interface contract).

These test the MockEmbedder to ensure the test infrastructure is sound.
Real Embedder tests (loading the actual model) are slow and marked separately.
"""
import numpy as np
import pytest

from tests.conftest import MockEmbedder


class TestMockEmbedder:
    """Verify MockEmbedder behaves like the real Embedder."""
    
    def test_output_shape(self):
        emb = MockEmbedder()
        vec = emb.encode("test text")
        assert vec.shape == (384,)
    
    def test_output_dtype(self):
        emb = MockEmbedder()
        vec = emb.encode("test text")
        assert vec.dtype == np.float32
    
    def test_deterministic(self):
        emb = MockEmbedder()
        v1 = emb.encode("hello world")
        v2 = emb.encode("hello world")
        np.testing.assert_array_equal(v1, v2)
    
    def test_different_texts_different_vectors(self):
        emb = MockEmbedder()
        v1 = emb.encode("hello world")
        v2 = emb.encode("goodbye moon")
        assert not np.allclose(v1, v2)
    
    def test_unit_normalized(self):
        emb = MockEmbedder()
        vec = emb.encode("test normalization")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5
