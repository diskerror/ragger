"""
Shared fixtures for Ragger Memory tests.

Uses a mock embedder to avoid loading the real sentence-transformers model,
keeping tests fast (~ms instead of seconds for model load).
"""
import os
import tempfile

import numpy as np
import pytest


class MockEmbedder:
    """
    Deterministic mock embedder for testing.
    
    Generates repeatable 384-dim vectors from text content so that
    similar texts produce similar embeddings (via simple hash-based seeding).
    """
    DIMS = 384
    
    def encode(self, text: str, **kwargs) -> np.ndarray:
        """Generate a deterministic pseudo-embedding from text."""
        # Seed from text hash for repeatability
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.DIMS).astype(np.float32)
        # Normalize to unit length (like real sentence-transformers output)
        vec /= np.linalg.norm(vec)
        return vec


@pytest.fixture
def mock_embedder():
    """Provide a MockEmbedder instance."""
    return MockEmbedder()


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test_memories.db")


@pytest.fixture
def sqlite_backend(mock_embedder, tmp_db):
    """Provide a fresh SqliteBackend with mock embedder and temp DB."""
    from ragger_memory.backend.sqlite import SqliteBackend
    backend = SqliteBackend(mock_embedder, tmp_db)
    yield backend
    backend.close()
