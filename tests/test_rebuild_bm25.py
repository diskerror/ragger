"""
Tests for rebuild-bm25 functionality.
"""
import pytest


class TestRebuildBm25:
    """Tests for SqliteBackend.rebuild_bm25_index()."""

    def test_rebuild_empty_db(self, sqlite_backend):
        """Rebuild on empty DB returns 0, doesn't error."""
        count = sqlite_backend.rebuild_bm25_index()
        assert count == 0

    def test_rebuild_restores_index(self, sqlite_backend):
        """After clearing bm25_index, rebuild restores BM25 token rows."""
        sqlite_backend.store("The capital of France is Paris.")
        sqlite_backend.store("SQLite is a lightweight database engine.")
        sqlite_backend.store("Python is a programming language.")

        # Verify BM25 index has rows
        row_count = sqlite_backend.conn.execute(
            "SELECT COUNT(*) FROM bm25_index"
        ).fetchone()[0]
        assert row_count > 0

        # Nuke the BM25 index
        sqlite_backend.conn.execute("DELETE FROM bm25_index")
        sqlite_backend.conn.commit()
        assert sqlite_backend.conn.execute(
            "SELECT COUNT(*) FROM bm25_index"
        ).fetchone()[0] == 0

        # Rebuild
        count = sqlite_backend.rebuild_bm25_index()
        assert count == 3

        # BM25 index should have rows again
        rebuilt_count = sqlite_backend.conn.execute(
            "SELECT COUNT(*) FROM bm25_index"
        ).fetchone()[0]
        assert rebuilt_count == row_count  # same as before

    def test_rebuild_count_matches_stored(self, sqlite_backend):
        """Rebuild returns the correct document count."""
        for i in range(5):
            sqlite_backend.store(f"Document number {i} with some content.")
        
        count = sqlite_backend.rebuild_bm25_index()
        assert count == 5

    def test_rebuild_idempotent(self, sqlite_backend):
        """Running rebuild twice gives same result."""
        sqlite_backend.store("Test document for idempotency.")
        
        count1 = sqlite_backend.rebuild_bm25_index()
        count2 = sqlite_backend.rebuild_bm25_index()
        assert count1 == count2 == 1
