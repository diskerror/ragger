"""
Tests for the SQLite backend.

Uses MockEmbedder from conftest to avoid loading the real model.
"""
import json
import sqlite3

import numpy as np
import pytest

from ragger_memory.sqlite_backend import SqliteBackend


class TestSqliteSchema:
    """Tests for schema creation and migration."""
    
    def test_creates_tables(self, sqlite_backend, tmp_db):
        """All expected tables should exist after init."""
        conn = sqlite3.connect(tmp_db)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        
        assert "memories" in tables
        assert "memory_usage" in tables
        assert "bm25_index" in tables
    
    def test_wal_mode_enabled(self, sqlite_backend, tmp_db):
        """WAL journal mode should be active."""
        conn = sqlite3.connect(tmp_db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"
    
    def test_foreign_keys_enabled(self, sqlite_backend):
        """Foreign keys should be enforced."""
        fk = sqlite_backend.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
    
    def test_usage_table_has_fk(self, sqlite_backend):
        """memory_usage should have a FK to memories."""
        fk_info = sqlite_backend.conn.execute(
            "PRAGMA foreign_key_list(memory_usage)"
        ).fetchall()
        assert len(fk_info) > 0
        # FK should reference 'memories' table
        assert any("memories" in str(row) for row in fk_info)


class TestSqliteStoreAndRetrieve:
    """Tests for storing and retrieving memories."""
    
    def test_store_returns_id(self, sqlite_backend):
        mid = sqlite_backend.store("test memory")
        assert mid is not None
        assert isinstance(mid, str)
        assert int(mid) > 0
    
    def test_count_increases(self, sqlite_backend):
        assert sqlite_backend.count() == 0
        sqlite_backend.store("first")
        assert sqlite_backend.count() == 1
        sqlite_backend.store("second")
        assert sqlite_backend.count() == 2
    
    def test_store_with_metadata(self, sqlite_backend):
        meta = {"source": "test.md", "collection": "docs"}
        mid = sqlite_backend.store("memory with metadata", meta)
        
        row = sqlite_backend.conn.execute(
            "SELECT metadata, collection FROM memories WHERE id = ?", (int(mid),)
        ).fetchone()
        stored_meta = json.loads(row[0]) if row[0] else {}
        assert stored_meta["source"] == "test.md"
        # collection is now a dedicated column, not in JSON metadata
        assert row[1] == "docs"
    
    def test_store_empty_metadata(self, sqlite_backend):
        mid = sqlite_backend.store("no metadata")
        row = sqlite_backend.conn.execute(
            "SELECT metadata, collection, category, tags FROM memories WHERE id = ?", (int(mid),)
        ).fetchone()
        stored_meta = json.loads(row[0]) if row[0] else {}
        assert isinstance(stored_meta, dict)
        # Dedicated columns get defaults
        assert row[1] == "memory"
        assert row[2] == ""
        assert row[3] == ""
    
    def test_embedding_stored_as_blob(self, sqlite_backend):
        sqlite_backend.store("test embedding storage")
        row = sqlite_backend.conn.execute(
            "SELECT embedding FROM memories WHERE id = 1"
        ).fetchone()
        blob = row[0]
        embedding = np.frombuffer(blob, dtype=np.float32)
        assert embedding.shape == (384,)
    
    def test_load_all_embeddings(self, sqlite_backend):
        sqlite_backend.store("first memory")
        sqlite_backend.store("second memory")
        
        ids, texts, embeddings, metadata, timestamps = sqlite_backend.load_all_embeddings()
        
        assert len(ids) == 2
        assert len(texts) == 2
        assert embeddings.shape == (2, 384)
        assert len(metadata) == 2
        assert len(timestamps) == 2
    
    def test_load_all_empty_db(self, sqlite_backend):
        ids, texts, embeddings, metadata, timestamps = sqlite_backend.load_all_embeddings()
        assert ids == []
        assert texts == []
        assert len(embeddings) == 0


class TestSqliteSearch:
    """Tests for search functionality."""
    
    def test_search_empty_db(self, sqlite_backend):
        result = sqlite_backend.search("anything")
        assert result["results"] == []
    
    def test_search_returns_results(self, sqlite_backend):
        sqlite_backend.store("Python programming language", {"collection": "memory"})
        sqlite_backend.store("cooking recipes for dinner", {"collection": "memory"})
        
        result = sqlite_backend.search("Python programming", min_score=-1.0)
        assert len(result["results"]) > 0
        assert "timing" in result
    
    def test_search_respects_limit(self, sqlite_backend):
        for i in range(10):
            sqlite_backend.store(f"memory number {i}", {"collection": "memory"})
        
        result = sqlite_backend.search("memory", limit=3, min_score=-1.0)
        assert len(result["results"]) <= 3
    
    def test_search_has_timing(self, sqlite_backend):
        sqlite_backend.store("test timing", {"collection": "memory"})
        result = sqlite_backend.search("timing", min_score=-1.0)
        timing = result["timing"]
        assert "embedding_ms" in timing
        assert "search_ms" in timing
        assert "total_ms" in timing
        assert "corpus_size" in timing
    
    def test_search_default_returns_all(self, sqlite_backend):
        sqlite_backend.store("Python docs", {"collection": "docs"})
        sqlite_backend.store("Python memory", {"collection": "memory"})
        
        # Default search (all collections) — min_score=-1 for mock embeddings
        result = sqlite_backend.search("Python", min_score=-1.0)
        texts = [r["text"] for r in result["results"]]
        assert any("memory" in t for t in texts)
        assert any("docs" in t for t in texts)
    
    def test_search_narrow_to_single_collection(self, sqlite_backend):
        sqlite_backend.store("Python docs", {"collection": "docs"})
        sqlite_backend.store("Python memory", {"collection": "memory"})
        
        # Narrow to memory only
        result = sqlite_backend.search("Python", collections=["memory"], min_score=-1.0)
        texts = [r["text"] for r in result["results"]]
        assert any("memory" in t for t in texts)
        assert not any("docs" in t for t in texts)
    
    def test_search_specific_collection(self, sqlite_backend):
        sqlite_backend.store("Python docs", {"collection": "docs"})
        sqlite_backend.store("Python memory", {"collection": "memory"})
        
        result = sqlite_backend.search("Python", collections=["docs"], min_score=-1.0)
        texts = [r["text"] for r in result["results"]]
        assert any("docs" in t for t in texts)
    
    def test_search_all_collections(self, sqlite_backend):
        sqlite_backend.store("alpha text", {"collection": "docs"})
        sqlite_backend.store("bravo text", {"collection": "memory"})
        sqlite_backend.store("charlie text", {"collection": "notes"})
        
        # min_score=-1 to include all results (mock embeddings can produce negative cosine)
        result = sqlite_backend.search("text", collections=["*"], min_score=-1.0)
        assert len(result["results"]) == 3
    
    def test_search_result_has_score(self, sqlite_backend):
        sqlite_backend.store("test score", {"collection": "memory"})
        result = sqlite_backend.search("score", min_score=-1.0)
        assert "score" in result["results"][0]
        assert isinstance(result["results"][0]["score"], float)
    
    def test_search_min_score_filtering(self, sqlite_backend):
        sqlite_backend.store("test document", {"collection": "memory"})
        # With an impossibly high min_score, nothing should match
        result = sqlite_backend.search("test", min_score=0.9999)
        assert len(result["results"]) == 0


class TestSqliteCascadeDelete:
    """Tests for FK cascade behavior."""
    
    def test_cascade_delete_usage(self, sqlite_backend):
        """Deleting a memory should cascade-delete its usage records."""
        mid = sqlite_backend.store("test cascade", {"collection": "memory"})
        
        # Manually insert usage
        sqlite_backend.conn.execute(
            "INSERT INTO memory_usage (memory_id, timestamp) VALUES (?, ?)",
            (int(mid), "2026-01-01T00:00:00")
        )
        sqlite_backend.conn.commit()
        
        # Verify usage exists
        count = sqlite_backend.conn.execute(
            "SELECT COUNT(*) FROM memory_usage WHERE memory_id = ?", (int(mid),)
        ).fetchone()[0]
        assert count == 1
        
        # Delete the memory
        sqlite_backend.conn.execute("DELETE FROM memories WHERE id = ?", (int(mid),))
        sqlite_backend.conn.commit()
        
        # Usage should be gone
        count = sqlite_backend.conn.execute(
            "SELECT COUNT(*) FROM memory_usage WHERE memory_id = ?", (int(mid),)
        ).fetchone()[0]
        assert count == 0


class TestSqliteUsageTracking:
    """Tests for usage tracking."""
    
    def test_search_tracks_usage(self, sqlite_backend):
        sqlite_backend.store("tracked memory", {"collection": "memory"})
        # Use min_score=-1 to ensure results aren't filtered out (mock embeddings)
        sqlite_backend.search("tracked", min_score=-1.0)
        
        count = sqlite_backend.conn.execute(
            "SELECT COUNT(*) FROM memory_usage"
        ).fetchone()[0]
        assert count > 0
    
    def test_usage_references_valid_memory(self, sqlite_backend):
        sqlite_backend.store("valid ref", {"collection": "memory"})
        sqlite_backend.search("valid")
        
        rows = sqlite_backend.conn.execute(
            "SELECT memory_id FROM memory_usage"
        ).fetchall()
        for row in rows:
            mem = sqlite_backend.conn.execute(
                "SELECT id FROM memories WHERE id = ?", (row[0],)
            ).fetchone()
            assert mem is not None


class TestBM25IndexPersistence:
    """Tests for persistent BM25 index table."""
    
    def test_store_indexes_tokens(self, sqlite_backend):
        """Storing a document should populate bm25_index."""
        sqlite_backend.store("the quick brown fox jumps", {"collection": "memory"})
        
        rows = sqlite_backend.conn.execute(
            "SELECT token, term_freq FROM bm25_index"
        ).fetchall()
        tokens = {row[0]: row[1] for row in rows}
        
        # "the" is a stop word, should not be indexed
        assert "the" not in tokens
        assert "quick" in tokens
        assert tokens["quick"] == 1
    
    def test_store_counts_term_frequency(self, sqlite_backend):
        """Repeated terms should have correct frequency."""
        sqlite_backend.store("fox fox fox dog dog", {"collection": "memory"})
        
        rows = sqlite_backend.conn.execute(
            "SELECT token, term_freq FROM bm25_index"
        ).fetchall()
        tokens = {row[0]: row[1] for row in rows}
        
        assert tokens["fox"] == 3
        assert tokens["dog"] == 2
    
    def test_cascade_delete_bm25(self, sqlite_backend):
        """Deleting a memory should cascade-delete its BM25 index rows."""
        mem_id = sqlite_backend.store("cascade test tokens here", {"collection": "memory"})
        
        count_before = sqlite_backend.conn.execute(
            "SELECT COUNT(*) FROM bm25_index"
        ).fetchone()[0]
        assert count_before > 0
        
        sqlite_backend.conn.execute(
            f"DELETE FROM memories WHERE id = ?", (int(mem_id),)
        )
        sqlite_backend.conn.commit()
        
        count_after = sqlite_backend.conn.execute(
            "SELECT COUNT(*) FROM bm25_index"
        ).fetchone()[0]
        assert count_after == 0
    
    def test_bm25_index_count(self, sqlite_backend):
        """bm25_index_count should match number of stored docs."""
        assert sqlite_backend.bm25_index_count() == 0
        
        sqlite_backend.store("first document", {"collection": "memory"})
        sqlite_backend.store("second document", {"collection": "memory"})
        
        assert sqlite_backend.bm25_index_count() == 2
    
    def test_rebuild_bm25_index(self, sqlite_backend):
        """rebuild_bm25_index should repopulate from all documents."""
        sqlite_backend.store("rebuild test one", {"collection": "memory"})
        sqlite_backend.store("rebuild test two", {"collection": "memory"})
        
        # Clear the index manually
        sqlite_backend.conn.execute("DELETE FROM bm25_index")
        sqlite_backend.conn.commit()
        assert sqlite_backend.bm25_index_count() == 0
        
        # Rebuild
        count = sqlite_backend.rebuild_bm25_index()
        assert count == 2
        assert sqlite_backend.bm25_index_count() == 2
    
    def test_load_bm25_index(self, sqlite_backend):
        """load_bm25_index should return per-doc token frequencies."""
        sqlite_backend.store("alpha beta gamma", {"collection": "memory"})
        
        data = sqlite_backend.load_bm25_index()
        assert data['count'] == 1
        
        # Should have one doc's worth of token freqs
        doc_freqs = data['doc_freqs']
        assert len(doc_freqs) == 1
        tf = list(doc_freqs.values())[0]
        assert "alpha" in tf
        assert "beta" in tf
        assert "gamma" in tf


class TestRebuildEmbeddings:
    """Tests for rebuild_embeddings functionality."""
    
    def test_rebuild_embeddings_updates_all(self, sqlite_backend, mock_embedder):
        """rebuild_embeddings should update all embeddings with current model."""
        # Store two memories
        mid1 = sqlite_backend.store("first memory", {"collection": "memory"})
        mid2 = sqlite_backend.store("second memory", {"collection": "memory"})
        
        # Get original embeddings
        emb1_before = sqlite_backend.conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (int(mid1),)
        ).fetchone()[0]
        emb2_before = sqlite_backend.conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (int(mid2),)
        ).fetchone()[0]
        
        # Change mock embedder behavior to produce different embeddings
        # (in real use, this would be switching to a different model)
        original_encode = mock_embedder.encode
        
        def new_encode(text):
            # Return different embeddings (scaled by 2)
            orig = original_encode(text)
            return [x * 2.0 for x in orig]
        
        mock_embedder.encode = new_encode
        
        # Rebuild embeddings
        count = sqlite_backend.rebuild_embeddings(mock_embedder)
        assert count == 2
        
        # Get new embeddings
        emb1_after = sqlite_backend.conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (int(mid1),)
        ).fetchone()[0]
        emb2_after = sqlite_backend.conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (int(mid2),)
        ).fetchone()[0]
        
        # Embeddings should be different
        assert emb1_before != emb1_after
        assert emb2_before != emb2_after
        
        # Verify they're actually scaled
        arr1_before = np.frombuffer(emb1_before, dtype=np.float32)
        arr1_after = np.frombuffer(emb1_after, dtype=np.float32)
        # Should be approximately 2x (allowing for floating point precision)
        np.testing.assert_array_almost_equal(arr1_after, arr1_before * 2.0, decimal=5)
    
    def test_rebuild_embeddings_count(self, sqlite_backend, mock_embedder):
        """rebuild_embeddings should return correct document count."""
        # Store multiple memories
        for i in range(5):
            sqlite_backend.store(f"memory {i}", {"collection": "memory"})
        
        count = sqlite_backend.rebuild_embeddings(mock_embedder)
        assert count == 5
    
    def test_rebuild_embeddings_empty_db(self, sqlite_backend, mock_embedder):
        """rebuild_embeddings on empty database should return 0."""
        count = sqlite_backend.rebuild_embeddings(mock_embedder)
        assert count == 0
