"""Tests for schema migrations — dedicated columns for collection, category, tags."""

import json
import sqlite3
import pytest
import numpy as np

from ragger_memory.sqlite_backend import SqliteBackend


class TestDedicatedColumns:
    """Test that collection, category, tags are stored in dedicated columns."""

    @pytest.fixture
    def backend(self, tmp_path):
        from unittest.mock import MagicMock
        embedder = MagicMock()
        embedder.embed.return_value = np.random.rand(384).astype(np.float32).tolist()
        backend = SqliteBackend(embedder, str(tmp_path / "test.db"))
        yield backend
        backend.close()

    def test_columns_exist(self, backend):
        cols = [row[1] for row in backend.conn.execute("PRAGMA table_info(memories)")]
        assert "collection" in cols
        assert "category" in cols
        assert "tags" in cols

    def test_store_populates_dedicated_columns(self, backend):
        mid = backend.store("test", {"collection": "docs", "category": "fact", "tags": ["a", "b"]})
        row = backend.conn.execute(
            "SELECT collection, category, tags, metadata FROM memories WHERE id = ?",
            (int(mid),)
        ).fetchone()
        assert row[0] == "docs"
        assert row[1] == "fact"
        assert row[2] == "a,b"
        # collection/category/tags should NOT be in the JSON metadata
        meta = json.loads(row[3]) if row[3] else {}
        assert "collection" not in meta
        assert "category" not in meta
        assert "tags" not in meta

    def test_store_defaults(self, backend):
        mid = backend.store("test with no meta")
        row = backend.conn.execute(
            "SELECT collection, category, tags FROM memories WHERE id = ?",
            (int(mid),)
        ).fetchone()
        assert row[0] == "memory"
        assert row[1] == ""
        assert row[2] == ""

    def test_load_all_reconstructs_metadata(self, backend):
        backend.store("test", {"collection": "reference", "category": "doc", "tags": ["x"], "source": "test.md"})
        ids, texts, embs, metas, timestamps = backend.load_all_embeddings()
        meta = metas[0]
        assert meta["collection"] == "reference"
        assert meta["category"] == "doc"
        assert meta["tags"] == ["x"]
        assert meta["source"] == "test.md"

    def test_search_by_metadata_uses_columns(self, backend):
        backend.store("one", {"collection": "docs", "category": "fact"})
        backend.store("two", {"collection": "memory", "category": "decision"})
        backend.store("three", {"collection": "docs", "category": "decision"})

        results = backend.search_by_metadata({"collection": "docs"})
        assert len(results) == 2

        results = backend.search_by_metadata({"category": "decision"})
        assert len(results) == 2

        results = backend.search_by_metadata({"collection": "docs", "category": "decision"})
        assert len(results) == 1
        assert results[0]["text"] == "three"

    def test_search_by_metadata_tags(self, backend):
        backend.store("tagged", {"tags": ["important", "ragger"]})
        backend.store("untagged", {})

        results = backend.search_by_metadata({"tags": "important"})
        assert len(results) == 1
        assert results[0]["text"] == "tagged"

    def test_search_by_metadata_mixed_filter(self, backend):
        """Filter on both dedicated column and JSON metadata field."""
        backend.store("target", {"collection": "docs", "source": "readme.md"})
        backend.store("other", {"collection": "docs", "source": "other.md"})

        results = backend.search_by_metadata({"collection": "docs", "source": "readme.md"})
        assert len(results) == 1
        assert results[0]["text"] == "target"


class TestMigrationBackfill:
    """Test that migration backfills existing data correctly."""

    @pytest.fixture
    def old_db(self, tmp_path):
        """Create a DB with old schema (no dedicated columns)."""
        db_path = str(tmp_path / "old.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                user_id INTEGER
            )
        """)
        # Insert old-style records with collection/category in JSON
        emb = np.random.rand(384).astype(np.float32).tobytes()
        conn.execute(
            "INSERT INTO memories (text, embedding, metadata, timestamp) VALUES (?, ?, ?, ?)",
            ("old memory", emb, json.dumps({"collection": "docs", "category": "reference", "tags": ["imported"], "source": "file.md"}), "2026-01-01T00:00:00")
        )
        conn.execute(
            "INSERT INTO memories (text, embedding, metadata, timestamp) VALUES (?, ?, ?, ?)",
            ("plain memory", emb, json.dumps({"source": "chat"}), "2026-01-02T00:00:00")
        )
        conn.execute(
            "INSERT INTO memories (text, embedding, metadata, timestamp) VALUES (?, ?, ?, ?)",
            ("kept memory", emb, json.dumps({"collection": "common", "keep": True, "bad": True, "source": "admin"}), "2026-01-03T00:00:00")
        )
        conn.commit()
        conn.close()
        return db_path

    def test_migration_backfills(self, old_db):
        from ragger_memory.migrations import migrate_add_dedicated_columns
        conn = sqlite3.connect(old_db)
        migrate_add_dedicated_columns(conn, "memories")

        row = conn.execute("SELECT collection, category, tags, metadata FROM memories WHERE id = 1").fetchone()
        assert row[0] == "docs"
        assert row[1] == "reference"
        assert row[2] == "imported"
        # JSON metadata should no longer have collection/category/tags
        meta = json.loads(row[3]) if row[3] else {}
        assert "collection" not in meta
        assert "category" not in meta
        assert "tags" not in meta
        assert meta["source"] == "file.md"

        # Plain memory gets defaults
        row2 = conn.execute("SELECT collection, category, tags FROM memories WHERE id = 2").fetchone()
        assert row2[0] == "memory"
        assert row2[1] == ""
        assert row2[2] == ""

        # Memory with keep/bad boolean flags converted to tags
        row3 = conn.execute("SELECT collection, tags, metadata FROM memories WHERE id = 3").fetchone()
        assert row3[0] == "common"
        assert "keep" in row3[1].split(",")
        assert "bad" in row3[1].split(",")
        # keep/bad should be removed from JSON metadata
        meta3 = json.loads(row3[2]) if row3[2] else {}
        assert "keep" not in meta3
        assert "bad" not in meta3
        assert meta3["source"] == "admin"

        conn.close()
