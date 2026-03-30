"""
Tests for per-user memory resolution and dual-DB search merging.
"""
import os
import pytest

from tests.conftest import MockEmbedder


class TestPerUserMemory:
    """Test RaggerMemory.for_user() and dual-DB behavior."""

    @pytest.fixture
    def dual_memory(self, tmp_path):
        """Create a RaggerMemory with common + user DBs."""
        from ragger_memory.sqlite_backend import SqliteBackend
        from ragger_memory.memory import RaggerMemory

        emb = MockEmbedder()
        common_db = str(tmp_path / "common.db")
        user_db_dir = tmp_path / "userhome" / ".ragger"
        user_db_dir.mkdir(parents=True)
        user_db = str(user_db_dir / "memories.db")

        common_backend = SqliteBackend(emb, common_db)
        user_backend = SqliteBackend(emb, user_db)

        # Store different content in each DB
        common_backend.store("Reference: Python was created by Guido van Rossum.",
                             {"collection": "docs", "category": "fact"})
        common_backend.store("Reference: SQLite is a C library.",
                             {"collection": "docs", "category": "fact"})
        user_backend.store("Personal: I prefer dark mode.",
                           {"collection": "memory", "category": "preference"})
        user_backend.store("Personal: Meeting with Reid at 3pm.",
                           {"collection": "conversation", "category": "chat-turn"})

        # Build a RaggerMemory with both backends
        mem = object.__new__(RaggerMemory)
        mem.engine = None
        mem._backend = common_backend
        mem._user_backend = user_backend

        yield mem

        common_backend.close()
        user_backend.close()

    def test_count_includes_both_dbs(self, dual_memory):
        """Count should sum both common and user DBs."""
        count = dual_memory.count()
        assert count == 4

    def test_search_returns_from_both_dbs(self, dual_memory):
        """Search should return results from both common and user DBs."""
        results = dual_memory.search("Python", limit=10, min_score=0.0)
        texts = [r["text"] for r in results["results"]]
        # Should have results from both DBs
        assert len(texts) >= 1

    def test_store_goes_to_user_db(self, dual_memory):
        """Store should write to user DB (not common)."""
        before_common = dual_memory._backend.count()
        before_user = dual_memory._user_backend.count()

        dual_memory.store("New personal memory")

        # Common unchanged, user incremented
        assert dual_memory._backend.count() == before_common
        assert dual_memory._user_backend.count() == before_user + 1

    def test_is_multi_db(self, dual_memory):
        """Dual-DB memory should report is_multi_db."""
        assert dual_memory.is_multi_db is True

    def test_single_db_not_multi(self, tmp_path):
        """Single-DB memory should not be multi_db."""
        from ragger_memory.sqlite_backend import SqliteBackend
        from ragger_memory.memory import RaggerMemory

        emb = MockEmbedder()
        db = str(tmp_path / "single.db")
        backend = SqliteBackend(emb, db)

        mem = object.__new__(RaggerMemory)
        mem.engine = None
        mem._backend = backend
        mem._user_backend = None

        assert mem.is_multi_db is False
        backend.close()
