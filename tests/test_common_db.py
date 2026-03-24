"""
Tests for common (shared) memory DB in multi-user mode.
"""
import pytest
import tempfile
import os
from ragger_memory.memory import RaggerMemory


def _temp_db():
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False)
    path = f.name
    f.close()
    return path


def _cleanup(path):
    for suffix in ['', '-wal', '-shm']:
        try:
            os.unlink(path + suffix)
        except FileNotFoundError:
            pass


class TestCommonDB:
    """Test dual-DB mode: common (shared) + user (private)."""

    def test_single_db_mode(self):
        """Default: single DB, no common/user split."""
        db = _temp_db()
        try:
            mem = RaggerMemory(uri=db)
            assert not mem.is_multi_db
            mem.store("test memory")
            assert mem.count() == 1
            mem.close()
        finally:
            _cleanup(db)

    def test_dual_db_mode_created(self):
        """Providing user_db_path enables multi-DB."""
        common = _temp_db()
        user = _temp_db()
        try:
            mem = RaggerMemory(uri=common, user_db_path=user)
            assert mem.is_multi_db
            mem.close()
        finally:
            _cleanup(common)
            _cleanup(user)

    def test_store_routes_to_user_by_default(self):
        """Default store goes to user DB."""
        common = _temp_db()
        user = _temp_db()
        try:
            mem = RaggerMemory(uri=common, user_db_path=user)
            mem.store("user memory", {"collection": "memory"})
            # User DB has it, common doesn't
            assert mem._user_backend.count() == 1
            assert mem._backend.count() == 0
            assert mem.count() == 1
            mem.close()
        finally:
            _cleanup(common)
            _cleanup(user)

    def test_store_common_routes_to_common_db(self):
        """common=True stores to shared DB."""
        common = _temp_db()
        user = _temp_db()
        try:
            mem = RaggerMemory(uri=common, user_db_path=user)
            mem.store("shared fact", {"collection": "memory"}, common=True)
            assert mem._backend.count() == 1
            assert mem._user_backend.count() == 0
            mem.close()
        finally:
            _cleanup(common)
            _cleanup(user)

    def test_common_store_auto_adds_keep_tag(self):
        """Storing to common DB auto-sets 'keep' tag."""
        common = _temp_db()
        user = _temp_db()
        try:
            mem = RaggerMemory(uri=common, user_db_path=user)
            mid = mem.store("important shared fact", {"collection": "memory"}, common=True)
            results = mem._backend.search_by_metadata({"collection": "memory"})
            assert len(results) == 1
            tags = results[0].get("metadata", {}).get("tags", "")
            if isinstance(tags, list):
                assert "keep" in tags
            else:
                assert "keep" in tags.split(",")
            mem.close()
        finally:
            _cleanup(common)
            _cleanup(user)

    def test_search_merges_both_dbs(self):
        """Search returns results from both DBs, ranked by score."""
        common = _temp_db()
        user = _temp_db()
        try:
            mem = RaggerMemory(uri=common, user_db_path=user)
            mem.store("the cat sat on the mat", {"collection": "memory"})  # user
            mem.store("dogs are loyal companions", {"collection": "memory"}, common=True)  # common
            results = mem.search("animals and pets", limit=10, min_score=0.0)
            assert len(results["results"]) == 2
            assert results["timing"]["corpus_size"] == 2
            mem.close()
        finally:
            _cleanup(common)
            _cleanup(user)

    def test_count_sums_both_dbs(self):
        """Count includes both common and user DBs."""
        common = _temp_db()
        user = _temp_db()
        try:
            mem = RaggerMemory(uri=common, user_db_path=user)
            mem.store("user memory 1")
            mem.store("user memory 2")
            mem.store("common memory 1", common=True)
            assert mem.count() == 3
            mem.close()
        finally:
            _cleanup(common)
            _cleanup(user)

    def test_delete_tries_user_first(self):
        """Delete checks user DB first, then common."""
        common = _temp_db()
        user = _temp_db()
        try:
            mem = RaggerMemory(uri=common, user_db_path=user)
            uid = mem.store("user memory")
            assert mem.delete(uid)
            assert mem.count() == 0
            mem.close()
        finally:
            _cleanup(common)
            _cleanup(user)

    def test_delete_common_memory(self):
        """Can delete from common DB (if not keep-tagged — but keep prevents it)."""
        common = _temp_db()
        user = _temp_db()
        try:
            mem = RaggerMemory(uri=common, user_db_path=user)
            # Common store auto-adds keep, so delete should be blocked
            cid = mem.store("shared fact", common=True)
            assert not mem.delete(cid)  # keep tag prevents deletion
            assert mem.count() == 1
            mem.close()
        finally:
            _cleanup(common)
            _cleanup(user)

    def test_search_by_metadata_merges(self):
        """search_by_metadata returns from both DBs."""
        common = _temp_db()
        user = _temp_db()
        try:
            mem = RaggerMemory(uri=common, user_db_path=user)
            mem.store("user decision", {"collection": "memory", "category": "decision"})
            mem.store("shared decision", {"collection": "memory", "category": "decision"}, common=True)
            results = mem.search_by_metadata({"category": "decision"})
            assert len(results) == 2
            mem.close()
        finally:
            _cleanup(common)
            _cleanup(user)
