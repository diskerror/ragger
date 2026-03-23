"""
Tests for "keep" tag functionality in memory deletion
"""
import pytest
import tempfile
import os
from ragger_memory.memory import RaggerMemory


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
    # Also clean up WAL files
    for suffix in ['-wal', '-shm']:
        wal_path = db_path + suffix
        if os.path.exists(wal_path):
            os.unlink(wal_path)


def test_delete_respects_keep_tag(temp_db):
    """Test that delete() respects the keep tag"""
    memory = RaggerMemory(uri=temp_db)
    
    # Store a memory with keep=true
    memory_id = memory.store("Important data that should not be deleted", {
        "keep": True,
        "category": "important"
    })
    
    # Try to delete it
    deleted = memory.delete(memory_id)
    
    # Should return False (not deleted)
    assert deleted is False
    
    # Memory should still exist
    results = memory.search_by_metadata({"category": "important"})
    assert len(results) == 1
    assert results[0]["id"] == memory_id
    
    memory.close()


def test_delete_without_keep_tag(temp_db):
    """Test that delete() works normally without keep tag"""
    memory = RaggerMemory(uri=temp_db)
    
    # Store a memory without keep tag
    memory_id = memory.store("Temporary data", {
        "category": "temporary"
    })
    
    # Delete it
    deleted = memory.delete(memory_id)
    
    # Should return True (deleted)
    assert deleted is True
    
    # Memory should not exist
    results = memory.search_by_metadata({"category": "temporary"})
    assert len(results) == 0
    
    memory.close()


def test_delete_batch_respects_keep_tag(temp_db):
    """Test that delete_batch() respects keep tags"""
    memory = RaggerMemory(uri=temp_db)
    
    # Store multiple memories, some with keep tag
    kept_id1 = memory.store("Keep this one", {
        "keep": True,
        "category": "test"
    })
    
    deletable_id1 = memory.store("Delete this one", {
        "category": "test"
    })
    
    kept_id2 = memory.store("Keep this too", {
        "keep": True,
        "category": "test"
    })
    
    deletable_id2 = memory.store("Delete this too", {
        "category": "test"
    })
    
    # Try to delete all of them
    all_ids = [kept_id1, deletable_id1, kept_id2, deletable_id2]
    deleted_count = memory.delete_batch(all_ids)
    
    # Should only delete the 2 without keep tag
    assert deleted_count == 2
    
    # Verify what's left
    results = memory.search_by_metadata({"category": "test"})
    assert len(results) == 2
    
    # Check that the kept ones are still there
    result_ids = {r["id"] for r in results}
    assert kept_id1 in result_ids
    assert kept_id2 in result_ids
    assert deletable_id1 not in result_ids
    assert deletable_id2 not in result_ids
    
    memory.close()


def test_store_common_sets_keep_tag(temp_db):
    """Test that storing to common DB automatically sets keep tag"""
    memory = RaggerMemory(uri=temp_db)
    
    # Store with common=True (even though we don't have a separate common DB in this test)
    memory_id = memory.store("Common data", {
        "category": "shared"
    }, common=True)
    
    # Verify the keep tag was automatically added
    results = memory.search_by_metadata({"category": "shared"})
    assert len(results) == 1
    assert results[0]["metadata"].get("keep") is True
    
    # Try to delete it - should fail
    deleted = memory.delete(memory_id)
    assert deleted is False
    
    # Verify it still exists
    results = memory.search_by_metadata({"category": "shared"})
    assert len(results) == 1
    
    memory.close()


def test_keep_tag_with_existing_metadata(temp_db):
    """Test that keep tag is added to existing metadata when common=True"""
    memory = RaggerMemory(uri=temp_db)
    
    # Store with existing metadata + common=True
    memory_id = memory.store("Data with metadata", {
        "category": "test",
        "source": "test-suite",
        "custom_field": "value"
    }, common=True)
    
    # Verify all metadata is preserved and keep is added
    results = memory.search_by_metadata({"category": "test"})
    assert len(results) == 1
    metadata = results[0]["metadata"]
    assert metadata.get("keep") is True
    assert metadata.get("category") == "test"
    assert metadata.get("source") == "test-suite"
    assert metadata.get("custom_field") == "value"
    
    memory.close()


def test_delete_batch_empty_list(temp_db):
    """Test that delete_batch handles empty list gracefully"""
    memory = RaggerMemory(uri=temp_db)
    
    deleted_count = memory.delete_batch([])
    assert deleted_count == 0
    
    memory.close()


def test_delete_nonexistent_memory(temp_db):
    """Test that deleting non-existent memory returns False"""
    memory = RaggerMemory(uri=temp_db)
    
    deleted = memory.delete("99999")
    assert deleted is False
    
    memory.close()
