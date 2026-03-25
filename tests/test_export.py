"""
Tests for the export module.
"""
import json
import sqlite3

import pytest

from ragger_memory.export import export_docs, export_memories, _split_heading_body
from ragger_memory.config import SQLITE_MEMORIES_TABLE


@pytest.fixture
def export_db(tmp_path):
    """Create a temporary SQLite DB with the memories table for export tests."""
    db_path = str(tmp_path / "export_test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(f"""
        CREATE TABLE {SQLITE_MEMORIES_TABLE} (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            metadata TEXT,
            collection TEXT DEFAULT 'memory',
            category TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            timestamp TEXT DEFAULT (datetime('now')),
            embedding BLOB
        )
    """)
    conn.commit()
    
    def _store(text, metadata=None, timestamp=None):
        meta = metadata or {}
        ts = timestamp or "2026-01-15T10:00:00"
        conn.execute(
            f"INSERT INTO {SQLITE_MEMORIES_TABLE} (text, metadata, collection, timestamp) "
            f"VALUES (?, ?, ?, ?)",
            (text, json.dumps(meta), meta.get('collection', 'memory'), ts)
        )
        conn.commit()
    
    yield db_path, _store
    conn.close()


class TestSplitHeadingBody:
    def test_no_headings(self):
        headings, body = _split_heading_body("just some text")
        assert headings == []
        assert body == "just some text"

    def test_heading_and_body(self):
        text = "# Title\n\nBody text here"
        headings, body = _split_heading_body(text)
        assert headings == ["# Title"]
        assert body == "Body text here"

    def test_multiple_headings(self):
        text = "# Doc\n\n## Section\n\nContent"
        headings, body = _split_heading_body(text)
        assert headings == ["# Doc", "## Section"]
        assert body == "Content"


class TestExportDocs:
    def test_export_basic(self, export_db, tmp_path):
        db_path, store = export_db
        dest = str(tmp_path / "out")
        store("# Doc\n\nFirst chunk content", {
            'collection': 'docs', 'filename': 'test.md', 'chunk': 0
        })
        store("# Doc\n\nSecond chunk content", {
            'collection': 'docs', 'filename': 'test.md', 'chunk': 1
        })
        
        export_docs('docs', dest, db_path=db_path)
        
        output = (tmp_path / "out" / "test.md").read_text()
        assert "First chunk content" in output
        assert "Second chunk content" in output

    def test_export_heading_deduplication(self, export_db, tmp_path):
        db_path, store = export_db
        dest = str(tmp_path / "out")
        store("# Doc\n\n## Section\n\nChunk 1 text", {
            'collection': 'docs', 'filename': 'test.md', 'chunk': 0
        })
        store("# Doc\n\n## Section\n\nChunk 2 text", {
            'collection': 'docs', 'filename': 'test.md', 'chunk': 1
        })
        
        export_docs('docs', dest, db_path=db_path)
        
        output = (tmp_path / "out" / "test.md").read_text()
        # Heading "# Doc" should appear only once
        assert output.count("# Doc") == 1
        assert output.count("## Section") == 1
        assert "Chunk 1 text" in output
        assert "Chunk 2 text" in output

    def test_export_by_collection(self, export_db, tmp_path):
        db_path, store = export_db
        dest = str(tmp_path / "out")
        store("Docs content", {
            'collection': 'docs', 'filename': 'a.md', 'chunk': 0
        })
        store("Notes content", {
            'collection': 'notes', 'filename': 'b.md', 'chunk': 0
        })
        
        export_docs('docs', dest, db_path=db_path)
        
        files = list((tmp_path / "out").iterdir())
        filenames = [f.name for f in files]
        assert "a.md" in filenames
        assert "b.md" not in filenames

    def test_export_empty_collection(self, export_db, tmp_path):
        db_path, _ = export_db
        dest = str(tmp_path / "out")
        # Should not crash
        export_docs('nonexistent', dest, db_path=db_path)
        # Dest dir created but empty
        assert (tmp_path / "out").exists()


class TestExportMemories:
    def test_export_basic(self, export_db, tmp_path):
        db_path, store = export_db
        dest = str(tmp_path / "out")
        store("A remembered fact", {'collection': 'memory', 'category': 'fact'},
              timestamp="2026-01-15T10:00:00")
        
        export_memories(dest, db_path=db_path)
        
        output = (tmp_path / "out" / "2026-01-15.md").read_text()
        assert "A remembered fact" in output

    def test_export_empty(self, export_db, tmp_path):
        db_path, _ = export_db
        dest = str(tmp_path / "out")
        # No memories stored — should not crash
        export_memories(dest, db_path=db_path)
