"""
Tests for file import and chunking logic.

Tests paragraph-aware chunking, heading preservation, and metadata.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragger_memory.cli import import_file


@pytest.fixture
def mock_memory():
    """Create a mock RaggerMemory for testing import."""
    mem = MagicMock()
    mem.store.return_value = "test-id-123"
    return mem


class TestImportFileBasics:
    """Basic import functionality tests."""
    
    def test_import_nonexistent_file_raises(self, mock_memory, tmp_path):
        """Importing non-existent file should raise FileNotFoundError."""
        missing = tmp_path / "nonexistent.md"
        with pytest.raises(FileNotFoundError):
            import_file(mock_memory, str(missing))
    
    def test_import_empty_file(self, mock_memory, tmp_path):
        """Empty file should not create any chunks."""
        empty = tmp_path / "empty.md"
        empty.write_text("")
        
        import_file(mock_memory, str(empty))
        
        # Should not store anything
        assert mock_memory.store.call_count == 0
    
    def test_import_single_paragraph(self, mock_memory, tmp_path):
        """Single paragraph should create one chunk."""
        doc = tmp_path / "single.md"
        doc.write_text("This is a single paragraph of text.")
        
        import_file(mock_memory, str(doc))
        
        assert mock_memory.store.call_count == 1
        call_args = mock_memory.store.call_args
        text = call_args[0][0]
        metadata = call_args[0][1]
        
        assert "single paragraph" in text
        assert metadata["source"] == str(doc)
        assert metadata["chunk"] == 1
        assert metadata["total_chunks"] == 1


class TestChunkingLogic:
    """Tests for paragraph chunking with minimum size."""
    
    def test_chunks_respect_minimum_size(self, mock_memory, tmp_path):
        """Short paragraphs should be merged until min_chunk_size."""
        # Create several short paragraphs (each under 300 chars)
        short_paras = [
            "First short paragraph.",
            "Second short paragraph.",
            "Third short paragraph.",
            "Fourth short paragraph.",
        ]
        doc = tmp_path / "short.md"
        doc.write_text("\n\n".join(short_paras))
        
        import_file(mock_memory, str(doc), min_chunk_size=300)
        
        # All should be merged into one chunk since total < 300 chars
        assert mock_memory.store.call_count == 1
        call_args = mock_memory.store.call_args
        text = call_args[0][0]
        
        # Should contain all paragraphs
        for para in short_paras:
            assert para in text
    
    def test_chunks_split_when_exceeding_minimum(self, mock_memory, tmp_path):
        """Paragraphs exceeding min size should split into separate chunks."""
        # Create long paragraphs (each > 300 chars)
        long_para = "A" * 350
        doc = tmp_path / "long.md"
        doc.write_text(f"{long_para}\n\n{long_para}\n\n{long_para}")
        
        import_file(mock_memory, str(doc), min_chunk_size=300)
        
        # Should create 3 chunks (each exceeds min_chunk_size)
        assert mock_memory.store.call_count == 3
    
    def test_custom_minimum_chunk_size(self, mock_memory, tmp_path):
        """Custom min_chunk_size should be respected."""
        paras = ["Short para 1.", "Short para 2.", "Short para 3."]
        doc = tmp_path / "custom.md"
        doc.write_text("\n\n".join(paras))
        
        # With very low min size, should create separate chunks
        import_file(mock_memory, str(doc), min_chunk_size=10)
        
        # Each paragraph should be its own chunk
        assert mock_memory.store.call_count == 3


class TestHeadingPreservation:
    """Tests for markdown heading preservation in chunks."""
    
    def test_headings_included_in_chunks(self, mock_memory, tmp_path):
        """Chunks should include their section headings."""
        markdown = """\
# Main Title

Introduction paragraph.

## Section One

Content for section one.

## Section Two

Content for section two.
"""
        doc = tmp_path / "headings.md"
        doc.write_text(markdown)
        
        import_file(mock_memory, str(doc), min_chunk_size=50)
        
        # Get all stored chunks
        chunks = [call[0][0] for call in mock_memory.store.call_args_list]
        
        # Headings should be preserved in chunks
        assert any("# Main Title" in chunk for chunk in chunks)
        assert any("## Section One" in chunk for chunk in chunks)
        assert any("## Section Two" in chunk for chunk in chunks)
    
    def test_section_metadata_attached(self, mock_memory, tmp_path):
        """Chunks should have section metadata."""
        markdown = """\
# Document Title

## Introduction

Intro text here.

## Methods

Methods text here.
"""
        doc = tmp_path / "sections.md"
        doc.write_text(markdown)
        
        import_file(mock_memory, str(doc), min_chunk_size=20)
        
        # Check section metadata
        sections = []
        for call in mock_memory.store.call_args_list:
            metadata = call[0][1]
            if "section" in metadata:
                sections.append(metadata["section"])
        
        # Should have section hierarchy
        assert any("Introduction" in s for s in sections)
        assert any("Methods" in s for s in sections)
    
    def test_nested_headings(self, mock_memory, tmp_path):
        """Nested headings should create proper section hierarchy."""
        markdown = """\
# Chapter 1

## Section 1.1

### Subsection 1.1.1

Content here.

### Subsection 1.1.2

More content.

## Section 1.2

Different section.
"""
        doc = tmp_path / "nested.md"
        doc.write_text(markdown)
        
        import_file(mock_memory, str(doc), min_chunk_size=30)
        
        # Get section metadata
        sections = []
        for call in mock_memory.store.call_args_list:
            metadata = call[0][1]
            if "section" in metadata:
                sections.append(metadata["section"])
        
        # Should have hierarchical sections with » separator
        assert any("Section 1.1" in s and "Subsection 1.1.1" in s for s in sections)


class TestDataCleaning:
    """Tests for data cleaning during import."""
    
    def test_strips_embedded_base64_images(self, mock_memory, tmp_path):
        """Base64 image data should be removed before chunking."""
        markdown = """\
Here is some text.

![image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==)

More text after image.
"""
        doc = tmp_path / "images.md"
        doc.write_text(markdown)
        
        import_file(mock_memory, str(doc), min_chunk_size=10)
        
        # Get all stored chunks
        all_text = " ".join(call[0][0] for call in mock_memory.store.call_args_list)
        
        # Base64 data should be stripped from all chunks
        assert "data:image" not in all_text
        assert "base64" not in all_text
        # Both text segments should be present (in whatever chunks)
        assert "Here is some text" in all_text
        assert "More text after image" in all_text
    
    def test_collapses_multiple_spaces(self, mock_memory, tmp_path):
        """Multiple spaces (OCR artifacts) should collapse to single space."""
        text_with_spaces = "This  has    multiple     spaces      between       words."
        doc = tmp_path / "spaces.md"
        doc.write_text(text_with_spaces)
        
        import_file(mock_memory, str(doc), min_chunk_size=10)
        
        stored_text = mock_memory.store.call_args[0][0]
        
        # Should have single spaces only
        assert "  " not in stored_text
        assert "This has multiple spaces between words." in stored_text
    
    def test_collapses_excessive_newlines(self, mock_memory, tmp_path):
        """More than 2 consecutive newlines should collapse to 2."""
        text = "Para 1\n\n\n\n\nPara 2\n\n\n\n\n\n\nPara 3"
        doc = tmp_path / "newlines.md"
        doc.write_text(text)
        
        import_file(mock_memory, str(doc), min_chunk_size=10)
        
        stored_text = mock_memory.store.call_args[0][0]
        
        # Should have at most 2 consecutive newlines (paragraph breaks)
        assert "\n\n\n" not in stored_text


class TestCustomMetadata:
    """Tests for custom metadata attachment."""
    
    def test_custom_metadata_merged(self, mock_memory, tmp_path):
        """Custom metadata should be merged with auto metadata."""
        doc = tmp_path / "custom.md"
        doc.write_text("Test content.")
        
        custom_meta = {"collection": "docs", "author": "test_user"}
        import_file(mock_memory, str(doc), metadata=custom_meta)
        
        stored_meta = mock_memory.store.call_args[0][1]
        
        # Should have both custom and auto metadata
        assert stored_meta["collection"] == "docs"
        assert stored_meta["author"] == "test_user"
        assert "source" in stored_meta
        assert "chunk" in stored_meta
    
    def test_source_metadata_always_present(self, mock_memory, tmp_path):
        """Source file path should always be in metadata."""
        doc = tmp_path / "source.md"
        doc.write_text("Content here.")
        
        import_file(mock_memory, str(doc))
        
        stored_meta = mock_memory.store.call_args[0][1]
        assert stored_meta["source"] == str(doc)


class TestChunkMetadata:
    """Tests for chunk numbering metadata."""
    
    def test_chunk_numbering_sequential(self, mock_memory, tmp_path):
        """Chunks should be numbered sequentially starting from 1."""
        long_text = ("Paragraph. " * 50 + "\n\n") * 5  # 5 long paragraphs
        doc = tmp_path / "numbered.md"
        doc.write_text(long_text)
        
        import_file(mock_memory, str(doc), min_chunk_size=200)
        
        # Extract chunk numbers
        chunk_numbers = [call[0][1]["chunk"] for call in mock_memory.store.call_args_list]
        
        # Should be 1, 2, 3, ...
        assert chunk_numbers == list(range(1, len(chunk_numbers) + 1))
    
    def test_total_chunks_consistent(self, mock_memory, tmp_path):
        """All chunks should have same total_chunks value."""
        long_text = ("Para. " * 40 + "\n\n") * 4
        doc = tmp_path / "total.md"
        doc.write_text(long_text)
        
        import_file(mock_memory, str(doc), min_chunk_size=150)
        
        total_chunks = [call[0][1]["total_chunks"] for call in mock_memory.store.call_args_list]
        
        # All should have same total
        assert len(set(total_chunks)) == 1
        assert total_chunks[0] == len(total_chunks)
