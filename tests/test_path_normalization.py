"""
Tests for home path normalization.
"""
import os

import pytest

from ragger_memory.backend import MemoryBackend, _HOME_PREFIX


class TestPathNormalization:
    """Tests for _normalize_paths and _normalize_metadata."""
    
    def test_normalize_home_in_text(self):
        home = os.path.expanduser("~")
        text = f"{home}/Documents/test.md"
        result = MemoryBackend._normalize_paths(text)
        assert result == "~/Documents/test.md"
    
    def test_normalize_preserves_non_home_paths(self):
        text = "/usr/local/bin/python"
        result = MemoryBackend._normalize_paths(text)
        assert result == "/usr/local/bin/python"
    
    def test_normalize_metadata_string_values(self):
        home = os.path.expanduser("~")
        meta = {
            "source": f"{home}/Projects/notes.md",
            "category": "test",
            "count": 42,
        }
        result = MemoryBackend._normalize_metadata(meta)
        assert result["source"] == "~/Projects/notes.md"
        assert result["category"] == "test"
        assert result["count"] == 42
    
    def test_normalize_metadata_list_values(self):
        home = os.path.expanduser("~")
        meta = {
            "sources": [f"{home}/a.md", f"{home}/b.md", "relative.md"],
        }
        result = MemoryBackend._normalize_metadata(meta)
        assert result["sources"] == ["~/a.md", "~/b.md", "relative.md"]
    
    def test_normalize_empty_metadata(self):
        result = MemoryBackend._normalize_metadata({})
        assert result == {}
    
    def test_home_prefix_ends_with_slash(self):
        """_HOME_PREFIX should end with / for clean replacement."""
        assert _HOME_PREFIX.endswith("/")
