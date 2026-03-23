"""Tests for chat context sizing: max_persona_chars and max_memory_results."""

import os
import tempfile
import pytest
from unittest.mock import patch


class TestLoadWorkspaceFiles:
    """Test _load_workspace_files() with max_chars parameter."""

    @pytest.fixture
    def workspace_dirs(self, tmp_path):
        """Create fake user and common dirs with persona files."""
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        common_dir = tmp_path / "common"
        common_dir.mkdir()

        # Write test files with known sizes
        (user_dir / "SOUL.md").write_text("# Soul\n\n" + "Soul content. " * 50)      # ~750 chars
        (user_dir / "USER.md").write_text("# User\n\n" + "User content. " * 50)      # ~750 chars
        (user_dir / "AGENTS.md").write_text("# Agents\n\n" + "Agent info. " * 100)   # ~1200 chars
        (user_dir / "TOOLS.md").write_text("# Tools\n\n" + "Tool info. " * 100)      # ~1200 chars
        (user_dir / "MEMORY.md").write_text("# Memory\n\n" + "Memory data. " * 100)  # ~1300 chars

        return user_dir, common_dir

    def _load_with_dirs(self, user_dir, common_dir, max_chars=0):
        """Call _load_workspace_files with patched paths."""
        with patch("ragger_memory.config.expand_path", return_value=str(user_dir)), \
             patch("ragger_memory.config.system_data_dir", return_value=str(common_dir)):
            from ragger_memory.cli import _load_workspace_files
            return _load_workspace_files(max_chars=max_chars)

    def test_unlimited_loads_all(self, workspace_dirs):
        user_dir, common_dir = workspace_dirs
        result = self._load_with_dirs(user_dir, common_dir, max_chars=0)
        # All 5 files should be present
        assert "# Soul" in result
        assert "# User" in result
        assert "# Agents" in result
        assert "# Tools" in result
        assert "# Memory" in result

    def test_cap_limits_total_size(self, workspace_dirs):
        user_dir, common_dir = workspace_dirs
        result = self._load_with_dirs(user_dir, common_dir, max_chars=2000)
        assert len(result) <= 2100  # small overshoot from truncation marker is OK

    def test_priority_order_soul_first(self, workspace_dirs):
        """SOUL.md should always be included first, even with tight cap."""
        user_dir, common_dir = workspace_dirs
        result = self._load_with_dirs(user_dir, common_dir, max_chars=500)
        assert "# Soul" in result

    def test_priority_order_memory_last(self, workspace_dirs):
        """MEMORY.md is lowest priority — dropped first when capped."""
        user_dir, common_dir = workspace_dirs
        result = self._load_with_dirs(user_dir, common_dir, max_chars=2000)
        # With ~750+750+1200+1200+1300 = ~5200 total, a 2000 cap
        # should include SOUL + USER + partial AGENTS, not MEMORY
        assert "# Memory" not in result

    def test_truncation_marker_added(self, workspace_dirs):
        """When a file is truncated, a marker should be appended."""
        user_dir, common_dir = workspace_dirs
        result = self._load_with_dirs(user_dir, common_dir, max_chars=1000)
        assert "truncated" in result.lower()

    def test_very_small_cap(self, workspace_dirs):
        """Even with tiny cap, should return something useful."""
        user_dir, common_dir = workspace_dirs
        result = self._load_with_dirs(user_dir, common_dir, max_chars=100)
        assert len(result) > 0
        assert "# Soul" in result

    def test_common_dir_takes_precedence(self, workspace_dirs):
        """Common dir files should override user dir for shared files."""
        user_dir, common_dir = workspace_dirs
        (common_dir / "SOUL.md").write_text("# Common Soul\n\nShared personality.")
        result = self._load_with_dirs(user_dir, common_dir, max_chars=0)
        assert "Common Soul" in result
        # Should NOT have the user version's content
        assert "Soul content" not in result

    def test_user_only_files_not_in_common(self, workspace_dirs):
        """USER.md and MEMORY.md should only come from user dir."""
        user_dir, common_dir = workspace_dirs
        # Even if USER.md exists in common, it should be ignored
        (common_dir / "USER.md").write_text("# Wrong User")
        result = self._load_with_dirs(user_dir, common_dir, max_chars=0)
        # USER.md search_common is False, so common dir is never checked
        assert "User content" in result

    def test_missing_files_skipped(self, tmp_path):
        """Missing files should be silently skipped."""
        user_dir = tmp_path / "empty_user"
        user_dir.mkdir()
        common_dir = tmp_path / "empty_common"
        common_dir.mkdir()
        (user_dir / "SOUL.md").write_text("# Just Soul")
        result = self._load_with_dirs(user_dir, common_dir, max_chars=0)
        assert "Just Soul" in result
        assert "---" not in result  # only one section, no separators

    def test_empty_files_skipped(self, workspace_dirs):
        """Empty files should not contribute to output."""
        user_dir, common_dir = workspace_dirs
        (user_dir / "SOUL.md").write_text("")
        (user_dir / "AGENTS.md").write_text("   ")  # whitespace only
        result = self._load_with_dirs(user_dir, common_dir, max_chars=0)
        assert "# Soul" not in result
        assert "# Agents" not in result
        assert "# User" in result  # non-empty files still included


class TestChatContextConfig:
    """Test that config keys load correctly."""

    def test_defaults(self):
        from ragger_memory.config import get_config
        cfg = get_config()
        assert cfg["chat_max_persona_chars"] == 0
        assert cfg["chat_max_memory_results"] == 3

    def test_keys_are_integers(self):
        """Config values should be integers, not strings."""
        from ragger_memory.config import get_config
        cfg = get_config()
        assert isinstance(cfg["chat_max_persona_chars"], int)
        assert isinstance(cfg["chat_max_memory_results"], int)
