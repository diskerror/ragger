"""
Tests for background summarization and session cleanup.
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from ragger_memory.chat_sessions import (
    ChatSession, get_or_create_session, cleanup_expired_sessions,
    run_housekeeping, _sessions, _sessions_lock
)


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear session state between tests."""
    with _sessions_lock:
        _sessions.clear()
    yield
    with _sessions_lock:
        _sessions.clear()


class TestSessionExpiry:
    def test_idle_session_expires(self):
        """Session idle beyond pause_minutes gets removed."""
        session = get_or_create_session("test-1", "reid")
        session.add_user_message("hello")
        session.add_assistant_message("hi there")
        # Force idle time
        session.last_activity = time.time() - 700  # >10 min

        with patch("ragger_memory.chat_sessions.get_config") as mock_cfg:
            mock_cfg.return_value = {"chat_pause_minutes": 10}
            cleanup_expired_sessions()

        with _sessions_lock:
            assert "test-1" not in _sessions

    def test_active_session_kept(self):
        """Session within pause_minutes stays."""
        session = get_or_create_session("test-2", "reid")
        session.add_user_message("hello")
        session.last_activity = time.time()  # just now

        with patch("ragger_memory.chat_sessions.get_config") as mock_cfg:
            mock_cfg.return_value = {"chat_pause_minutes": 10}
            cleanup_expired_sessions()

        with _sessions_lock:
            assert "test-2" in _sessions


class TestSummarization:
    def test_expired_session_triggers_summarize(self):
        """Expired session with unsummarized turns triggers background summarize."""
        session = get_or_create_session("sum-1", "reid")
        session.add_user_message("What is Python?")
        session.add_assistant_message("A programming language.")
        session.last_activity = time.time() - 700

        mock_memory = MagicMock()
        mock_inference = MagicMock()
        mock_inference.chat.return_value = {"choices": [{"message": {"content": "Summary"}}]}
        mock_inference.extract_content.return_value = "Summary of conversation"

        with patch("ragger_memory.chat_sessions.get_config") as mock_cfg:
            mock_cfg.return_value = {"chat_pause_minutes": 10}
            cleanup_expired_sessions(
                memory=mock_memory,
                inference_client=mock_inference
            )

        # Give background thread time to run
        time.sleep(0.5)

        # Inference should have been called
        mock_inference.chat.assert_called_once()
        # Memory should have stored the summary
        mock_memory.store.assert_called_once()
        # store(text, metadata_dict) — check the call happened with summary content
        call_args = mock_memory.store.call_args
        # Could be positional or keyword — just verify it was called with something
        assert call_args is not None
        all_args = str(call_args)
        assert "session-summary" in all_args

    def test_no_summarize_without_inference_client(self):
        """Without inference client, expired sessions just get removed."""
        session = get_or_create_session("sum-2", "reid")
        session.add_user_message("test")
        session.add_assistant_message("response")
        session.last_activity = time.time() - 700

        with patch("ragger_memory.chat_sessions.get_config") as mock_cfg:
            mock_cfg.return_value = {"chat_pause_minutes": 10}
            cleanup_expired_sessions(memory=None, inference_client=None)

        with _sessions_lock:
            assert "sum-2" not in _sessions


class TestRunHousekeeping:
    def test_housekeeping_cleans_old_conversations(self, tmp_path):
        """Housekeeping deletes conversation entries older than max_age."""
        import sqlite3

        db_path = str(tmp_path / "test_user.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY,
                text TEXT, embedding BLOB, metadata TEXT,
                timestamp TEXT, collection TEXT, category TEXT, tags TEXT
            )
        """)
        # Old conversation entry
        conn.execute(
            "INSERT INTO memories (text, timestamp, collection, category) VALUES (?, ?, ?, ?)",
            ("old turn", "2020-01-01T00:00:00Z", "conversation", "chat-turn")
        )
        # Recent conversation entry
        conn.execute(
            "INSERT INTO memories (text, timestamp, collection, category) VALUES (?, ?, ?, ?)",
            ("new turn", "2099-01-01T00:00:00Z", "conversation", "chat-turn")
        )
        # Non-conversation entry (should not be deleted)
        conn.execute(
            "INSERT INTO memories (text, timestamp, collection, category) VALUES (?, ?, ?, ?)",
            ("important fact", "2020-01-01T00:00:00Z", "memory", "fact")
        )
        conn.commit()
        conn.close()

        with patch("ragger_memory.chat_sessions.get_config") as mock_cfg:
            mock_cfg.return_value = {
                "chat_pause_minutes": 10,
                "cleanup_max_age_hours": 336  # 2 weeks
            }
            results = run_housekeeping(user_db_paths=[db_path])

        assert results["conversations_cleaned"] == 1

        # Verify correct rows remain
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT text, collection FROM memories").fetchall()
        conn.close()
        texts = [r[0] for r in rows]
        assert "old turn" not in texts
        assert "new turn" in texts
        assert "important fact" in texts
