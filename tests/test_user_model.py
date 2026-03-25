"""
Tests for per-user model selection feature.

Users can set their preferred inference model via API.
Server stores it in the users table and uses it for inference requests.
"""

import pytest

from ragger_memory.auth import generate_token, hash_token
from ragger_memory.sqlite_backend import SqliteBackend


class TestUserModel:
    """Test per-user model selection functionality"""

    def test_migration_adds_preferred_model_column(self, mock_embedder, tmp_path):
        """preferred_model column is added by migration"""
        db_path = str(tmp_path / "test.db")
        backend = SqliteBackend(mock_embedder, db_path)
        
        # Check that column exists
        cols = [row[1] for row in backend.conn.execute(
            "PRAGMA table_info(users)"
        ).fetchall()]
        assert "preferred_model" in cols
        
        backend.close()

    def test_set_preferred_model(self, mock_embedder, tmp_path):
        """update_user_preferred_model sets the model for a user"""
        db_path = str(tmp_path / "test.db")
        backend = SqliteBackend(mock_embedder, db_path)
        
        # Create user
        token = generate_token()
        backend.create_user("testuser", hash_token(token))
        
        # Set preferred model
        backend.update_user_preferred_model("testuser", "claude-sonnet-4-5")
        
        # Verify it's set
        model = backend.get_user_preferred_model("testuser")
        assert model == "claude-sonnet-4-5"
        
        backend.close()

    def test_get_preferred_model_returns_none_when_not_set(self, mock_embedder, tmp_path):
        """get_user_preferred_model returns None when not set"""
        db_path = str(tmp_path / "test.db")
        backend = SqliteBackend(mock_embedder, db_path)
        
        # Create user without setting model
        token = generate_token()
        backend.create_user("testuser", hash_token(token))
        
        # Should return None
        model = backend.get_user_preferred_model("testuser")
        assert model is None
        
        backend.close()

    def test_clear_preferred_model(self, mock_embedder, tmp_path):
        """Setting preferred_model to None clears it"""
        db_path = str(tmp_path / "test.db")
        backend = SqliteBackend(mock_embedder, db_path)
        
        # Create user and set model
        token = generate_token()
        backend.create_user("testuser", hash_token(token))
        backend.update_user_preferred_model("testuser", "claude-sonnet-4-5")
        
        # Clear it
        backend.update_user_preferred_model("testuser", None)
        
        # Should be None
        model = backend.get_user_preferred_model("testuser")
        assert model is None
        
        backend.close()

    def test_different_users_different_models(self, mock_embedder, tmp_path):
        """Different users can have different preferred models"""
        db_path = str(tmp_path / "test.db")
        backend = SqliteBackend(mock_embedder, db_path)
        
        # Create two users
        backend.create_user("alice", hash_token(generate_token()))
        backend.create_user("bob", hash_token(generate_token()))
        
        # Set different models
        backend.update_user_preferred_model("alice", "claude-sonnet-4-5")
        backend.update_user_preferred_model("bob", "gpt-4")
        
        # Verify each user has their own model
        assert backend.get_user_preferred_model("alice") == "claude-sonnet-4-5"
        assert backend.get_user_preferred_model("bob") == "gpt-4"
        
        backend.close()

    def test_preferred_model_survives_token_update(self, mock_embedder, tmp_path):
        """Preferred model persists when token is updated"""
        db_path = str(tmp_path / "test.db")
        backend = SqliteBackend(mock_embedder, db_path)
        
        # Create user and set model
        old_token = generate_token()
        backend.create_user("testuser", hash_token(old_token))
        backend.update_user_preferred_model("testuser", "claude-sonnet-4-5")
        
        # Update token (simulate rotation)
        new_token = generate_token()
        backend.update_user_token("testuser", hash_token(new_token))
        
        # Model should still be set
        model = backend.get_user_preferred_model("testuser")
        assert model == "claude-sonnet-4-5"
        
        backend.close()

    def test_model_field_accepts_various_model_names(self, mock_embedder, tmp_path):
        """preferred_model field accepts various model name formats"""
        db_path = str(tmp_path / "test.db")
        backend = SqliteBackend(mock_embedder, db_path)
        
        backend.create_user("testuser", hash_token(generate_token()))
        
        # Test various model name formats
        test_cases = [
            "claude-sonnet-4-5",
            "gpt-4-turbo",
            "llama-3.1-70b",
            "qwen/qwen-2.5-72b",
            "anthropic/claude-3-opus-20240229",
            "openai/gpt-4o-2024-11-20",
        ]
        
        for model_name in test_cases:
            backend.update_user_preferred_model("testuser", model_name)
            retrieved = backend.get_user_preferred_model("testuser")
            assert retrieved == model_name, f"Failed for {model_name}"
        
        backend.close()

    def test_null_preferred_model_falls_back_to_system_default(self, mock_embedder, tmp_path):
        """NULL preferred_model means use system default (tested at server level)"""
        # This is more of an integration test — the server should:
        # 1. Check user's preferred_model
        # 2. If NULL, use system default from config
        # 3. If set, use user's preference
        
        db_path = str(tmp_path / "test.db")
        backend = SqliteBackend(mock_embedder, db_path)
        
        backend.create_user("testuser", hash_token(generate_token()))
        
        # Default should be None (NULL in DB)
        model = backend.get_user_preferred_model("testuser")
        assert model is None
        
        # Server logic should fall back to system default
        # (we test this in integration tests)
        
        backend.close()
