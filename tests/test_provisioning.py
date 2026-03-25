"""Tests for user provisioning (add-self, add-user, add-all)."""

import os
import stat
import tempfile
import pytest

from ragger_memory.auth import (
    provision_user, hash_token, generate_token
)


class TestProvisionUser:
    """Test provision_user() with a temp directory as fake home."""

    def test_creates_token(self, tmp_path):
        """First call creates ~/.ragger/token."""
        home = str(tmp_path / "fakehome")
        os.makedirs(home)
        token, created = provision_user("testuser", home_dir=home)
        assert created is True
        assert len(token) > 20
        tok_file = os.path.join(home, ".ragger", "token")
        assert os.path.exists(tok_file)
        with open(tok_file) as f:
            assert f.read().strip() == token

    def test_idempotent(self, tmp_path):
        """Second call returns existing token, created=False."""
        home = str(tmp_path / "fakehome")
        os.makedirs(home)
        token1, created1 = provision_user("testuser", home_dir=home)
        token2, created2 = provision_user("testuser", home_dir=home)
        assert created1 is True
        assert created2 is False
        assert token1 == token2

    def test_permissions(self, tmp_path):
        """Token file should be 0640."""
        home = str(tmp_path / "fakehome")
        os.makedirs(home)
        provision_user("testuser", home_dir=home)
        tok_file = os.path.join(home, ".ragger", "token")
        mode = os.stat(tok_file).st_mode & 0o777
        assert mode == 0o640

    def test_creates_ragger_dir(self, tmp_path):
        """Should create .ragger directory."""
        home = str(tmp_path / "fakehome")
        os.makedirs(home)
        provision_user("testuser", home_dir=home)
        assert os.path.isdir(os.path.join(home, ".ragger"))


class TestRegisterUserInDb:
    """Test register_user_in_db with a temp database."""

    def test_register_new_user(self, tmp_path):
        """Should create user in DB."""
        from ragger_memory.embedding import Embedder
        from ragger_memory.sqlite_backend import SqliteBackend
        from ragger_memory.auth import register_user_in_db, hash_token

        db_path = str(tmp_path / "test.db")
        token = generate_token()

        # Direct DB setup
        embedder = Embedder()
        backend = SqliteBackend(embedder, db_path=db_path)

        hashed = hash_token(token)
        user_id = backend.create_user("testuser", hashed, is_admin=False)
        assert user_id > 0

        # Verify
        user = backend.get_user_by_username("testuser")
        assert user is not None
        assert user["username"] == "testuser"
        assert user["token_hash"] == hashed
        backend.close()

    def test_idempotent_register(self, tmp_path):
        """Re-registering same user with same token should not error."""
        from ragger_memory.embedding import Embedder
        from ragger_memory.sqlite_backend import SqliteBackend

        db_path = str(tmp_path / "test.db")
        token = generate_token()

        embedder = Embedder()
        backend = SqliteBackend(embedder, db_path=db_path)

        hashed = hash_token(token)
        id1 = backend.create_user("testuser", hashed)
        user = backend.get_user_by_username("testuser")
        assert user["id"] == id1

        # Same token hash — no error
        user2 = backend.get_user_by_username("testuser")
        assert user2["id"] == id1
        backend.close()

    def test_update_token(self, tmp_path):
        """If token changes, update_user_token should update the hash."""
        from ragger_memory.embedding import Embedder
        from ragger_memory.sqlite_backend import SqliteBackend

        db_path = str(tmp_path / "test.db")

        embedder = Embedder()
        backend = SqliteBackend(embedder, db_path=db_path)

        token1 = generate_token()
        token2 = generate_token()
        hash1 = hash_token(token1)
        hash2 = hash_token(token2)

        backend.create_user("testuser", hash1)
        backend.update_user_token("testuser", hash2)

        user = backend.get_user_by_username("testuser")
        assert user["token_hash"] == hash2
        backend.close()
