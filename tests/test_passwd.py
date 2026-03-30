"""
Tests for password hashing, verification, and the passwd workflow.
"""
import pytest

from ragger_memory.auth import hash_password, verify_password


class TestPasswordHashing:
    """Test PBKDF2 password hashing and verification."""

    def test_hash_format(self):
        """Hash should have pbkdf2:iterations:salt_hex:key_hex format."""
        h = hash_password("testpass")
        parts = h.split(":")
        assert len(parts) == 4
        assert parts[0] == "pbkdf2"
        assert int(parts[1]) > 0  # iterations
        assert len(parts[2]) > 0  # salt hex
        assert len(parts[3]) > 0  # key hex

    def test_verify_correct_password(self):
        h = hash_password("mypassword")
        assert verify_password("mypassword", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("mypassword")
        assert verify_password("wrongpassword", h) is False

    def test_verify_empty_password(self):
        h = hash_password("mypassword")
        assert verify_password("", h) is False

    def test_different_hashes_for_same_password(self):
        """Salt should make each hash unique."""
        h1 = hash_password("samepass")
        h2 = hash_password("samepass")
        assert h1 != h2  # different salts
        # But both should verify
        assert verify_password("samepass", h1) is True
        assert verify_password("samepass", h2) is True

    def test_verify_invalid_format(self):
        """Non-pbkdf2 prefix should fail."""
        assert verify_password("test", "invalid:hash") is False
        assert verify_password("test", "") is False
        assert verify_password("test", "bcrypt:foo:bar:baz") is False

    def test_verify_malformed_pbkdf2(self):
        """Too few parts should fail."""
        assert verify_password("test", "pbkdf2:600000:abc") is False


class TestPasswdBackend:
    """Test set/get password via SqliteBackend."""

    def test_set_and_get_password(self, sqlite_backend):
        """Store and retrieve password hash for a user."""
        from ragger_memory.auth import hash_token
        # Create a user first
        token_hash = hash_token("testtoken")
        sqlite_backend.create_user("testuser", token_hash)
        
        # Set password
        pw_hash = hash_password("secret123")
        sqlite_backend.set_user_password("testuser", pw_hash)
        
        # Get and verify
        stored = sqlite_backend.get_user_password("testuser")
        assert stored is not None
        assert verify_password("secret123", stored) is True

    def test_password_initially_none(self, sqlite_backend):
        """New user should have no password."""
        from ragger_memory.auth import hash_token
        token_hash = hash_token("testtoken2")
        sqlite_backend.create_user("newuser", token_hash)
        
        stored = sqlite_backend.get_user_password("newuser")
        assert stored is None

    def test_change_password(self, sqlite_backend):
        """Changing password should invalidate old one."""
        from ragger_memory.auth import hash_token
        token_hash = hash_token("testtoken3")
        sqlite_backend.create_user("changeuser", token_hash)
        
        # Set initial password
        sqlite_backend.set_user_password("changeuser", hash_password("oldpass"))
        assert verify_password("oldpass",
                               sqlite_backend.get_user_password("changeuser"))
        
        # Change it
        sqlite_backend.set_user_password("changeuser", hash_password("newpass"))
        stored = sqlite_backend.get_user_password("changeuser")
        assert verify_password("newpass", stored) is True
        assert verify_password("oldpass", stored) is False
