"""
Authentication utilities for Ragger Memory

Token-based auth for API access. Tokens are generated on first run
and stored in ~/.ragger/token with restricted permissions.

Token hashing: tokens are stored as SHA-256 hashes in the DB.
The raw token lives only in the user's token file.
"""

import hashlib
import os
import secrets
import stat


def token_path() -> str:
    """Path to the user's token file"""
    return os.path.expanduser("~/.ragger/token")


def hash_token(token: str) -> str:
    """SHA-256 hash of a token for DB storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def generate_token() -> str:
    """Generate a cryptographically secure token"""
    return secrets.token_urlsafe(32)


def ensure_token() -> str:
    """
    Load existing token or generate a new one.
    Creates ~/.ragger/ and token file if needed.
    Returns the token string.
    """
    path = token_path()
    ragger_dir = os.path.dirname(path)

    # Read existing token
    if os.path.exists(path):
        with open(path, "r") as f:
            token = f.read().strip()
            if token:
                return token

    # Generate new token
    os.makedirs(ragger_dir, exist_ok=True)
    token = generate_token()
    with open(path, "w") as f:
        f.write(token + "\n")

    # Set permissions: owner only (0600)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    return token


def load_token() -> str | None:
    """Load token from file, or None if not found"""
    path = token_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            token = f.read().strip()
            return token if token else None
    return None


def validate_token(provided: str, expected: str) -> bool:
    """Constant-time token comparison"""
    return secrets.compare_digest(provided, expected)
