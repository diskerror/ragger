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

    # Set permissions: owner read/write + group read (0640)
    # Group read allows the daemon (_ragger) to read user tokens
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)

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


def token_path_for_user(username: str) -> str:
    """Token file path for a given username, using their home directory."""
    import pwd
    pw = pwd.getpwnam(username)
    return os.path.join(pw.pw_dir, ".ragger", "token")


def provision_user(username: str, home_dir: str | None = None) -> tuple[str, bool]:
    """
    Provision a user: create ~/.ragger/, generate token, set permissions.

    Args:
        username: System username
        home_dir: Override home directory (default: look up from passwd)

    Returns:
        (token, created) — token string and whether it was newly created.
        If user already had a token, returns (existing_token, False).
    """
    if home_dir is None:
        import pwd
        pw = pwd.getpwnam(username)
        home_dir = pw.pw_dir

    ragger_dir = os.path.join(home_dir, ".ragger")
    path = os.path.join(ragger_dir, "token")

    # Check existing
    if os.path.exists(path):
        with open(path, "r") as f:
            token = f.read().strip()
            if token:
                return token, False

    # Create directory and token
    os.makedirs(ragger_dir, exist_ok=True)
    token = generate_token()
    with open(path, "w") as f:
        f.write(token + "\n")

    # Permissions: 0640 (owner rw + group read for daemon)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)

    # Set ownership to the target user if we're running as root
    if os.getuid() == 0:
        import pwd
        try:
            pw = pwd.getpwnam(username)
            os.chown(ragger_dir, pw.pw_uid, pw.pw_gid)
            os.chown(path, pw.pw_uid, pw.pw_gid)
        except KeyError:
            pass  # user doesn't exist in passwd — skip chown

    return token, True


def register_user_in_db(username: str, token: str, is_admin: bool = False):
    """
    Register a user in the common DB by calling the daemon's store endpoint,
    or directly if the daemon isn't available.
    """
    from .sqlite_backend import SqliteBackend
    from .embedding import Embedder
    from .config import get_config

    cfg = get_config()
    db_path = cfg["common_db_path"] if not cfg["single_user"] else cfg["db_path"]

    # Direct DB access — CLI doesn't use HTTP auth
    embedder = Embedder()
    backend = SqliteBackend(embedder, db_path=db_path)

    hashed = hash_token(token)

    # Check if user already exists
    existing = backend.get_user_by_username(username)
    if existing:
        # Update token hash if changed
        if existing["token_hash"] != hashed:
            backend.update_user_token(username, hashed)
        backend.close()
        return existing["id"]

    user_id = backend.create_user(username, hashed, is_admin=is_admin)
    backend.close()
    return user_id
