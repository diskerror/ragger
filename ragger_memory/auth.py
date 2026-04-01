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
import hmac


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


# ---- Password hashing (PBKDF2-SHA256) ----

PBKDF2_ITERATIONS = 600000
PBKDF2_SALT_LEN = 16
PBKDF2_KEY_LEN = 32


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-SHA256. Returns 'pbkdf2:iterations:salt_hex:key_hex'."""
    salt = os.urandom(PBKDF2_SALT_LEN)
    key = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, PBKDF2_ITERATIONS, dklen=PBKDF2_KEY_LEN
    )
    return f"pbkdf2:{PBKDF2_ITERATIONS}:{salt.hex()}:{key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored PBKDF2 hash string."""
    if not stored_hash.startswith("pbkdf2:"):
        return False
    parts = stored_hash.split(":")
    if len(parts) != 4:
        return False
    iterations = int(parts[1])
    salt = bytes.fromhex(parts[2])
    expected_key = bytes.fromhex(parts[3])
    key = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, iterations, dklen=len(expected_key)
    )
    return hmac.compare_digest(key, expected_key)


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

    # Permissions: 0660 (owner+group rw for daemon access)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)

    # Set ownership: user owns, ragger group for daemon access
    if os.getuid() == 0:
        import pwd
        import grp as _grp
        try:
            pw = pwd.getpwnam(username)
            try:
                rg = _grp.getgrnam("ragger")
                gid = rg.gr_gid
            except KeyError:
                gid = pw.pw_gid
            os.chown(ragger_dir, pw.pw_uid, gid)
            os.chmod(ragger_dir, 0o770)
            os.chown(path, pw.pw_uid, gid)
            # Also fix memories.db if it exists
            db_path = os.path.join(ragger_dir, "memories.db")
            if os.path.exists(db_path):
                os.chown(db_path, pw.pw_uid, gid)
                os.chmod(db_path, 0o660)
        except KeyError:
            pass  # user doesn't exist in passwd — skip chown

    return token, True


def rotate_token_for_user(username: str, home_dir: str | None = None) -> tuple[str, str]:
    """
    Rotate a user's token: generate new token, write to file, return new token + hash.
    
    Args:
        username: System username
        home_dir: Override home directory (default: look up from passwd)
    
    Returns:
        (new_token, new_token_hash) tuple
    """
    if home_dir is None:
        import pwd
        pw = pwd.getpwnam(username)
        home_dir = pw.pw_dir
    
    path = os.path.join(home_dir, ".ragger", "token")
    
    # Generate new token
    new_token = generate_token()
    
    # Write to file
    with open(path, "w") as f:
        f.write(new_token + "\n")
    
    # Ensure permissions are still correct (0640)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
    
    return new_token, hash_token(new_token)


def rotate_token_for_user(username: str, home_dir: str | None = None) -> tuple[str, str]:
    """
    Generate a new token for a user and write it to their token file.
    
    Args:
        username: System username
        home_dir: Override home directory (default: look up from passwd)
    
    Returns:
        (new_token, new_hash) — the new token and its SHA-256 hash
    """
    if home_dir is None:
        import pwd
        pw = pwd.getpwnam(username)
        home_dir = pw.pw_dir
    
    # Generate new token
    new_token = generate_token()
    
    # Write to user's token file
    ragger_dir = os.path.join(home_dir, ".ragger")
    token_file = os.path.join(ragger_dir, "token")
    
    # Ensure directory exists
    os.makedirs(ragger_dir, exist_ok=True)
    
    with open(token_file, "w") as f:
        f.write(new_token + "\n")
    
    # Set permissions: 0640 (owner rw + group read for daemon)
    os.chmod(token_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
    
    # Set ownership if running as root
    if os.getuid() == 0:
        import pwd
        try:
            pw = pwd.getpwnam(username)
            os.chown(token_file, pw.pw_uid, pw.pw_gid)
        except KeyError:
            pass
    
    new_hash = hash_token(new_token)
    return new_token, new_hash
