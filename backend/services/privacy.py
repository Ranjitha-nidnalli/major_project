"""
Privacy helpers (K-044). The raw phone number or chat ID is used only to reply to the
message in front of us; it is never stored. Everything stored is keyed by a salted HMAC.
"""
import hashlib
import hmac
import os

# How long answers (with the question text) are kept. Stated to farmers in the consent text.
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "90"))


def hash_user(channel, raw_id, salt=None):
    """Stable pseudonym for one person on one channel. Needs PRIVACY_SALT (never committed)."""
    salt = salt or os.getenv("PRIVACY_SALT")
    if not salt:
        raise RuntimeError("PRIVACY_SALT is not set; refusing to store anything about a user")
    msg = f"{channel}:{raw_id}".encode("utf-8")
    return hmac.new(salt.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:32]
