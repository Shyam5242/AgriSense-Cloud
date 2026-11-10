"""
auth_token.py - signed, expiring login tokens so a browser refresh does not log you out.

The token is  "<expiry-unix-time>.<HMAC-SHA256 of the expiry, keyed with the password>".
Changing the password invalidates every token. Tokens expire after `ttl` seconds (default 8 hours).
"""
from __future__ import annotations

import hashlib
import hmac
import time

DEFAULT_TTL = 8 * 3600


def _sig(password: str, expiry: str) -> str:
    return hmac.new(password.encode(), expiry.encode(), hashlib.sha256).hexdigest()


def make_token(password: str, ttl: int = DEFAULT_TTL, now: float | None = None) -> str:
    expiry = str(int((now if now is not None else time.time()) + ttl))
    return f"{expiry}.{_sig(password, expiry)}"


def valid_token(password: str, token, now: float | None = None) -> bool:
    if not password or not isinstance(token, str) or "." not in token:
        return False
    expiry, sig = token.split(".", 1)
    if not expiry.isdigit():
        return False
    if int(expiry) < (now if now is not None else time.time()):
        return False
    return hmac.compare_digest(sig, _sig(password, expiry))
