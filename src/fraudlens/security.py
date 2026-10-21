"""Access control for the deployed services.

Scoring API: role-based API keys, sent in the `X-API-Key` header.

    FRAUDLENS_API_KEYS="scorer:<key>,analyst:<key>,admin:<key>"

    role      may call
    scorer    POST /score, /score/batch           (the payment system)
    analyst   GET /alerts, /alerts/{id}, POST /alerts/{id}/decision, /feedback,
              GET /monitoring/feedback            (fraud analysts)
    admin     everything

/health and /metrics stay open (no transaction data). When FRAUDLENS_API_KEYS is not set the
API runs in development mode with access control off, and says so in /health.

Dashboard: named users with a role, in FRAUDLENS_DASHBOARD_USERS (or Streamlit secrets `users`):

    FRAUDLENS_DASHBOARD_USERS="alice:analyst:<password or pbkdf2 hash>,bob:viewer:<...>"

Viewers can explore; only analysts record decisions, and every decision carries the
signed-in name. Hash a password with `python -m fraudlens.security hash`.
"""
from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import os
import secrets
import sys

ROLES = ("scorer", "analyst", "admin", "viewer")
PBKDF2_ROUNDS = 200_000


def parse_keys(spec: str | None) -> dict[str, str]:
    """'role:key,role:key' -> {key: role}."""
    out = {}
    for item in filter(None, (spec or "").split(",")):
        role, _, key = item.strip().partition(":")
        role, key = role.strip(), key.strip()
        if role not in ROLES or not key:
            raise ValueError(f"bad API key entry {item!r}: expected role:key with role in {ROLES}")
        out[key] = role
    return out


def role_for_key(keys: dict[str, str], given: str | None) -> str | None:
    if not given:
        return None
    for key, role in keys.items():
        if hmac.compare_digest(key.encode(), given.encode()):
            return role
    return None


def key_id(key: str | None) -> str | None:
    """A non-secret identifier for audit logs."""
    return None if not key else "key-" + hashlib.sha256(key.encode()).hexdigest()[:8]


# ------------------------------------------------------------------ dashboard users
def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return "pbkdf2$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def check_password(stored: str, given: str) -> bool:
    if stored.startswith("pbkdf2$"):
        _, salt, dk = stored.split("$")
        test = hashlib.pbkdf2_hmac("sha256", given.encode(), base64.b64decode(salt), PBKDF2_ROUNDS)
        return hmac.compare_digest(test, base64.b64decode(dk))
    return hmac.compare_digest(stored.encode(), given.encode())


def parse_users(spec) -> dict[str, dict]:
    """'name:role:secret,...' (or a mapping from Streamlit secrets) -> {name: {role, secret}}."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {n: {"role": v["role"], "secret": v["password"]} for n, v in spec.items()}
    out = {}
    for item in filter(None, spec.split(",")):
        name, role, secret = item.strip().split(":", 2)
        if role not in ("analyst", "viewer", "admin"):
            raise ValueError(f"bad dashboard user role {role!r}")
        out[name] = {"role": role, "secret": secret}
    return out


def authenticate(users: dict[str, dict], name: str, password: str) -> str | None:
    """The user's role if the name and password match, else None."""
    u = users.get(name)
    return u["role"] if u and check_password(u["secret"], password) else None


def env_users():
    return parse_users(os.environ.get("FRAUDLENS_DASHBOARD_USERS"))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "hash":
        print(hash_password(getpass.getpass("Password: ")))
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "keys":
        print(",".join(f"{r}:{secrets.token_urlsafe(24)}" for r in ("scorer", "analyst", "admin")))
        return 0
    print("usage: python -m fraudlens.security hash | keys")
    return 1


if __name__ == "__main__":
    sys.exit(main())
