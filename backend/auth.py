"""Application authentication and role-based access control.

The first Owner is created once through /api/auth/setup using AUTH_BOOTSTRAP_TOKEN.
No public sign-up or default password is provided.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is in requirements.txt
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"
SESSION_COOKIE_NAME = "defirmation_session"
CSRF_COOKIE_NAME = "defirmation_csrf"
SESSION_TTL_SECONDS = 8 * 60 * 60
PASSWORD_ITERATIONS = 310_000
ALLOWED_USER_ROLES = {"editor", "operator", "viewer"}
ROLE_PERMISSIONS = {
    "owner": ["user_management", "settings", "generate_models", "rim_calibration", "view_models"],
    "editor": ["generate_models", "rim_calibration", "view_models"],
    "operator": ["generate_models", "view_models"],
    "viewer": ["view_models"],
}


def _load_signing_secret() -> str:
    configured = os.getenv("AUTH_SECRET_KEY", "").strip()
    if configured:
        if len(configured) < 32:
            raise RuntimeError("AUTH_SECRET_KEY must contain at least 32 characters.")
        return configured

    if IS_PRODUCTION:
        raise RuntimeError(
            "AUTH_SECRET_KEY must be configured in the production environment. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )

    # Local development convenience: persist a private random key outside Git.
    secret_path = PROJECT_ROOT / ".auth_secret"
    try:
        if secret_path.exists():
            secret = secret_path.read_text(encoding="utf-8").strip()
        else:
            secret = secrets.token_urlsafe(48)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(secret_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(secret)
        if len(secret) < 32:
            raise RuntimeError("The local .auth_secret file is invalid; replace it with a new random secret.")
        return secret
    except OSError as exc:
        raise RuntimeError(
            "Could not create/read .auth_secret. Set AUTH_SECRET_KEY in your environment."
        ) from exc


SIGNING_SECRET = _load_signing_secret()
BOOTSTRAP_TOKEN = os.getenv("AUTH_BOOTSTRAP_TOKEN", "").strip()
COOKIE_SECURE = os.getenv(
    "AUTH_COOKIE_SECURE", "true" if IS_PRODUCTION else "false"
).strip().lower() in {"1", "true", "yes", "on"}

_db_setting = os.getenv("AUTH_DB_PATH", "data/auth.sqlite3")
DB_PATH = Path(_db_setting)
if not DB_PATH.is_absolute():
    DB_PATH = PROJECT_ROOT / DB_PATH
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(str(DB_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_auth_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('owner', 'editor', 'operator', 'viewer')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL,
                created_by TEXT,
                last_login_at INTEGER
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")


initialize_auth_database()


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${_b64encode(salt)}${_b64encode(digest)}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        if iterations < 100_000 or iterations > 2_000_000:
            return False
        salt = _b64decode(salt_text)
        expected = _b64decode(digest_text)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _create_session_token(user_id: str) -> str:
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + SESSION_TTL_SECONDS,
        "jti": secrets.token_urlsafe(12),
    }
    encoded_payload = _b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(
        SIGNING_SECRET.encode("utf-8"),
        encoded_payload.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{encoded_payload}.{_b64encode(signature)}"


def _read_session_token(token: str | None) -> dict | None:
    if not token:
        return None
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        expected_signature = hmac.new(
            SIGNING_SECRET.encode("utf-8"),
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected_signature, _b64decode(encoded_signature)):
            return None
        payload = json.loads(_b64decode(encoded_payload))
        now = int(time.time())
        if not isinstance(payload, dict) or int(payload.get("exp", 0)) <= now:
            return None
        if int(payload.get("iat", 0)) > now + 60 or not payload.get("sub"):
            return None
        return payload
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def set_session_cookies(response: Response, user_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        _create_session_token(user_id),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="strict",
        path="/",
    )
    response.set_cookie(
        CSRF_COOKIE_NAME,
        secrets.token_urlsafe(32),
        max_age=SESSION_TTL_SECONDS,
        httponly=False,
        secure=COOKIE_SECURE,
        samesite="strict",
        path="/",
    )


def clear_session_cookies(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/", secure=COOKIE_SECURE, httponly=True, samesite="strict")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/", secure=COOKIE_SECURE, httponly=False, samesite="strict")


def csrf_token_is_valid(request: Request) -> bool:
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME, "")
    header_token = request.headers.get("X-CSRF-Token", "")
    return bool(cookie_token and header_token and hmac.compare_digest(cookie_token, header_token))


def _normalise_username(username: str) -> str:
    value = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,49}", value):
        raise HTTPException(
            status_code=422,
            detail="Username must be 3–50 characters and use letters, numbers, dots, underscores or hyphens.",
        )
    return value


def _validate_password(password: str) -> None:
    if len(password) < 12:
        raise HTTPException(status_code=422, detail="Password must contain at least 12 characters.")
    if len(password) > 256:
        raise HTTPException(status_code=422, detail="Password is too long.")


def _public_user(row: sqlite3.Row | dict) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def _owner_exists(connection: sqlite3.Connection) -> bool:
    row = connection.execute("SELECT 1 FROM users WHERE role = 'owner' LIMIT 1").fetchone()
    return row is not None


def get_current_user_optional(request: Request) -> dict | None:
    payload = _read_session_token(request.cookies.get(SESSION_COOKIE_NAME))
    if not payload:
        return None
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE id = ? AND is_active = 1",
            (payload["sub"],),
        ).fetchone()
    return dict(row) if row else None


def get_current_user(request: Request) -> dict:
    user = get_current_user_optional(request)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in again.",
            headers={"WWW-Authenticate": "Cookie"},
        )
    return user


def require_roles(*roles: str):
    allowed = set(roles)

    def dependency(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed:
            raise HTTPException(status_code=403, detail="Your account does not have permission for this action.")
        return user

    return dependency


class SetupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=12, max_length=256)
    bootstrap_token: str = Field(min_length=1, max_length=512)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=256)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=12, max_length=256)
    role: Literal["editor", "operator", "viewer"]

    @field_validator("display_name")
    @classmethod
    def clean_display_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Display name cannot be empty.")
        return value


class UserUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    role: Literal["editor", "operator", "viewer"] | None = None
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=12, max_length=256)

    @field_validator("display_name")
    @classmethod
    def clean_optional_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("Display name cannot be empty.")
        return value


auth_router = APIRouter(prefix="/api/auth", tags=["authentication"])
admin_router = APIRouter(
    prefix="/api/admin",
    tags=["administration"],
    dependencies=[Depends(require_roles("owner"))],
)


@auth_router.get("/status")
def authentication_status() -> dict:
    with _connect() as connection:
        setup_required = not _owner_exists(connection)
    return {
        "setup_required": setup_required,
        "bootstrap_configured": bool(BOOTSTRAP_TOKEN),
        "session_hours": SESSION_TTL_SECONDS // 3600,
    }


@auth_router.post("/setup")
def setup_owner(body: SetupRequest, response: Response) -> dict:
    if not BOOTSTRAP_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Initial setup is disabled. Configure AUTH_BOOTSTRAP_TOKEN in the server environment and restart the app.",
        )
    if not hmac.compare_digest(body.bootstrap_token, BOOTSTRAP_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid initial setup token.")

    username = _normalise_username(body.username)
    display_name = body.display_name.strip()
    if not display_name:
        raise HTTPException(status_code=422, detail="Display name cannot be empty.")
    _validate_password(body.password)

    user_id = str(uuid.uuid4())
    now = int(time.time())
    try:
        with _connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if _owner_exists(connection):
                raise HTTPException(status_code=409, detail="The Owner account has already been created.")
            connection.execute(
                """
                INSERT INTO users (id, username, display_name, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, ?, 'owner', 1, ?)
                """,
                (user_id, username, display_name, _hash_password(body.password), now),
            )
            connection.commit()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="That username is already in use.") from exc

    set_session_cookies(response, user_id)
    return {
        "message": "Owner account created. Keep your credentials private.",
        "user": {
            "id": user_id,
            "username": username,
            "display_name": display_name,
            "role": "owner",
            "is_active": True,
            "created_at": now,
            "last_login_at": None,
        },
    }


@auth_router.post("/login")
def login(body: LoginRequest, response: Response) -> dict:
    username = body.username.strip().lower()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()

    if not row:
        _hash_password("not-a-real-password-value")
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    if not row["is_active"] or not _verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    now = int(time.time())
    with _connect() as connection:
        connection.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, row["id"]))
    set_session_cookies(response, row["id"])
    return {
        "message": "Signed in successfully.",
        "user": {
            **_public_user(row),
            "last_login_at": now,
        },
    }


@auth_router.post("/logout")
def logout(response: Response, user: dict = Depends(get_current_user)) -> dict:
    clear_session_cookies(response)
    return {"message": "Signed out successfully."}


@auth_router.get("/me")
def current_user_profile(user: dict = Depends(get_current_user)) -> dict:
    return {
        "user": _public_user(user),
        "permissions": ROLE_PERMISSIONS.get(user["role"], []),
    }


@auth_router.post("/change-password")
def change_password(
    body: ChangePasswordRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    _validate_password(body.new_password)
    if not _verify_password(body.current_password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    with _connect() as connection:
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (_hash_password(body.new_password), user["id"]),
        )
    return {"message": "Password changed. Your existing session remains active until it expires."}


@admin_router.get("/users")
def list_users() -> dict:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, username, display_name, role, is_active, created_at, last_login_at
            FROM users
            ORDER BY CASE role WHEN 'owner' THEN 0 ELSE 1 END, username COLLATE NOCASE
            """
        ).fetchall()
    return {"users": [_public_user(row) for row in rows]}


@admin_router.post("/users", status_code=201)
def create_user(body: UserCreateRequest, owner: dict = Depends(require_roles("owner"))) -> dict:
    username = _normalise_username(body.username)
    display_name = body.display_name.strip()
    if not display_name:
        raise HTTPException(status_code=422, detail="Display name cannot be empty.")
    _validate_password(body.password)

    user_id = str(uuid.uuid4())
    now = int(time.time())
    try:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO users (id, username, display_name, password_hash, role, is_active, created_at, created_by)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (user_id, username, display_name, _hash_password(body.password), body.role, now, owner["id"]),
            )
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="That username is already in use.") from exc
    return {"message": "User account created.", "user": _public_user(row)}


@admin_router.patch("/users/{user_id}")
def update_user(user_id: str, body: UserUpdateRequest) -> dict:
    changes = body.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=400, detail="No changes were supplied.")

    with _connect() as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found.")
        if row["role"] == "owner":
            raise HTTPException(status_code=403, detail="The Owner account cannot be changed or disabled from user management.")

        fields: list[str] = []
        values: list[object] = []
        for field in ("display_name", "role", "is_active"):
            if field in changes and changes[field] is not None:
                fields.append(f"{field} = ?")
                values.append(int(changes[field]) if field == "is_active" else changes[field])
        if changes.get("password"):
            _validate_password(changes["password"])
            fields.append("password_hash = ?")
            values.append(_hash_password(changes["password"]))
        if not fields:
            raise HTTPException(status_code=400, detail="No valid changes were supplied.")
        values.append(user_id)
        connection.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", values)
        updated = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return {"message": "User account updated.", "user": _public_user(updated)}
