"""Supabase-backed Google OAuth authentication and role checks.

Required environment:
  SUPABASE_URL
  SUPABASE_ANON_KEY (or SUPABASE_PUBLISHABLE_KEY)
  SUPABASE_ADMIN_EMAIL (the one Google account allowed to download GLBs)
"""
from __future__ import annotations

import os
import secrets
import hmac
from pathlib import Path
from typing import Callable

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY = (
    os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
    or os.getenv("SUPABASE_ANON_KEY", "").strip()
)
ADMIN_EMAIL = os.getenv("SUPABASE_ADMIN_EMAIL", "").strip().lower()
SESSION_COOKIE_NAME = "defirmation_supabase_access"
REFRESH_COOKIE_NAME = "defirmation_supabase_refresh"
CSRF_COOKIE_NAME = "defirmation_csrf"
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", os.getenv("APP_ENV", "development").lower() == "production")
COOKIE_SECURE = str(COOKIE_SECURE).lower() in {"1", "true", "yes", "on"}
COOKIE_MAX_AGE = 60 * 60 * 24 * 30
AUTH_TIMEOUT = 8

auth_router = APIRouter(prefix="/api/auth", tags=["authentication"])


def _configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _auth_headers(access_token: str | None = None) -> dict[str, str]:
    headers = {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def _get_supabase_user(access_token: str) -> dict | None:
    if not _configured() or not access_token:
        return None
    try:
        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers=_auth_headers(access_token),
            timeout=AUTH_TIMEOUT,
        )
        if response.status_code != 200:
            return None
        user = response.json()
        return user if isinstance(user, dict) and user.get("id") and user.get("email") else None
    except (requests.RequestException, ValueError):
        return None


def _refresh_session(refresh_token: str) -> dict | None:
    if not _configured() or not refresh_token:
        return None
    try:
        response = requests.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
            headers=_auth_headers(),
            json={"refresh_token": refresh_token},
            timeout=AUTH_TIMEOUT,
        )
        if response.status_code != 200:
            return None
        data = response.json()
        if data.get("access_token") and data.get("refresh_token"):
            return data
        return None
    except (requests.RequestException, ValueError):
        return None


def _set_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME, access_token, max_age=COOKIE_MAX_AGE,
        httponly=True, secure=COOKIE_SECURE, samesite="lax", path="/",
    )
    response.set_cookie(
        REFRESH_COOKIE_NAME, refresh_token, max_age=COOKIE_MAX_AGE,
        httponly=True, secure=COOKIE_SECURE, samesite="lax", path="/",
    )
    response.set_cookie(
        CSRF_COOKIE_NAME, secrets.token_urlsafe(32), max_age=COOKIE_MAX_AGE,
        httponly=False, secure=COOKIE_SECURE, samesite="lax", path="/",
    )


def clear_session_cookies(response: Response) -> None:
    for name, http_only in ((SESSION_COOKIE_NAME, True), (REFRESH_COOKIE_NAME, True), (CSRF_COOKIE_NAME, False)):
        response.delete_cookie(
            name, path="/", secure=COOKIE_SECURE, httponly=http_only, samesite="lax",
        )


def csrf_token_is_valid(request: Request) -> bool:
    cookie = request.cookies.get(CSRF_COOKIE_NAME, "")
    header = request.headers.get("X-CSRF-Token", "")
    return bool(cookie and header and hmac.compare_digest(cookie, header))


def get_current_user_optional(request: Request) -> dict | None:
    access_token = request.cookies.get(SESSION_COOKIE_NAME, "")
    if not access_token:
        return None

    raw_user = _get_supabase_user(access_token)
    if raw_user is None:
        refreshed = _refresh_session(request.cookies.get(REFRESH_COOKIE_NAME, ""))
        if not refreshed:
            return None
        raw_user = _get_supabase_user(refreshed["access_token"])
        if raw_user is None:
            return None
        request.state.supabase_refreshed_session = refreshed

    email = str(raw_user.get("email", "")).strip().lower()
    role = "owner" if ADMIN_EMAIL and email == ADMIN_EMAIL else "editor"
    metadata = raw_user.get("user_metadata") or {}
    display_name = (
        metadata.get("full_name") or metadata.get("name")
        or raw_user.get("email") or "Google user"
    )
    return {
        "id": raw_user["id"],
        "username": email,
        "email": email,
        "display_name": str(display_name)[:120],
        "role": role,
        "is_active": True,
        "supabase_user": raw_user,
    }


def get_current_user(request: Request) -> dict:
    user = get_current_user_optional(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign-in required. Please sign in with Google.")
    return user


def require_roles(*roles: str) -> Callable:
    allowed = set(roles)

    def dependency(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed:
            raise HTTPException(status_code=403, detail="Your account does not have permission for this action.")
        return user

    return dependency


class SessionRequest(BaseModel):
    access_token: str = Field(min_length=20, max_length=8192)
    refresh_token: str = Field(min_length=20, max_length=8192)


@auth_router.get("/config")
def auth_config() -> dict:
    if not _configured():
        return {"configured": False}
    return {"configured": True, "supabase_url": SUPABASE_URL, "supabase_key": SUPABASE_KEY}


@auth_router.get("/status")
def auth_status() -> dict:
    return {"configured": _configured(), "setup_required": False, "provider": "google"}


@auth_router.post("/session")
def create_session(body: SessionRequest, response: Response) -> dict:
    if not _configured():
        raise HTTPException(status_code=503, detail="Supabase is not configured on the server.")
    user = _get_supabase_user(body.access_token)
    if not user:
        raise HTTPException(status_code=401, detail="Supabase could not verify this Google session.")
    _set_cookies(response, body.access_token, body.refresh_token)
    email = str(user.get("email", "")).strip().lower()
    return {"user": {"id": user["id"], "email": email}, "message": "Signed in."}


@auth_router.post("/logout")
def logout(response: Response) -> dict:
    clear_session_cookies(response)
    return {"message": "Signed out."}


@auth_router.get("/me")
def current_user_profile(request: Request, response: Response, user: dict = Depends(get_current_user)) -> dict:
    refreshed = getattr(request.state, "supabase_refreshed_session", None)
    if refreshed:
        _set_cookies(response, refreshed["access_token"], refreshed["refresh_token"])
    permissions = ["view_models", "generate_models", "rim_calibration"]
    if user["role"] == "owner":
        permissions.extend(["download_glb", "user_management", "settings"])
    return {
        "user": {key: user[key] for key in ("id", "username", "email", "display_name", "role", "is_active")},
        "permissions": permissions,
    }
