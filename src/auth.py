"""
OAuth2 authentication — Google + Microsoft Azure AD.
Uses authlib for OAuth flows and itsdangerous for signed session cookies.
"""
import os
import json
from functools import wraps
from typing import Optional

from authlib.integrations.requests_client import OAuth2Session
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse

# ---------------------------------------------------------------------------
# Config (read from env)
# ---------------------------------------------------------------------------

SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
SESSION_COOKIE = "rag_session"
SESSION_MAX_AGE = 60 * 60 * 8  # 8 hours

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

MS_CLIENT_ID     = os.environ.get("MS_CLIENT_ID", "")
MS_CLIENT_SECRET = os.environ.get("MS_CLIENT_SECRET", "")
MS_TENANT_ID     = os.environ.get("MS_TENANT_ID", "common")
MS_REDIRECT_URI  = os.environ.get("MS_REDIRECT_URI", "http://localhost:8000/auth/microsoft/callback")

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

_signer = URLSafeTimedSerializer(SECRET_KEY)


def create_session_cookie(user: dict) -> str:
    return _signer.dumps(user)


def read_session_cookie(token: str) -> Optional[dict]:
    try:
        return _signer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return read_session_cookie(token)


def require_auth(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# ---------------------------------------------------------------------------
# Google OAuth2
# ---------------------------------------------------------------------------

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO  = "https://www.googleapis.com/oauth2/v3/userinfo"


def google_login_url(state: str) -> str:
    client = OAuth2Session(
        client_id=GOOGLE_CLIENT_ID,
        redirect_uri=GOOGLE_REDIRECT_URI,
        scope=["openid", "email", "profile"],
    )
    url, _ = client.create_authorization_url(GOOGLE_AUTH_URL, state=state)
    return url


def google_callback(code: str) -> dict:
    client = OAuth2Session(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        redirect_uri=GOOGLE_REDIRECT_URI,
    )
    token = client.fetch_token(GOOGLE_TOKEN_URL, code=code)
    resp = client.get(GOOGLE_USERINFO)
    info = resp.json()
    return {
        "provider": "google",
        "email": info.get("email"),
        "name": info.get("name"),
        "picture": info.get("picture"),
    }

# ---------------------------------------------------------------------------
# Microsoft Azure AD OAuth2
# ---------------------------------------------------------------------------

def ms_auth_url() -> str:
    return f"https://login.microsoftonline.com/{MS_TENANT_ID}/oauth2/v2.0/authorize"

def ms_token_url() -> str:
    return f"https://login.microsoftonline.com/{MS_TENANT_ID}/oauth2/v2.0/token"

MS_USERINFO = "https://graph.microsoft.com/v1.0/me"


def microsoft_login_url(state: str) -> str:
    client = OAuth2Session(
        client_id=MS_CLIENT_ID,
        redirect_uri=MS_REDIRECT_URI,
        scope=["openid", "email", "profile", "User.Read"],
    )
    url, _ = client.create_authorization_url(ms_auth_url(), state=state)
    return url


def microsoft_callback(code: str) -> dict:
    client = OAuth2Session(
        client_id=MS_CLIENT_ID,
        client_secret=MS_CLIENT_SECRET,
        redirect_uri=MS_REDIRECT_URI,
    )
    token = client.fetch_token(ms_token_url(), code=code)
    resp = client.get(MS_USERINFO)
    info = resp.json()
    return {
        "provider": "microsoft",
        "email": info.get("mail") or info.get("userPrincipalName"),
        "name": info.get("displayName"),
        "picture": None,
    }
