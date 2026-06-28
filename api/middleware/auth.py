"""Auth + tenant-context middleware.

Validates the JWT, cross-checks the ``X-Tenant-ID`` header against the token
claim (so a token can't be replayed against another tenant's namespace), and
attaches a :class:`Tenant` to ``request.state``. Health/docs are open; every
``/v1`` data endpoint is closed by default.
"""
from __future__ import annotations

import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from api.security import decode_token, tenant_from_claims
from sdk.config import get_settings

OPEN_PATHS = {"/v1/health", "/docs", "/openapi.json", "/redoc", "/"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in OPEN_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse({"detail": "Missing bearer token"}, status_code=401)
        token = auth.split(" ", 1)[1].strip()

        try:
            claims = decode_token(token, get_settings())
        except jwt.ExpiredSignatureError:
            return JSONResponse({"detail": "Token expired"}, status_code=401)
        except jwt.InvalidTokenError:
            return JSONResponse({"detail": "Invalid token"}, status_code=401)

        header_tenant = request.headers.get("x-tenant-id")
        if header_tenant and header_tenant != claims.get("tenant_id"):
            return JSONResponse(
                {"detail": "X-Tenant-ID does not match token"}, status_code=403
            )

        request.state.tenant = tenant_from_claims(claims)
        return await call_next(request)
