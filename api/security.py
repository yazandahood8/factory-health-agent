"""JWT helpers shared by middleware and tests.

Tokens carry the tenant identity and budget. A small ``mint_token`` helper is
exposed so the demo / tests can produce valid tokens without an external IdP.
"""
from __future__ import annotations

import time
from typing import Any

import jwt

from sdk.config import Settings, get_settings
from sdk.models import Tenant, Tier


def mint_token(
    tenant_id: str,
    *,
    name: str = "",
    tier: str = "pro",
    budget_usd: float = 10.0,
    ttl_seconds: int = 3600,
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()
    payload = {
        "tenant_id": tenant_id,
        "name": name or tenant_id,
        "tier": tier,
        "budget_usd": budget_usd,
        "iat": int(time.time()),
        "exp": int(time.time()) + ttl_seconds,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


def tenant_from_claims(claims: dict[str, Any]) -> Tenant:
    return Tenant(
        id=claims["tenant_id"],
        name=claims.get("name", ""),
        tier=Tier(claims.get("tier", "pro")),
        llm_budget_usd=float(claims.get("budget_usd", 10.0)),
    )
