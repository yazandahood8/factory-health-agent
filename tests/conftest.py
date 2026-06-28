"""Shared fixtures. All tests run fully offline on in-memory backends."""
from __future__ import annotations

import pytest

from sdk.config import Settings
from sdk.models import Tenant


@pytest.fixture
def offline_settings() -> Settings:
    # Empty creds → every component selects its in-memory / mock fallback.
    return Settings(
        mongodb_uri="", redis_url="", chroma_host="",
        azure_openai_api_key="", gemini_api_key="", jwt_secret="test-secret",
    )


@pytest.fixture
def tenant_a() -> Tenant:
    return Tenant(id="acme", name="ACME", llm_budget_usd=10.0)


@pytest.fixture
def tenant_b() -> Tenant:
    return Tenant(id="globex", name="Globex", llm_budget_usd=10.0)
