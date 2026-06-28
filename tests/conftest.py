"""Shared fixtures. All tests run fully offline on in-memory backends."""
from __future__ import annotations

import os

import pytest

from sdk.config import Settings, get_settings
from sdk.models import Tenant


@pytest.fixture(autouse=True, scope="session")
def _force_offline():
    """Make the whole suite hermetic: no real LLM/DB calls regardless of .env.

    Env vars take precedence over .env in pydantic-settings, so blanking the
    credentials here forces every component onto its mock/in-memory fallback.
    """
    for var in (
        "GEMINI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
        "MONGODB_URI", "REDIS_URL", "CHROMA_HOST",
    ):
        os.environ[var] = ""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["JWT_SECRET"] = "test-secret-test-secret-test-secret-32b"
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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
