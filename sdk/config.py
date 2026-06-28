"""Central configuration.

Loaded once from the environment (and optional ``.env``). Every value has a
safe default so the platform boots with zero configuration — missing
credentials transparently route components to in-memory fallbacks.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM providers ---
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o"
    openai_api_version: str = "2024-08-01-preview"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # --- Observability ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "factory-health-agent"

    # --- Data layer ---
    mongodb_uri: str = ""
    mongodb_db: str = "factory_health"
    redis_url: str = ""
    chroma_host: str = ""
    chroma_port: int = 8000
    chroma_persist_dir: str = "./chroma_data"

    # --- API security ---
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    rate_limit_per_minute: int = 100

    # --- Defaults ---
    default_tenant_budget_usd: float = 10.0

    @property
    def azure_enabled(self) -> bool:
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
