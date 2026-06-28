"""Health + metrics endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import HealthResponse, MetricsResponse
from sdk.config import get_settings

router = APIRouter(prefix="/v1", tags=["ops"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    settings = get_settings()
    services = request.app.state.services

    def probe_mongo() -> str:
        if not settings.mongodb_uri:
            return "in-memory"
        try:
            services.store._db.command("ping")
            return "connected"
        except Exception:  # pragma: no cover
            return "degraded"

    components = {
        "mongodb": probe_mongo(),
        "chromadb": type(services.rag.backend).__name__,
        "redis": "connected" if settings.redis_url else "in-memory",
        "llm_primary": services.llm.primary,
    }
    return HealthResponse(status="ok", components=components)


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(request: Request):
    services = request.app.state.services
    collector = request.app.state.metrics
    tenant = request.state.tenant
    stats = collector.stats(tenant.id)
    return MetricsResponse(
        tenant_id=tenant.id,
        spend_usd=round(services.llm.budget_manager.current_spend(tenant), 6),
        budget_usd=tenant.llm_budget_usd,
        llm_primary=services.llm.primary,
        requests_total=stats.requests,
        error_rate=round(stats.error_rate, 4),
        p95_latency_ms=stats.p95_latency_ms,
    )
