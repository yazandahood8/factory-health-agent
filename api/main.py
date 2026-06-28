"""FastAPI application entrypoint.

Wires middleware (auth → rate limit), builds shared services + the compiled
LangGraph pipeline once at startup, and mounts the route modules.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from agents.orchestrator import Pipeline, build_services
from api.metrics_collector import MetricsCollector
from api.middleware.auth import AuthMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.routes import analyze, demo, health

_UI_PATH = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = build_services(seed=True)
    app.state.services = services
    app.state.pipeline = Pipeline(services)
    app.state.metrics = MetricsCollector()
    yield


class _RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Records per-tenant latency and error rate for /v1/metrics."""

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        tenant = getattr(request.state, "tenant", None)
        if tenant and hasattr(request.app.state, "metrics"):
            request.app.state.metrics.record(
                tenant.id, elapsed, error=response.status_code >= 400
            )
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="Factory Health Agent",
        version="1.0.0",
        description="GenAI infrastructure platform for industrial machine diagnostics.",
        lifespan=lifespan,
    )
    # Middleware runs in reverse order: auth → rate limit → request metrics.
    app.add_middleware(_RequestMetricsMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)

    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(demo.router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def ui():
        return HTMLResponse(_UI_PATH.read_text(encoding="utf-8"))

    @app.get("/v1/info", include_in_schema=False)
    async def info():
        return {"service": "factory-health-agent", "docs": "/docs", "health": "/v1/health"}

    return app


app = create_app()
