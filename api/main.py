"""FastAPI application entrypoint.

Wires middleware (auth → rate limit), builds shared services + the compiled
LangGraph pipeline once at startup, and mounts the route modules.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from agents.orchestrator import Pipeline, build_services
from api.middleware.auth import AuthMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.routes import analyze, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = build_services(seed=True)
    app.state.services = services
    app.state.pipeline = Pipeline(services)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Factory Health Agent",
        version="1.0.0",
        description="GenAI infrastructure platform for industrial machine diagnostics.",
        lifespan=lifespan,
    )
    # Middleware runs in reverse order of registration: auth first, then rate limit.
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)

    app.include_router(health.router)
    app.include_router(analyze.router)

    @app.get("/")
    async def root():
        return {"service": "factory-health-agent", "docs": "/docs", "health": "/v1/health"}

    return app


app = create_app()
