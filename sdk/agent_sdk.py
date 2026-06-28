"""Main SDK entry point.

Wires LLM Router, Tenant Store, RAG Engine, Hallucination Guard, and
LangSmith tracing into a single :class:`AgentSDK` object. Domain agents
receive an :class:`AgentContext` scoped to one tenant and never construct
infrastructure themselves.

Usage::

    from sdk.agent_sdk import AgentSDK
    sdk = AgentSDK()
    ctx = sdk.context_for(tenant)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sdk.config import Settings, get_settings
from sdk.hallucination_guard import HallucinationGuard
from sdk.llm_router import LLMRouter
from sdk.models import Tenant
from sdk.observability import enable_tracing
from sdk.rag_engine import RAGEngine
from sdk.tenant_store import TenantIsolatedStore


@dataclass
class AgentSDK:
    """Process-wide singletons shared across all requests and tenants.

    Build once at application startup; call :meth:`context_for` per request.
    """

    settings: Settings
    store: TenantIsolatedStore
    rag: RAGEngine
    llm: LLMRouter
    guard: HallucinationGuard

    def context_for(self, tenant: Tenant):
        from agents.base import AgentContext

        return AgentContext(
            tenant=tenant,
            store=self.store,
            rag=self.rag,
            llm=self.llm,
            guard=self.guard,
        )


def build_sdk(settings: Optional[Settings] = None) -> AgentSDK:
    """Construct a fully wired :class:`AgentSDK` from the current configuration."""
    settings = settings or get_settings()
    enable_tracing(settings)
    return AgentSDK(
        settings=settings,
        store=TenantIsolatedStore(settings),
        rag=RAGEngine(settings),
        llm=LLMRouter(settings),
        guard=HallucinationGuard(threshold=0.8),
    )
