"""Shared agent context.

Every agent receives an :class:`AgentContext` — the bundle of SDK services it
is allowed to touch (LLM router, tenant store, RAG, guard) scoped to a single
tenant. Agents never construct infrastructure themselves; they are handed it.
This is what keeps an agent ~one file of domain logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sdk.hallucination_guard import HallucinationGuard
from sdk.llm_router import LLMRouter
from sdk.models import Tenant
from sdk.rag_engine import RAGEngine
from sdk.tenant_store import TenantIsolatedStore
from sdk.trace import TraceRecorder


@dataclass
class AgentContext:
    tenant: Tenant
    store: TenantIsolatedStore
    rag: RAGEngine
    llm: LLMRouter
    guard: HallucinationGuard
    # When set (via the instrumented context), service calls are traced.
    recorder: Optional[TraceRecorder] = None
