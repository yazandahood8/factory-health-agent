"""Execution tracing — make the agent run observable end to end.

A :class:`TraceRecorder` collects an ordered, structured log of everything the
pipeline does: tool/store reads (with tenant scoping), RAG retrievals (with the
actual documents + similarity scores), LLM calls (provider/model/tokens/cost),
routing decisions, and guard checks. Thin wrappers around the SDK services record
automatically, so agent code stays clean and unaware of tracing.

This powers the "show everything" view in the web UI and is independent of (and
complementary to) LangSmith tracing in :mod:`sdk.observability`.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    seq: int
    t_ms: float          # ms since the run started
    kind: str            # agent | tool | rag | llm | store | route | guard
    name: str
    agent: str
    detail: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    def __init__(self, sink: Any = None) -> None:
        self._events: list[TraceEvent] = []
        self._t0 = time.perf_counter()
        self._agent = ""
        self._seq = 0
        # Optional callback(dict) fired the instant an event is recorded — used
        # to stream the pipeline live to the UI.
        self._sink = sink

    def set_agent(self, name: str) -> None:
        self._agent = name

    def add(self, kind: str, name: str, **detail: Any) -> None:
        self._seq += 1
        event = TraceEvent(
            seq=self._seq,
            t_ms=round((time.perf_counter() - self._t0) * 1000, 1),
            kind=kind,
            name=name,
            agent=self._agent,
            detail=detail,
        )
        self._events.append(event)
        if self._sink is not None:
            try:
                self._sink(asdict(event))
            except Exception:  # never let observability break the pipeline
                pass

    @property
    def events(self) -> list[TraceEvent]:
        return self._events

    def snapshot(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self._events]

    def llm_cost(self) -> float:
        return round(sum(e.detail.get("cost_usd", 0.0) for e in self._events if e.kind == "llm"), 6)

    def llm_tokens(self) -> int:
        return sum(int(e.detail.get("tokens", 0)) for e in self._events if e.kind == "llm")


def _hits(docs) -> list[dict[str, Any]]:
    return [
        {"source": d.source, "score": round(getattr(d, "score", 0.0), 3), "snippet": d.text[:180]}
        for d in docs
    ]


class TracedRAG:
    """Wraps :class:`~sdk.rag_engine.RAGEngine`, recording every retrieval."""

    def __init__(self, rag, recorder: TraceRecorder) -> None:
        self._rag = rag
        self._rec = recorder

    def retrieve_with_sources(self, query: str, collection_name: str, k: int = 5):
        t = time.perf_counter()
        res = self._rag.retrieve_with_sources(query, collection_name, k)
        self._rec.add(
            "rag", collection_name, query=query, k=k,
            latency_ms=round((time.perf_counter() - t) * 1000, 1), hits=_hits(res.documents),
        )
        return res

    def retrieve(self, query: str, collection_name: str, k: int = 5):
        res = self._rag.retrieve(query, collection_name, k)
        self._rec.add("rag", collection_name, query=query, k=k, hits=_hits(res))
        return res

    def __getattr__(self, name):
        return getattr(self._rag, name)


class TracedLLM:
    """Wraps :class:`~sdk.llm_router.LLMRouter`, recording every completion."""

    def __init__(self, llm, recorder: TraceRecorder) -> None:
        self._llm = llm
        self._rec = recorder

    def complete(self, prompt: str, tenant, *, system: str = "", task_type: str = "reasoning"):
        t = time.perf_counter()
        res = self._llm.complete(prompt, tenant, system=system, task_type=task_type)
        self._rec.add(
            "llm", res.provider, model=getattr(res, "model", res.provider) or res.provider,
            task_type=task_type, tokens=res.tokens, cost_usd=round(res.cost_usd, 6),
            latency_ms=round((time.perf_counter() - t) * 1000, 1),
            prompt_preview=prompt[:280], response_preview=res.text[:280],
        )
        return res

    def __getattr__(self, name):
        return getattr(self._llm, name)


class TracedStore:
    """Wraps :class:`~sdk.tenant_store.TenantIsolatedStore`, recording each access.

    Surfaces that every read is tenant-scoped — the multi-tenant isolation made
    visible in the trace.
    """

    def __init__(self, store, recorder: TraceRecorder) -> None:
        self._store = store
        self._rec = recorder

    def query(self, collection: str, query: dict, tenant):
        rows = self._store.query(collection, query, tenant)
        self._rec.add(
            "store", collection, op="query", tenant_id=tenant.id,
            filter={k: v for k, v in query.items()}, rows=len(rows),
        )
        return rows

    def insert(self, collection: str, doc: dict, tenant):
        res = self._store.insert(collection, doc, tenant)
        self._rec.add("store", collection, op="insert", tenant_id=tenant.id)
        return res

    def __getattr__(self, name):
        return getattr(self._store, name)
