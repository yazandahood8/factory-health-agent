# Architecture

## Three layers

### Layer 1 — API Gateway (`api/`)
FastAPI app with two middlewares applied outermost-first:

1. **AuthMiddleware** — validates the JWT, cross-checks `X-Tenant-ID` against the
   token claim, attaches a `Tenant` to `request.state`. Health/docs are open.
2. **RateLimitMiddleware** — fixed-window, per-tenant (Redis or in-memory),
   returns `429` + `Retry-After`.

Services and the compiled LangGraph pipeline are built once in the FastAPI
`lifespan` and shared across requests.

### Layer 2 — GenAI SDK (`sdk/`)
The reusable infrastructure. Nothing here knows about pumps or gearboxes.

| Module | Responsibility |
|--------|----------------|
| `config.py` | One settings object; safe defaults → in-memory fallbacks |
| `llm_router.py` | Provider chain Azure → Gemini → Mock, fail-over, cost accounting |
| `budget.py` | Per-tenant spend tracking + enforcement (Redis/in-memory) |
| `tenant_store.py` | Tenant-isolated reads/writes + audit log (Mongo/in-memory) |
| `rag_engine.py` | Vector retrieval (Chroma/in-memory lexical) |
| `hallucination_guard.py` | Groundedness scoring + escalation decision |
| `models.py` | Shared dataclasses / enums / pipeline state |

### Layer 3 — Domain Agents (`agents/`)
Each agent receives an `AgentContext` (tenant + SDK services) and is ~one file.
Agents compute safety-critical structured fields deterministically from tools and
use the LLM only for grounded narrative.

`orchestrator.py` wires them into a LangGraph `StateGraph` with conditional edges.

## Key design decisions

- **Deterministic safety fields.** Severity comes from ISO 10816 zone math, not
  the LLM. Confidence is evidence-derived. The LLM cannot invent a safety number.
- **Isolation by construction.** `tenant_id` is injected last into every query so
  a caller can't override it; there is no un-scoped data path.
- **Graceful degradation.** Missing credentials select in-memory/mock backends
  with identical interfaces, so tests and demos run anywhere and the same code
  scales to real services in production.
- **Measure what you ship.** The groundedness eval reuses the production guard's
  scorer, so there's no drift between the metric and the guardrail.

## Request lifecycle

```
POST /v1/analyze
  → AuthMiddleware (tenant) → RateLimitMiddleware
  → analyze route → Pipeline.run(machine_id, tenant, sensor_data)
      → analyzer  (ISO severity, cite standard)
      → [NORMAL? END]
      → diagnostics (retrieve failures, confidence)
      → [conf<0.6? escalate]
      → guard (groundedness ≥0.8?)
      → [ungrounded? escalate]
      → recommender (action plan, urgency)
  → serialize → AnalyzeResponse
```
