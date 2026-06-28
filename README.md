# 🏭 Factory Health Agent

**A GenAI _infrastructure platform_ for industrial machine diagnostics — not a demo agent.**

The core is a reusable **Agent SDK** that any product team can build a new
industrial AI agent on in under an hour. Three domain agents (Analyzer →
Diagnostics → Recommender) are thin plugins on top of it, orchestrated with
LangGraph and gated by production guardrails.

> **Runs with zero setup.** Every external dependency — Azure OpenAI, Gemini,
> MongoDB, Redis, ChromaDB, LangSmith — has an in-memory / mock fallback. Clone,
> install, run. Add credentials to upgrade each component to the real thing.

**▶ Live demo: https://factory-health-agent.onrender.com** — pick a machine, set
sensor readings, and watch the multi-agent pipeline diagnose it live (powered by Gemini).

[![CI](https://github.com/yazandahood8/factory-health-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/yazandahood8/factory-health-agent/actions)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## Architecture — 3 layers

```
LAYER 1  API Gateway        FastAPI · JWT auth · tenant context · rate limiting
                                            │
LAYER 2  GenAI SDK          LLM Router (Azure→Gemini→Mock) · Tenant-Isolated Store
                            RAG Engine · Hallucination Guard · Budget Manager · Eval
                                            │
LAYER 3  Domain Agents      Analyzer · Diagnostics · Recommender  (LangGraph)
                                            │
DATA                        MongoDB (multi-tenant) · ChromaDB · Redis
```

The pipeline encodes *when not to trust the AI*:

```
analyzer ─ NORMAL ─────────────────────────────────► END
        └ WARNING/CRITICAL ─► diagnostics
                                 ├ confidence < 0.6 ───────────► escalate ► END
                                 └ confidence ≥ 0.6 ─► guard
                                                        ├ ungrounded ─► escalate ► END
                                                        └ grounded ──► recommender ► END
```

---

## Why this is senior-level

1. **LLM Router with fallback** — providers implement a tiny protocol; the
   router fails over Azure → Gemini → Mock and enforces a per-tenant budget
   *before* spending a token. No vendor lock-in, no surprise bills.
2. **Tenant isolation as infrastructure** — every query goes through
   `TenantIsolatedStore`, which injects `tenant_id` into the filter and audit-logs
   the access. Cross-tenant reads are impossible *by construction*, not convention.
3. **Hallucination guard** — safety-critical numbers (severity, thresholds) are
   computed deterministically; the LLM only writes narrative. If that narrative
   isn't grounded in retrieved sources (≥0.8), the system escalates to a human
   instead of returning.
4. **Evaluation framework** — 5 metrics (diagnosis accuracy, severity accuracy,
   groundedness, p95 latency, cost) with SLA thresholds; `run_eval` exits non-zero
   on regression, so it's a CI quality gate, and uploads to LangSmith when configured.
5. **Reusable SDK** — a new agent is one file of domain logic: it's handed an
   `AgentContext` (LLM, store, RAG, guard) and never builds infrastructure itself.

---

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate     # (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# Run the test suite (fully offline)
pytest -q

# Run the evaluation suite / CI gate
python -m evaluation.run_eval

# Start the API
uvicorn api.main:app --reload --port 8080
# open http://localhost:8080/docs
```

Or with Docker (brings up api + mongo + redis + chromadb):

```bash
cp .env.example .env
docker compose up --build
```

---

## Deploy

A [`render.yaml`](render.yaml) blueprint is included. To get a live URL:

1. Push to GitHub (done).
2. On [Render](https://render.com): **New → Blueprint**, point it at this repo.
3. Set `GEMINI_API_KEY` in the dashboard (marked `sync:false`); `JWT_SECRET` is
   auto-generated. Render builds the `Dockerfile` and exposes `/v1/health`.

With no managed databases attached the app uses its in-memory fallbacks
(ephemeral but fully functional). Attach Render Mongo/Redis and set `MONGODB_URI`
/ `REDIS_URL` for durable state. The `Dockerfile` honors `$PORT`, so the same
image runs on Railway / Fly / Cloud Run unchanged.

### Observability (LangSmith)

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` (locally in `.env` or in
your host's env). `sdk/observability.py` propagates these so every LLM and graph
call is traced; `run_eval` also uploads the eval dataset.

---

## Web UI

A self-contained dashboard is served at `/` (one `api/static/index.html`, no
build step, same-origin — so it deploys unchanged via the Dockerfile). It loads a
machine catalog, offers Healthy/Warning/Critical sensor presets, and streams the
live pipeline trace before rendering severity → diagnosis → action plan.

So the public demo works without credentials, an open **`GET /v1/demo-token`**
endpoint issues a short-lived, budget-capped, IP-rate-limited JWT that the UI uses
automatically. It *exercises* the real auth system rather than bypassing it —
every other `/v1` endpoint stays locked.

## API

| Method | Path                  | Description                          |
|--------|-----------------------|--------------------------------------|
| POST   | `/v1/analyze`         | Full agent pipeline                  |
| POST   | `/v1/analyze/stream`  | Same, streamed via SSE               |
| GET    | `/v1/machines`        | Tenant machine catalog (authed)      |
| GET    | `/v1/demo-token`      | Short-lived demo JWT (open)          |
| GET    | `/v1/health`          | Component health (open)              |
| GET    | `/v1/metrics`         | Per-tenant spend / latency / errors  |

All `/v1` data endpoints require `Authorization: Bearer <jwt>`. Mint a dev token:

```python
from api.security import mint_token
print(mint_token("acme", budget_usd=100))
```

```bash
curl -X POST localhost:8080/v1/analyze \
  -H "Authorization: Bearer $TOKEN" -H "X-Tenant-ID: acme" \
  -H "Content-Type: application/json" \
  -d '{"machine_id":"pump_001","sensor_data":{"vibration_mm_s":6.5,"temperature_c":95}}'
```

---

## Evaluation report (example)

```
Factory Health Agent — Eval Report
====================================================
Test Cases:   12
LLM provider: mock
Metric                       Score       SLA  Status
----------------------------------------------------
Diagnosis Accuracy          100.0%       85%  PASS
Severity Accuracy           100.0%       90%  PASS
Groundedness                 1.00      0.80   PASS
P95 Latency (ms)               12      3000   PASS
Avg Cost (USD)            $0.0000    $0.0500  PASS
====================================================
RESULT: ALL METRICS PASS
```

---

## Project layout

```
sdk/         LLM router · tenant store · RAG · guard · budget · eval primitives
agents/      analyzer · diagnostics · recommender · orchestrator (LangGraph)
api/         FastAPI app · auth/tenant/rate-limit middleware · routes
data/        sample data · bootstrap · seed scripts
evaluation/  evaluators · test_cases · run_eval (CI gate)
tests/       tenant isolation (critical) · agents · sdk · api
docs/        architecture · sdk guide
```

See [docs/architecture.md](docs/architecture.md) and the
[SDK guide](docs/sdk_guide.md) (build a new agent in ~1 hour).

---

## Tech stack

FastAPI · LangGraph · LangChain · ChromaDB · MongoDB · Redis · Pydantic ·
PyJWT · pytest · Azure OpenAI / Gemini · LangSmith · Docker

---

*Built by Yazan Dawud — Senior Full-Stack & AI Engineer · github.com/yazandahood8*
