# 🏭 Factory Health Agent — Senior Engineer Edition
## Production-Grade Industrial AI Infrastructure Platform
### Designed to prove Senior-level engineering for Augury's GenAI Infrastructure team

---

## 🎯 What Makes This Senior-Level

> A junior builds an agent that works.
> A senior builds the infrastructure that lets a team build 10 agents — reliably, securely, and at scale.

This project is NOT a demo. It's a **GenAI infrastructure platform** that:
- Any product team can plug into to ship an AI feature
- Handles multi-tenant security from day one
- Has evaluation, observability, fallback strategies baked in
- Is production-ready: tested, containerized, monitored

---

## 🏗️ Architecture — 3 Layers (Senior Thinking)

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: API Gateway                      │
│   FastAPI + Auth Middleware + Rate Limiting + Tenant Context │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│              LAYER 2: GenAI Infrastructure SDK               │
│                                                              │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ Agent Runner │  │  RAG Engine │  │  Eval Framework  │   │
│  │  (LangGraph) │  │ (ChromaDB)  │  │  (LangSmith)     │   │
│  └──────┬───────┘  └──────┬──────┘  └────────┬─────────┘   │
│         │                 │                   │             │
│  ┌──────▼─────────────────▼───────────────────▼─────────┐  │
│  │              Shared Infrastructure                     │  │
│  │  - Tenant Isolation Layer                              │  │
│  │  - LLM Router (Azure OpenAI / Gemini fallback)        │  │
│  │  - Cost + Token Budget Manager                        │  │
│  │  - Hallucination Guard                                │  │
│  │  - Retry + Fallback Strategies                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    LAYER 3: Domain Agents                    │
│                                                              │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │  Analyzer   │  │  Diagnostics │  │   Recommender    │  │
│   │   Agent     │  │    Agent     │  │     Agent        │  │
│   └─────────────┘  └──────────────┘  └──────────────────┘  │
│                                                              │
│   Each agent is a PLUGIN — uses the SDK, not reimplements it │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      Data Layer                              │
│   MongoDB (multi-tenant) │ ChromaDB │ Redis Cache           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 The 5 Senior Engineering Decisions

### 1. LLM Router with Fallback Strategy
```python
class LLMRouter:
    """
    Senior engineers don't hardcode one LLM.
    They route intelligently and handle failures.
    """
    def get_llm(self, task_type: str, tenant: Tenant) -> BaseLLM:
        # Primary: Azure OpenAI (lower latency for EU tenants)
        # Fallback: Gemini (when Azure rate-limited)
        # Budget check: don't overspend per tenant
        if self.budget_manager.is_over_limit(tenant):
            raise BudgetExceededException()
        try:
            return self._azure_openai(task_type)
        except RateLimitError:
            return self._gemini_fallback(task_type)
```

**Why this is Senior:** Junior hardcodes `ChatOpenAI()`. Senior thinks about availability, cost, and vendor lock-in.

---

### 2. Tenant Isolation — Zero Trust Data Access
```python
class TenantIsolatedStore:
    """
    Every single DB query MUST include tenant_id.
    No exceptions. No bypasses. Not optional.
    """
    def query(self, collection: str, query: dict, tenant: Tenant) -> list:
        # Inject tenant_id into EVERY query — no way to bypass
        secure_query = {**query, "tenant_id": tenant.id}
        
        # Audit log every data access
        self.audit_log.record(
            tenant_id=tenant.id,
            collection=collection,
            query_hash=hash(str(secure_query))
        )
        return self.db[collection].find(secure_query)
```

**Why this is Senior:** Augury serves industrial customers globally. A data leak between tenants = catastrophic. This is their #1 concern.

---

### 3. Hallucination Guard — Industrial Context
```python
class HallucinationGuard:
    """
    In a factory, a hallucinated recommendation can
    cause a safety incident or equipment damage.
    Senior engineers build guardrails.
    """
    def validate(self, response: AgentResponse, context: RetrievedContext) -> ValidatedResponse:
        # Check: is every claim grounded in retrieved documents?
        groundedness_score = self._check_groundedness(
            claims=response.recommendations,
            sources=context.documents
        )
        
        if groundedness_score < 0.8:
            # Don't return hallucinated response
            # Escalate to human expert instead
            return ValidatedResponse(
                action="ESCALATE_TO_HUMAN",
                reason=f"Low groundedness: {groundedness_score}",
                original_response=response
            )
        
        return ValidatedResponse(action="RETURN", response=response)
```

**Why this is Senior:** Junior returns whatever the LLM says. Senior knows hallucinations in industrial AI = real danger.

---

### 4. Evaluation Framework — Not Just Testing
```python
class EvalFramework:
    """
    Senior engineers measure output quality continuously,
    not just at deployment time.
    """
    evaluators = [
        DiagnosisAccuracyEvaluator(),    # vs expert ground truth
        RecommendationQualityEvaluator(), # expert review 1-5
        GroundednessEvaluator(),          # RAG citation coverage
        LatencyEvaluator(sla_ms=3000),    # p95 < 3s
        TokenCostEvaluator(),             # cost per request
    ]
    
    def run_eval_suite(self, test_cases: list) -> EvalReport:
        # Run all evaluators
        # Track regression over time via LangSmith
        # Alert if accuracy drops below threshold
        pass
```

**Why this is Senior:** Augury's JD explicitly asks for "evaluation, tracing, monitoring, and quality-control mechanisms." This is that — production-grade.

---

### 5. Agent SDK — Reusable Infrastructure
```python
class AgentSDK:
    """
    The core SDK that any product team can use
    to build a new agent in < 1 hour.
    This is what Augury's JD means by:
    'Build reusable infrastructure, SDKs, and
    internal frameworks for AI-powered product capabilities.'
    """
    def create_agent(
        self,
        name: str,
        tools: list[BaseTool],
        system_prompt: str,
        tenant: Tenant,
        eval_dataset: str | None = None
    ) -> Agent:
        return Agent(
            llm=self.llm_router.get_llm("reasoning", tenant),
            tools=tools,
            memory=self._build_memory(tenant),
            guardrails=HallucinationGuard(),
            tracing=LangSmithTracer(project=name),
            evaluation=self.eval_framework if eval_dataset else None,
        )
```

**Why this is Senior:** Any new agent = 10 lines of code. The infrastructure handles everything else.

---

## 🤖 The 3 Domain Agents (Built on the SDK)

### Agent 1: Analyzer Agent
```
Input:  Sensor data (vibration, temperature, pressure, RPM)
RAG:    ISO 10816 vibration standards + machine specs
Output: Anomaly report with severity score
Tools:  - query_sensor_history(machine_id, hours=72)
        - get_machine_specs(machine_id)
        - retrieve_vibration_standards(machine_type)
```

### Agent 2: Diagnostics Agent
```
Input:  Anomaly report from Analyzer
RAG:    Historical failure cases + machine manuals
Output: Root cause diagnosis with confidence (0.0-1.0)
Tools:  - search_failure_history(symptoms)
        - query_machine_manual(machine_id, chapter)
        - calculate_severity(anomaly_data)
```

### Agent 3: Recommender Agent
```
Input:  Diagnosis from Diagnostics Agent
RAG:    Maintenance procedures + spare parts catalog
Output: Prioritized action plan with estimated downtime
Tools:  - get_maintenance_procedures(diagnosis)
        - check_spare_parts_availability(parts_needed)
        - estimate_downtime(repair_type)
        - notify_maintenance_team(urgency, actions)
```

---

## 📡 API Design

```
POST   /v1/analyze          → Full agent pipeline
POST   /v1/analyze/stream   → Streaming response (SSE)
GET    /v1/machines/{id}    → Machine status
GET    /v1/reports/{id}     → Historical analysis
GET    /v1/health           → System health
GET    /v1/metrics          → Performance metrics

Headers (all endpoints):
  X-Tenant-ID: {tenant_id}   ← Required
  Authorization: Bearer {jwt} ← Required
```

---

## 📁 Project Structure

```
factory-health-agent/
│
├── README.md
├── docker-compose.yml
├── requirements.txt
├── .env.example
│
├── sdk/                        ← THE CORE SDK (Senior work)
│   ├── agent_sdk.py            # Main SDK entrypoint
│   ├── llm_router.py           # LLM routing + fallback
│   ├── tenant_store.py         # Tenant-isolated data access
│   ├── hallucination_guard.py  # Guardrails
│   ├── rag_engine.py           # RAG abstraction
│   └── eval_framework.py       # Evaluation framework
│
├── agents/                     ← Domain agents (use SDK)
│   ├── orchestrator.py         # LangGraph graph
│   ├── analyzer.py             # Agent 1
│   ├── diagnostics.py          # Agent 2
│   └── recommender.py          # Agent 3
│
├── api/
│   ├── main.py                 # FastAPI app
│   ├── middleware/
│   │   ├── auth.py             # JWT auth
│   │   ├── tenant.py           # Tenant context injection
│   │   └── rate_limit.py       # Rate limiting
│   └── routes/
│       ├── analyze.py
│       └── health.py
│
├── data/
│   ├── seed_mongodb.py         # Machine data
│   ├── seed_chromadb.py        # Manuals + docs embeddings
│   └── sample_data/
│       ├── machines.json
│       ├── sensor_logs.json
│       ├── failure_cases.json
│       └── manuals/
│
├── evaluation/
│   ├── evaluators/
│   │   ├── diagnosis_accuracy.py
│   │   ├── groundedness.py
│   │   ├── latency.py
│   │   └── cost.py
│   ├── run_eval.py
│   └── test_cases.json
│
├── tests/
│   ├── test_sdk.py
│   ├── test_agents.py
│   ├── test_api.py
│   └── test_tenant_isolation.py  ← Critical security tests
│
└── docs/
    ├── architecture.md
    ├── sdk_guide.md             # How to build a new agent
    └── api_reference.md
```

---

## 🔄 LangGraph Orchestration

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class PipelineState(TypedDict):
    # Input
    machine_id: str
    sensor_data: dict
    tenant: Tenant
    
    # Intermediate
    anomaly_report: dict | None
    diagnosis: dict | None
    
    # Output
    final_report: dict | None
    escalated: bool
    trace_id: str

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)
    
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("diagnostics", diagnostics_node)
    graph.add_node("recommender", recommender_node)
    graph.add_node("escalate", escalate_to_human_node)
    
    graph.add_edge("analyzer", "diagnostics")
    graph.add_conditional_edges(
        "diagnostics",
        route_by_confidence,
        {
            "high_confidence": "recommender",
            "low_confidence": "escalate",  # ← Senior: knows when NOT to trust AI
        }
    )
    graph.add_edge("recommender", END)
    graph.add_edge("escalate", END)
    
    graph.set_entry_point("analyzer")
    return graph.compile()
```

---

## 📊 Evaluation Report (What LangSmith Shows)

```
Factory Health Agent — Eval Report
=====================================
Test Cases:     50
Date:           2026-06-30

Metric                  Score    SLA      Status
─────────────────────────────────────────────────
Diagnosis Accuracy      91.2%    > 85%    ✅ PASS
Groundedness Score      0.89     > 0.80   ✅ PASS
P95 Latency             2,340ms  < 3,000  ✅ PASS
Cost per Request        $0.018   < $0.05  ✅ PASS
Hallucination Rate      3.2%     < 5%     ✅ PASS
Tenant Isolation Tests  100%     100%     ✅ PASS

Regression vs Last Run:
  Accuracy: +2.1% ↑
  Latency:  -180ms ↑
```

---

## 💬 Senior-Level Interview Answers

### "Tell me about a project you built"
> "I built Factory Health Agent — not just an agent, but an infrastructure platform for GenAI. The core is an Agent SDK that any team can use to build a new industrial AI agent in under an hour. I built LLM routing with Azure OpenAI and Gemini fallback, tenant-isolated data access where every query enforces tenant_id at the infrastructure level, and a hallucination guard that escalates to human experts when confidence is low — because in industrial settings, a wrong recommendation can cause real damage. Everything is traced through LangSmith with 5 evaluators running continuously."

### "How do you handle hallucinations in production?"
> "I built a groundedness evaluator that checks every agent response against its retrieved context. If fewer than 80% of claims are grounded in retrieved documents, the system doesn't return the response — it escalates to a human expert instead. In industrial AI, the cost of a wrong recommendation is too high to risk."

### "How do you think about multi-tenant security?"
> "I treat tenant isolation as infrastructure, not application logic. Every data access goes through a TenantIsolatedStore that injects tenant_id at the query level — not optional, not bypassable. I also audit-log every data access for compliance. This is non-negotiable in enterprise B2B."

### "What's your experience with LangGraph?"
> "I used LangGraph to build the orchestration layer. I defined a StateGraph with conditional edges — for example, if the Diagnostics Agent returns low confidence, it routes to a human escalation node instead of the Recommender. This conditional routing is exactly where LangGraph shines over simple chains."

---

## 🏆 What This Proves

| Augury Needs | This Project Proves |
|-------------|---------------------|
| GenAI Infrastructure builder | Agent SDK + LLM Router |
| Multi-agent orchestration | LangGraph with conditional routing |
| Production-grade systems | Docker + tests + monitoring |
| RAG expert | ChromaDB + groundedness eval |
| Security-minded | Zero-trust tenant isolation |
| LangSmith | 5-metric eval suite |
| System design | 3-layer architecture |
| Senior engineer | Infrastructure others use |
| Industrial domain understanding | Machine health context |
| Azure OpenAI + Gemini | LLM Router supports both |

---

## 📅 Build Plan

| # | Task | Time |
|---|------|------|
| 1 | Project setup + Docker | 30 min |
| 2 | SDK core (LLM Router + Tenant Store) | 90 min |
| 3 | RAG Engine + ChromaDB seeding | 60 min |
| 4 | Hallucination Guard | 45 min |
| 5 | Analyzer Agent | 45 min |
| 6 | Diagnostics Agent | 45 min |
| 7 | Recommender Agent | 45 min |
| 8 | LangGraph orchestration | 45 min |
| 9 | FastAPI + Auth Middleware | 45 min |
| 10 | Evaluation Framework + LangSmith | 45 min |
| 11 | Tests (especially tenant isolation) | 30 min |
| 12 | README + Architecture docs | 30 min |
| **Total** | | **~9 hours** |

---

*Built by Yazan Dawud — Senior Full-Stack & AI Engineer*
*GitHub: github.com/yazandahood8*

---

## 🏃 Sprint Plan — Production-Ready in 1 Day

> **Total time:** ~9 hours
> **Goal:** Senior-level GenAI infrastructure platform on GitHub
> **Strategy:** Build SDK first, then agents, then API, then eval

---

## Sprint 0 — Setup (30 min)
*Everything working before writing business logic*

| # | Task | Details | Done |
|---|------|---------|------|
| 0.1 | Create GitHub repo | `factory-health-agent`, MIT license, .gitignore Python | ☐ |
| 0.2 | Project structure | Create all folders from the structure above | ☐ |
| 0.3 | `requirements.txt` | langchain, langgraph, langsmith, chromadb, pymongo, fastapi, uvicorn, pydantic, python-dotenv, redis, pytest | ☐ |
| 0.4 | `.env.example` | AZURE_OPENAI_KEY, OPENAI_API_VERSION, GEMINI_API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, MONGODB_URI, REDIS_URL | ☐ |
| 0.5 | `docker-compose.yml` | Services: api, mongo, chromadb, redis | ☐ |
| 0.6 | Verify containers run | `docker compose up` — all green | ☐ |

**Checkpoint:** `docker compose up` works, all services healthy ✅

---

## Sprint 1 — SDK Core (90 min)
*The infrastructure that makes this Senior-level*

### Task 1.1 — Tenant Model (15 min)
```
File: sdk/models.py
- Tenant dataclass (id, name, llm_budget_usd, tier)
- AgentState TypedDict
- AgentResponse dataclass
- ValidatedResponse dataclass
```

### Task 1.2 — LLM Router (25 min)
```
File: sdk/llm_router.py
- BudgetManager class (track spend per tenant in Redis)
- LLMRouter class:
  - get_llm(task_type, tenant) → BaseLLM
  - _azure_openai(task_type) → AzureChatOpenAI
  - _gemini_fallback(task_type) → ChatGoogleGenerativeAI
  - Catches RateLimitError → switches to fallback
  - Raises BudgetExceededException if over limit
```

### Task 1.3 — Tenant Isolated Store (25 min)
```
File: sdk/tenant_store.py
- TenantIsolatedStore class:
  - __init__(mongo_client, audit_logger)
  - query(collection, query, tenant) → list
    - ALWAYS injects tenant_id into query
    - ALWAYS writes to audit_log
  - insert(collection, doc, tenant) → str
  - AuditLogger class (writes to MongoDB audit collection)
- Unit test: verify tenant_id always present in query
```

### Task 1.4 — Hallucination Guard (25 min)
```
File: sdk/hallucination_guard.py
- GroundednessChecker:
  - check(response_text, source_docs) → float (0.0-1.0)
  - Uses sentence similarity vs retrieved docs
- HallucinationGuard:
  - validate(response, context) → ValidatedResponse
  - If score < 0.8 → action="ESCALATE_TO_HUMAN"
  - If score >= 0.8 → action="RETURN"
```

**Checkpoint:** All SDK classes importable, unit tests pass ✅

---

## Sprint 2 — RAG Engine (60 min)
*The knowledge retrieval layer*

### Task 2.1 — Sample Data Creation (20 min)
```
File: data/sample_data/machines.json
- 5 machines: pump_001, motor_002, compressor_003, fan_004, gearbox_005
- Each: machine_type, manufacturer, install_date, normal_ranges

File: data/sample_data/sensor_logs.json
- 100 sensor readings per machine (last 7 days)
- Mix of normal + anomalous readings
- Fields: machine_id, timestamp, vibration, temperature, pressure, rpm

File: data/sample_data/failure_cases.json
- 20 historical failure cases
- Fields: machine_type, symptoms, root_cause, resolution, downtime_hours
```

### Task 2.2 — Seed MongoDB (15 min)
```
File: data/seed_mongodb.py
- Insert machines.json into machines collection
- Insert sensor_logs.json into sensor_logs collection
- Insert failure_cases.json into failure_cases collection
- Print confirmation
```

### Task 2.3 — RAG Engine + ChromaDB Seeding (25 min)
```
File: sdk/rag_engine.py
- RAGEngine class:
  - __init__(chroma_client, embeddings)
  - index_documents(docs, collection_name)
  - retrieve(query, collection_name, k=5) → list[Document]
  - retrieve_with_sources(query, collection_name) → RetrievalResult

File: data/seed_chromadb.py
- Index machine manuals (text excerpts about common failures)
- Index ISO 10816 vibration standards
- Index maintenance procedures
- Index failure cases from JSON
```

**Checkpoint:** RAG retrieval returns relevant docs for "bearing vibration" query ✅

---

## Sprint 3 — The 3 Agents (135 min — 45 min each)

### Task 3.1 — Analyzer Agent (45 min)
```
File: agents/analyzer.py

Tools:
  - query_sensor_history(machine_id, hours) → list[SensorReading]
  - get_machine_specs(machine_id) → MachineSpecs
  - retrieve_vibration_standards(machine_type) → list[Document]

System Prompt:
  "You are an industrial machine analyzer. 
   Analyze sensor data against ISO 10816 standards.
   Always cite which standard you're referencing.
   If data is outside normal range, classify severity:
   NORMAL / WARNING / CRITICAL"

Output: AnomalyReport(machine_id, severity, anomaly_type, details, confidence)
```

### Task 3.2 — Diagnostics Agent (45 min)
```
File: agents/diagnostics.py

Tools:
  - search_failure_history(machine_type, symptoms) → list[FailureCase]
  - query_machine_manual(machine_id, topic) → list[Document]
  - calculate_rul(vibration_trend) → int  # Remaining Useful Life in days

System Prompt:
  "You are an expert reliability engineer.
   Given anomaly data, diagnose the root cause.
   You must cite specific failure cases from the database.
   Provide confidence score 0.0-1.0 based on evidence quality.
   If confidence < 0.6, recommend escalation to human expert."

Output: Diagnosis(root_cause, confidence, evidence, escalate_flag)
```

### Task 3.3 — Recommender Agent (45 min)
```
File: agents/recommender.py

Tools:
  - get_maintenance_procedures(diagnosis) → list[Procedure]
  - check_parts_availability(parts) → dict
  - estimate_downtime(repair_type) → int  # hours
  - classify_urgency(confidence, severity) → str  # LOW/MEDIUM/HIGH/CRITICAL

System Prompt:
  "You are a maintenance planning expert.
   Generate a prioritized action plan.
   Always specify: what to do, when, estimated downtime, parts needed.
   If urgency is CRITICAL, first action must be 'reduce load immediately'."

Output: ActionPlan(actions, urgency, estimated_downtime, parts_needed)
```

**Checkpoint:** Each agent returns correct output on test sensor data ✅

---

## Sprint 4 — LangGraph Orchestration (45 min)

### Task 4.1 — State Definition (10 min)
```
File: agents/orchestrator.py

class PipelineState(TypedDict):
    machine_id: str
    sensor_data: dict
    tenant: Tenant
    anomaly_report: AnomalyReport | None
    diagnosis: Diagnosis | None
    action_plan: ActionPlan | None
    validated_response: ValidatedResponse | None
    escalated: bool
    trace_id: str
    total_tokens: int
    total_cost_usd: float
```

### Task 4.2 — Node Functions (20 min)
```
- analyzer_node(state) → state update
- diagnostics_node(state) → state update
- recommender_node(state) → state update
- escalate_node(state) → state update (notify human team)
- guard_node(state) → state update (run HallucinationGuard)
```

### Task 4.3 — Graph Wiring + Conditional Routing (15 min)
```
Graph:
START → analyzer_node
     → diagnostics_node
     → [conditional]
         confidence >= 0.6 → guard_node → recommender_node → END
         confidence < 0.6  → escalate_node → END
         severity == NORMAL → END (skip diagnostics)
```

**Checkpoint:** Full pipeline runs end-to-end, trace visible in LangSmith ✅

---

## Sprint 5 — FastAPI + Middleware (45 min)

### Task 5.1 — Auth + Tenant Middleware (20 min)
```
File: api/middleware/auth.py
- JWT validation middleware
- Extract tenant_id from token claims
- Inject Tenant object into request state

File: api/middleware/rate_limit.py
- Redis-based rate limiting
- 100 requests/minute per tenant
- Returns 429 with retry-after header
```

### Task 5.2 — API Routes (15 min)
```
File: api/routes/analyze.py

POST /v1/analyze
  - Validate request (Pydantic)
  - Get tenant from request.state
  - Run LangGraph pipeline
  - Return AnalysisResponse

POST /v1/analyze/stream
  - Same as above but SSE streaming
  - Stream agent thoughts + final result
```

### Task 5.3 — Health + Metrics Endpoints (10 min)
```
GET /v1/health
  - Check MongoDB connection
  - Check ChromaDB connection
  - Check Redis connection
  - Return status of each

GET /v1/metrics
  - Requests today (per tenant)
  - Average latency
  - Error rate
  - Token usage + cost
```

**Checkpoint:** `curl POST /v1/analyze` returns valid response ✅

---

## Sprint 6 — LangSmith Evaluation (45 min)

### Task 6.1 — Test Cases (10 min)
```
File: evaluation/test_cases.json
[
  {
    "id": "tc_001",
    "input": {"machine_id": "pump_001", "vibration": 6.2, "temp": 95, "rpm": 1380},
    "expected_diagnosis": "Bearing outer race defect",
    "expected_urgency": "HIGH",
    "expected_confidence_min": 0.75
  },
  ... (20 test cases total)
]
```

### Task 6.2 — Evaluators (25 min)
```
File: evaluation/evaluators/diagnosis_accuracy.py
  - Compare predicted vs expected diagnosis (exact + fuzzy match)

File: evaluation/evaluators/groundedness.py
  - Check every claim is cited from retrieved docs
  - Score 0.0-1.0

File: evaluation/evaluators/latency.py
  - Measure p50, p95, p99
  - Fail if p95 > 3000ms

File: evaluation/evaluators/cost.py
  - Track tokens per request
  - Calculate USD cost
  - Alert if avg > $0.05
```

### Task 6.3 — Run Eval + LangSmith Upload (10 min)
```
File: evaluation/run_eval.py
- Load test cases
- Run each through pipeline
- Collect all evaluator scores
- Upload to LangSmith dataset
- Print summary report
- Exit 1 if any metric fails SLA
```

**Checkpoint:** Eval report shows all metrics green ✅

---

## Sprint 7 — Tests + Security (30 min)

### Task 7.1 — Tenant Isolation Tests (15 min) ← MOST IMPORTANT
```
File: tests/test_tenant_isolation.py

def test_query_always_includes_tenant_id():
    # Mock MongoDB, verify tenant_id in every query

def test_cross_tenant_data_not_accessible():
    # tenant_a cannot see tenant_b's machines

def test_audit_log_written_on_every_access():
    # Every data access creates audit record

def test_budget_exceeded_blocks_request():
    # Tenant over budget gets BudgetExceededException
```

### Task 7.2 — Agent Tests (15 min)
```
File: tests/test_agents.py

def test_analyzer_detects_high_vibration():
def test_diagnostics_escalates_on_low_confidence():
def test_recommender_critical_urgency_first_action():
def test_hallucination_guard_blocks_ungrounded_response():
def test_full_pipeline_end_to_end():
```

**Checkpoint:** All tests pass, `pytest --tb=short` shows 0 failures ✅

---

## Sprint 8 — README + GitHub Polish (30 min)

### Task 8.1 — README.md (20 min)
```
Sections:
1. What this is (1 paragraph)
2. Architecture diagram (ASCII from plan)
3. Why Senior-level (5 bullet points)
4. Quick start (5 commands to run)
5. API reference (request/response examples)
6. Evaluation results (screenshot or text table)
7. Key engineering decisions (link to docs/)
8. Tech stack table
```

### Task 8.2 — Final Polish (10 min)
```
- Add GitHub Actions CI (run tests on push)
- Add badges to README (tests passing, Python version)
- Tag v1.0.0 release
- Verify all env vars documented in .env.example
- Clean up any TODO comments
```

**Checkpoint:** Anyone can clone + run in < 5 minutes ✅

---

## ✅ Final Checklist Before Interview

```
Infrastructure
  ☐ Agent SDK importable and tested
  ☐ LLM Router with Azure + Gemini fallback
  ☐ Tenant isolation tests all pass
  ☐ Hallucination Guard working

Agents
  ☐ Analyzer returns AnomalyReport
  ☐ Diagnostics returns Diagnosis with confidence
  ☐ Recommender returns ActionPlan with urgency
  ☐ LangGraph conditional routing working

API
  ☐ POST /v1/analyze returns valid response
  ☐ Auth middleware working
  ☐ Rate limiting working

Evaluation
  ☐ All 5 evaluators running
  ☐ LangSmith traces visible
  ☐ Eval report shows metrics passing SLA

GitHub
  ☐ Clean README with architecture
  ☐ All tests passing
  ☐ Docker Compose works from scratch
  ☐ .env.example complete
```

---

## 💬 Final Answer for "Show me something you built"

> "I built factory-health-agent — a GenAI infrastructure platform for industrial machine diagnostics.
>
> The core is an Agent SDK that handles LLM routing between Azure OpenAI and Gemini with automatic fallback, tenant-isolated data access where tenant_id is enforced at the infrastructure level and cannot be bypassed, and a hallucination guard that escalates to human experts when response groundedness drops below 80%.
>
> Three specialized agents — Analyzer, Diagnostics, and Recommender — are built on this SDK using LangGraph for orchestration with conditional routing: low-confidence diagnoses automatically escalate to human review.
>
> The eval framework runs 5 metrics continuously through LangSmith: diagnosis accuracy, groundedness, p95 latency, cost per request, and hallucination rate. All metrics have SLA thresholds and the CI pipeline fails if any drops below target.
>
> The entire system runs in Docker Compose with JWT auth, Redis rate limiting, and audit logging on every data access for compliance."

---

*Yazan Dawud | Senior Full-Stack & AI Engineer*
*github.com/yazandahood8/factory-health-agent*
