# SDK Guide — build a new agent in ~1 hour

The SDK hands every agent an `AgentContext`:

```python
@dataclass
class AgentContext:
    tenant: Tenant                 # who we're acting for (isolation + budget)
    store: TenantIsolatedStore     # tenant-scoped data access + audit
    rag: RAGEngine                 # knowledge retrieval
    llm: LLMRouter                 # routed, budgeted, failover LLM calls
    guard: HallucinationGuard      # groundedness validation
```

You never construct infrastructure. You write domain logic.

## 1. Write the agent

```python
from agents.base import AgentContext
from sdk.models import AgentResponse

class VibrationTrendAgent:
    name = "vibration_trend"
    SYSTEM = "You summarize vibration trends. Cite the data you used."

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    # tools = tenant-scoped data / retrieval
    def recent(self, machine_id: str):
        rows = self.ctx.store.query("sensor_logs", {"machine_id": machine_id}, self.ctx.tenant)
        return sorted(rows, key=lambda r: r["timestamp"])[-50:]

    def run(self, machine_id: str):
        data = self.recent(machine_id)
        # Compute the numbers yourself (don't let the LLM invent them):
        slope = (data[-1]["vibration_mm_s"] - data[0]["vibration_mm_s"]) if data else 0.0
        docs = self.ctx.rag.retrieve_with_sources("vibration trend fault", "vibration_standards")
        text = self.ctx.llm.complete(
            f"Trend slope {slope:.2f} mm/s. Context: {' '.join(docs.texts)}",
            self.ctx.tenant, system=self.SYSTEM, task_type="analysis",
        ).text
        resp = AgentResponse(text=text, structured={"slope": slope}, sources=docs.sources)
        return resp
```

## 2. Guard the output (optional but recommended)

```python
validated = ctx.guard.validate(resp, docs.as_context())
if validated.action == "ESCALATE_TO_HUMAN":
    ...  # route to a human; don't return ungrounded output
```

## 3. Get a context

```python
from agents.orchestrator import build_services
from sdk.models import Tenant

services = build_services()
ctx = services.context_for(Tenant(id="acme", llm_budget_usd=10.0))
print(VibrationTrendAgent(ctx).run("pump_001").text)
```

## 4. (Optional) add it to the graph

In `agents/orchestrator.py`, add a node function and wire an edge — conditional
if the agent should sometimes escalate. The SDK gives you tracing, budget,
isolation, and guardrails for free.

## What you get for free

- **Budget + fail-over**: every `llm.complete` is gated and routed.
- **Isolation + audit**: every `store.query` is tenant-scoped and logged.
- **Offline dev**: no credentials needed; mock/in-memory backends kick in.
- **Eval**: drop expected outputs into `evaluation/test_cases.json`.
