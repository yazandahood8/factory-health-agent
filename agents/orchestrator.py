"""LangGraph orchestration of the three domain agents.

The graph encodes the senior decision of *when not to trust the AI*:

    analyzer
       └─ NORMAL ─────────────────────────────► END
       └─ WARNING/CRITICAL ─► diagnostics
                                  └─ conf < 0.6 ─────────► escalate ► END
                                  └─ conf ≥ 0.6 ─► guard
                                                     └─ ungrounded ─► escalate ► END
                                                     └─ grounded ──► recommender ► END

Services (LLM router, store, RAG, guard) are built once and shared; only the
tenant-scoped :class:`AgentContext` is per-request.
"""
from __future__ import annotations

import uuid
from typing import Optional

from langgraph.graph import END, StateGraph

from agents.analyzer import AnalyzerAgent
from agents.base import AgentContext
from agents.diagnostics import CONFIDENCE_FLOOR, DiagnosticsAgent
from agents.recommender import RecommenderAgent
from data.bootstrap import MACHINES, seed_rag, seed_store
from data.seed_mongodb import DEMO_TENANTS
from sdk.agent_sdk import AgentSDK, build_sdk
from sdk.config import Settings
from sdk.models import PipelineState, Severity, Tenant

# Re-export so existing callers (api, tests) need no changes.
Services = AgentSDK


def build_services(settings: Optional[Settings] = None, seed: bool = True) -> AgentSDK:
    sdk = build_sdk(settings)

    if seed:
        seed_rag(sdk.rag)
        for tenant in DEMO_TENANTS:
            if not sdk.store.query(MACHINES, {}, tenant):
                seed_store(sdk.store, tenant)
    return sdk


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def _machine_type(ctx: AgentContext, machine_id: str) -> str:
    specs = ctx.store.query(MACHINES, {"machine_id": machine_id}, ctx.tenant)
    return specs[0]["machine_type"] if specs else ""


def _make_nodes(services: Services):
    def analyzer_node(state: PipelineState) -> dict:
        ctx = services.context_for(state["tenant"])
        agent = AnalyzerAgent(ctx)
        report, resp = agent.run(state["machine_id"], state.get("sensor_data"))
        return {
            "anomaly_report": report,
            "total_tokens": state.get("total_tokens", 0) + resp.tokens,
            "messages": state.get("messages", []) + [f"analyzer: {report.severity.value}"],
        }

    def diagnostics_node(state: PipelineState) -> dict:
        ctx = services.context_for(state["tenant"])
        agent = DiagnosticsAgent(ctx)
        mt = _machine_type(ctx, state["machine_id"])
        diagnosis, resp = agent.run(state["anomaly_report"], mt)
        return {
            "diagnosis": diagnosis,
            "total_tokens": state.get("total_tokens", 0) + resp.tokens,
            "messages": state.get("messages", [])
            + [f"diagnostics: {diagnosis.root_cause} ({diagnosis.confidence})"],
        }

    def guard_node(state: PipelineState) -> dict:
        ctx = services.context_for(state["tenant"])
        agent = DiagnosticsAgent(ctx)
        mt = _machine_type(ctx, state["machine_id"])
        _, resp = agent.run(state["anomaly_report"], mt)
        context = _context_from_sources(ctx, state, mt)
        validated = services.guard.validate(resp, context)
        return {
            "validated_response": validated,
            "messages": state.get("messages", []) + [f"guard: {validated.action}"],
        }

    def recommender_node(state: PipelineState) -> dict:
        ctx = services.context_for(state["tenant"])
        agent = RecommenderAgent(ctx)
        severity = state["anomaly_report"].severity
        plan, resp = agent.run(state["diagnosis"], severity)
        return {
            "action_plan": plan,
            "total_tokens": state.get("total_tokens", 0) + resp.tokens,
            "total_cost_usd": services.llm.budget_manager.current_spend(state["tenant"]),
            "messages": state.get("messages", []) + [f"recommender: {plan.urgency.value}"],
        }

    def escalate_node(state: PipelineState) -> dict:
        return {
            "escalated": True,
            "total_cost_usd": services.llm.budget_manager.current_spend(state["tenant"]),
            "messages": state.get("messages", []) + ["escalate: routed to human expert"],
        }

    return analyzer_node, diagnostics_node, guard_node, recommender_node, escalate_node


def _context_from_sources(ctx: AgentContext, state: PipelineState, machine_type: str) -> RetrievedContext:
    """Re-retrieve the evidence the diagnosis used, for groundedness scoring."""
    report = state["anomaly_report"]
    res = ctx.rag.retrieve_with_sources(
        f"{machine_type} {report.anomaly_type} {report.details}", "maintenance_procedures", k=4
    )
    return res.as_context()


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def route_after_analysis(state: PipelineState) -> str:
    return "normal" if state["anomaly_report"].severity is Severity.NORMAL else "investigate"


def route_after_diagnosis(state: PipelineState) -> str:
    d = state["diagnosis"]
    return "escalate" if (d.escalate or d.confidence < CONFIDENCE_FLOOR) else "guard"


def route_after_guard(state: PipelineState) -> str:
    return "recommend" if state["validated_response"].action == "RETURN" else "escalate"


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_graph(services: Services):
    analyzer_node, diagnostics_node, guard_node, recommender_node, escalate_node = _make_nodes(
        services
    )
    g = StateGraph(PipelineState)
    g.add_node("analyzer", analyzer_node)
    g.add_node("diagnostics", diagnostics_node)
    g.add_node("guard", guard_node)
    g.add_node("recommender", recommender_node)
    g.add_node("escalate", escalate_node)

    g.set_entry_point("analyzer")
    g.add_conditional_edges(
        "analyzer", route_after_analysis, {"normal": END, "investigate": "diagnostics"}
    )
    g.add_conditional_edges(
        "diagnostics", route_after_diagnosis, {"escalate": "escalate", "guard": "guard"}
    )
    g.add_conditional_edges(
        "guard", route_after_guard, {"recommend": "recommender", "escalate": "escalate"}
    )
    g.add_edge("recommender", END)
    g.add_edge("escalate", END)
    return g.compile()


class Pipeline:
    """Convenience wrapper: build once, run per request."""

    def __init__(self, services: Optional[Services] = None) -> None:
        self.services = services or build_services()
        self.graph = build_graph(self.services)

    def run(self, machine_id: str, tenant: Tenant, sensor_data: Optional[dict] = None) -> PipelineState:
        initial: PipelineState = {
            "machine_id": machine_id,
            "sensor_data": sensor_data or {},
            "tenant": tenant,
            "escalated": False,
            "trace_id": str(uuid.uuid4()),
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "messages": [],
        }
        return self.graph.invoke(initial)
