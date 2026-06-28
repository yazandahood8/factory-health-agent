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
from sdk.trace import TraceRecorder

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
        rec: TraceRecorder = state["recorder"]
        rec.set_agent("analyzer")
        rec.add("agent", "analyzer", phase="start", input=state.get("sensor_data"))
        ctx = services.instrumented_context_for(state["tenant"], rec)
        report, resp = AnalyzerAgent(ctx).run(state["machine_id"], state.get("sensor_data"))
        rec.add("agent", "analyzer", phase="end",
                severity=report.severity.value, confidence=report.confidence)
        return {
            "anomaly_report": report,
            "total_tokens": state.get("total_tokens", 0) + resp.tokens,
            "messages": state.get("messages", []) + [f"analyzer: {report.severity.value}"],
        }

    def diagnostics_node(state: PipelineState) -> dict:
        rec: TraceRecorder = state["recorder"]
        rec.set_agent("diagnostics")
        rec.add("agent", "diagnostics", phase="start")
        ctx = services.instrumented_context_for(state["tenant"], rec)
        mt = _machine_type(ctx, state["machine_id"])
        diagnosis, resp = DiagnosticsAgent(ctx).run(state["anomaly_report"], mt)
        rec.add("agent", "diagnostics", phase="end",
                root_cause=diagnosis.root_cause, confidence=diagnosis.confidence)
        return {
            "diagnosis": diagnosis,
            "diag_response": resp,
            "machine_type": mt,
            "total_tokens": state.get("total_tokens", 0) + resp.tokens,
            "messages": state.get("messages", [])
            + [f"diagnostics: {diagnosis.root_cause} ({diagnosis.confidence})"],
        }

    def guard_node(state: PipelineState) -> dict:
        # Reuses the diagnostics response (no re-run) and scores groundedness
        # against the evidence it cited.
        rec: TraceRecorder = state["recorder"]
        rec.set_agent("guard")
        ctx = services.instrumented_context_for(state["tenant"], rec)
        mt = state.get("machine_type") or _machine_type(ctx, state["machine_id"])
        resp = state["diag_response"]
        context = _context_from_sources(ctx, state, mt)
        validated = services.guard.validate(resp, context)
        rec.add("guard", "groundedness", score=round(validated.groundedness, 3),
                threshold=services.guard.threshold, action=validated.action)
        return {
            "validated_response": validated,
            "messages": state.get("messages", []) + [f"guard: {validated.action}"],
        }

    def recommender_node(state: PipelineState) -> dict:
        rec: TraceRecorder = state["recorder"]
        rec.set_agent("recommender")
        rec.add("agent", "recommender", phase="start")
        ctx = services.instrumented_context_for(state["tenant"], rec)
        severity = state["anomaly_report"].severity
        plan, resp = RecommenderAgent(ctx).run(state["diagnosis"], severity)
        rec.add("agent", "recommender", phase="end", urgency=plan.urgency.value)
        return {
            "action_plan": plan,
            "total_tokens": state.get("total_tokens", 0) + resp.tokens,
            "total_cost_usd": services.llm.budget_manager.current_spend(state["tenant"]),
            "messages": state.get("messages", []) + [f"recommender: {plan.urgency.value}"],
        }

    def escalate_node(state: PipelineState) -> dict:
        rec: TraceRecorder = state["recorder"]
        rec.set_agent("escalate")
        rec.add("agent", "escalate", phase="end", action="ESCALATE_TO_HUMAN")
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
def _route(state: PipelineState, at: str, decision: str, reason: str) -> str:
    rec = state.get("recorder")
    if rec is not None:
        rec.set_agent("router")
        rec.add("route", at, decision=decision, reason=reason)
    return decision


def route_after_analysis(state: PipelineState) -> str:
    sev = state["anomaly_report"].severity
    decision = "normal" if sev is Severity.NORMAL else "investigate"
    return _route(state, "after_analysis", decision, f"severity={sev.value}")


def route_after_diagnosis(state: PipelineState) -> str:
    d = state["diagnosis"]
    decision = "escalate" if (d.escalate or d.confidence < CONFIDENCE_FLOOR) else "guard"
    return _route(state, "after_diagnosis", decision,
                  f"confidence={d.confidence} vs floor {CONFIDENCE_FLOOR}")


def route_after_guard(state: PipelineState) -> str:
    v = state["validated_response"]
    decision = "recommend" if v.action == "RETURN" else "escalate"
    return _route(state, "after_guard", decision, f"groundedness={round(v.groundedness, 3)}")


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
        recorder = TraceRecorder()
        initial: PipelineState = {
            "machine_id": machine_id,
            "sensor_data": sensor_data or {},
            "tenant": tenant,
            "escalated": False,
            "trace_id": str(uuid.uuid4()),
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "messages": [],
            "recorder": recorder,
        }
        final = self.graph.invoke(initial)
        # Ensure the (in-place mutated) recorder is present on the returned state.
        final["recorder"] = recorder
        return final
