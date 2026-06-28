"""Recommender Agent — diagnosis → prioritized action plan.

Turns a root cause into concrete, ordered maintenance actions with parts
availability, downtime estimate, and an urgency class. For CRITICAL urgency the
first action is always load reduction — a hard-coded safety rule, not an LLM
suggestion.
"""
from __future__ import annotations

from typing import Any

from agents.base import AgentContext
from data.bootstrap import FAILURE_CASES, KB_PROCEDURES, SPARE_PARTS
from sdk.models import ActionPlan, AgentResponse, Diagnosis, Severity, Urgency


class RecommenderAgent:
    name = "recommender"
    SYSTEM = (
        "You are a maintenance planning expert. Produce a prioritized, actionable "
        "plan grounded in the retrieved procedures. Specify what, when, downtime, parts."
    )

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    # ---- tools -----------------------------------------------------------
    def get_maintenance_procedures(self, root_cause: str):
        return self.ctx.rag.retrieve_with_sources(root_cause, KB_PROCEDURES, k=3)

    def _matching_failure_case(self, root_cause: str) -> dict[str, Any]:
        cases = self.ctx.store.query(FAILURE_CASES, {}, self.ctx.tenant)
        rc = root_cause.lower()
        for c in cases:
            if c.get("root_cause", "").lower() in rc or rc in c.get("root_cause", "").lower():
                return c
        return {}

    def check_parts_availability(self, parts: list[str]) -> dict[str, dict]:
        stock = {p["part"]: p for p in self.ctx.store.query(SPARE_PARTS, {}, self.ctx.tenant)}
        out: dict[str, dict] = {}
        for part in parts:
            info = stock.get(part)
            out[part] = (
                {"in_stock": info["stock"], "lead_time_days": info["lead_time_days"]}
                if info
                else {"in_stock": 0, "lead_time_days": None}
            )
        return out

    def estimate_downtime(self, failure_case: dict[str, Any]) -> float:
        return float(failure_case.get("downtime_hours", 8))

    def classify_urgency(self, confidence: float, severity: Severity) -> Urgency:
        if severity is Severity.CRITICAL:
            return Urgency.CRITICAL
        if severity is Severity.WARNING:
            return Urgency.HIGH if confidence >= 0.75 else Urgency.MEDIUM
        return Urgency.LOW

    # ---- core ------------------------------------------------------------
    def recommend(self, diagnosis: Diagnosis, severity: Severity) -> ActionPlan:
        fc = self._matching_failure_case(diagnosis.root_cause)
        parts = fc.get("parts", [])
        downtime = self.estimate_downtime(fc)
        urgency = self.classify_urgency(diagnosis.confidence, severity)
        procedures = self.get_maintenance_procedures(diagnosis.root_cause)

        actions = self._build_actions(diagnosis, urgency, fc, procedures)
        # LLM narrative (grounded); structured plan above stays authoritative.
        self._narrate(diagnosis, urgency, procedures)

        return ActionPlan(
            actions=actions,
            urgency=urgency,
            estimated_downtime_hours=downtime,
            parts_needed=parts,
        )

    def _build_actions(self, diagnosis, urgency, failure_case, procedures) -> list[str]:
        actions: list[str] = []
        if urgency is Urgency.CRITICAL:
            actions.append("Reduce load immediately and prepare for controlled shutdown.")
        resolution = failure_case.get("resolution")
        if resolution:
            actions.append(f"Corrective action: {resolution}.")
        elif procedures.documents:
            actions.append(f"Follow procedure: {procedures.documents[0].source}.")
        if diagnosis.rul_days is not None:
            window = (
                "within 24 hours"
                if urgency is Urgency.CRITICAL
                else f"within {min(diagnosis.rul_days, 30)} days"
            )
            actions.append(f"Schedule maintenance {window}.")
        actions.append("Verify repair with a post-maintenance vibration check.")
        return actions

    def _narrate(self, diagnosis, urgency, procedures) -> str:
        context = " ".join(procedures.texts)
        prompt = (
            f"Root cause {diagnosis.root_cause}, urgency {urgency.value}. "
            f"Procedures: {context}\nWrite a 1-2 sentence grounded action summary."
        )
        return self.ctx.llm.complete(
            prompt, self.ctx.tenant, system=self.SYSTEM, task_type="reasoning"
        ).text

    def run(self, diagnosis: Diagnosis, severity: Severity) -> tuple[ActionPlan, AgentResponse]:
        plan = self.recommend(diagnosis, severity)
        procedures = self.get_maintenance_procedures(diagnosis.root_cause)
        text = (
            f"Urgency {plan.urgency.value}. Actions: " + " ".join(plan.actions) +
            f" Estimated downtime {plan.estimated_downtime_hours}h."
        )
        response = AgentResponse(
            text=text, structured=plan.__dict__, sources=procedures.sources
        )
        return plan, response
