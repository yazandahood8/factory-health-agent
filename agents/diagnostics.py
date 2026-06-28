"""Diagnostics Agent — anomaly report → root-cause diagnosis.

Retrieves similar historical failures and machine-manual passages, then forms
a diagnosis with an evidence-based confidence score. Crucially, it knows when
*not* to trust itself: below the confidence floor it flags for escalation.
"""
from __future__ import annotations

from typing import Any

from agents.base import AgentContext
from data.bootstrap import FAILURE_CASES, KB_MANUALS, KB_PROCEDURES
from sdk.models import AgentResponse, AnomalyReport, Diagnosis, Severity

CONFIDENCE_FLOOR = 0.6


class DiagnosticsAgent:
    name = "diagnostics"
    SYSTEM = (
        "You are an expert reliability engineer. Given anomaly data and retrieved "
        "failure cases, state the most likely root cause and cite the cases you used. "
        "Do not speculate beyond the evidence."
    )

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    # ---- tools -----------------------------------------------------------
    def search_failure_history(self, machine_type: str, symptoms: str):
        return self.ctx.rag.retrieve_with_sources(
            f"{machine_type} {symptoms}", KB_PROCEDURES, k=4
        )

    def query_machine_manual(self, machine_type: str, topic: str):
        return self.ctx.rag.retrieve_with_sources(f"{machine_type} {topic}", KB_MANUALS, k=2)

    def list_failure_cases(self, machine_type: str) -> list[dict[str, Any]]:
        return self.ctx.store.query(
            FAILURE_CASES, {"machine_type": machine_type}, self.ctx.tenant
        )

    def calculate_rul(self, severity: Severity) -> int:
        """Coarse Remaining-Useful-Life estimate in days from severity."""
        return {Severity.NORMAL: 365, Severity.WARNING: 45, Severity.CRITICAL: 7}[severity]

    # ---- core ------------------------------------------------------------
    def diagnose(self, report: AnomalyReport, machine_type: str = "") -> Diagnosis:
        """Structured-only entry point (delegates to :meth:`run`)."""
        return self.run(report, machine_type)[0]

    def _infer_root_cause(self, cases) -> tuple[str, float]:
        if not cases.documents:
            return "Undetermined — insufficient matching history", 0.0
        top = cases.documents[0]
        root = top.metadata.get("root_cause") or top.metadata.get("fault")
        if not root:
            # Pull the "Root cause: X." span out of a failure-case text blob.
            text = top.text
            if "Root cause:" in text:
                root = text.split("Root cause:")[1].split(".")[0].strip()
            else:
                root = text[:60]
        return root, top.score

    def _confidence(self, top_score: float, n_evidence: int, report: AnomalyReport) -> float:
        retrieval = min(1.0, top_score / 0.35)  # ~0.35 cosine is a strong lexical match
        coverage = min(1.0, n_evidence / 3)
        conf = 0.35 * retrieval + 0.25 * coverage + 0.4 * report.confidence
        if report.severity is Severity.NORMAL:
            conf = max(conf, 0.7)  # "nothing wrong" is itself a confident call
        return max(0.0, min(0.99, conf))

    def _narrate(self, report, root_cause, cases, manual) -> str:
        context = " ".join(cases.texts + manual.texts)
        prompt = (
            f"Anomaly: {report.anomaly_type} on {report.machine_id} "
            f"(severity {report.severity.value}). Candidate root cause: {root_cause}. "
            f"Evidence: {context}\nWrite a 1-2 sentence grounded diagnosis."
        )
        return self.ctx.llm.complete(
            prompt, self.ctx.tenant, system=self.SYSTEM, task_type="reasoning"
        ).text

    def run(self, report: AnomalyReport, machine_type: str = "") -> tuple[Diagnosis, AgentResponse]:
        """Does all retrieval/LLM work once; :meth:`diagnose` delegates here."""
        symptoms = f"{report.anomaly_type} {report.details}"
        cases = self.search_failure_history(machine_type, symptoms)
        manual = self.query_machine_manual(machine_type, report.anomaly_type)

        root_cause, top_score = self._infer_root_cause(cases)
        confidence = self._confidence(top_score, len(cases.documents), report)
        # LLM narrative (grounded); captured in the trace. The guard validates the
        # deterministic summary below so groundedness stays stable offline.
        self._narrate(report, root_cause, cases, manual)

        diagnosis = Diagnosis(
            root_cause=root_cause,
            confidence=round(confidence, 2),
            evidence=cases.sources + manual.sources,
            escalate=confidence < CONFIDENCE_FLOOR,
            rul_days=self.calculate_rul(report.severity),
        )
        response = AgentResponse(
            text=(
                f"Root cause: {root_cause} "
                f"(confidence {diagnosis.confidence}, RUL ~{diagnosis.rul_days} days)."
            ),
            structured=diagnosis.__dict__,
            sources=cases.sources + manual.sources,
        )
        return diagnosis, response
