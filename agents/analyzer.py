"""Analyzer Agent — sensor data → anomaly report.

Safety-critical numbers (severity, thresholds) are computed deterministically
from ISO 10816 zone boundaries and the machine's spec, *not* by the LLM. The
LLM only writes the human-readable narrative, grounded in retrieved standards.
That split is intentional: never let a language model invent a safety figure.
"""
from __future__ import annotations

from typing import Any, Optional

from agents.base import AgentContext
from data.bootstrap import KB_STANDARDS, MACHINES, SENSOR_LOGS
from sdk.models import AgentResponse, AnomalyReport, Severity

# ISO 10816-3 RMS velocity zone boundaries (mm/s) by evaluation class.
# Zone B upper = acceptable; Zone C upper = unsatisfactory; above = unacceptable.
ISO_ZONES = {
    "Class I": (1.4, 2.8),
    "Class II": (2.8, 4.5),
    "Class III": (4.5, 7.1),
    "Class IV": (7.1, 11.0),
}


class AnalyzerAgent:
    name = "analyzer"
    SYSTEM = (
        "You are an industrial machine analyzer. Summarize the vibration "
        "assessment against ISO 10816. Cite the standard you reference. Be concise."
    )

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    # ---- tools -----------------------------------------------------------
    def get_machine_specs(self, machine_id: str) -> Optional[dict[str, Any]]:
        rows = self.ctx.store.query(MACHINES, {"machine_id": machine_id}, self.ctx.tenant)
        return rows[0] if rows else None

    def query_sensor_history(self, machine_id: str, limit: int = 72) -> list[dict[str, Any]]:
        rows = self.ctx.store.query(SENSOR_LOGS, {"machine_id": machine_id}, self.ctx.tenant)
        rows.sort(key=lambda r: r.get("timestamp", ""))
        return rows[-limit:]

    def retrieve_vibration_standards(self, machine_type: str):
        return self.ctx.rag.retrieve_with_sources(
            f"{machine_type} vibration ISO 10816 limits", KB_STANDARDS, k=2
        )

    # ---- core ------------------------------------------------------------
    def analyze(self, machine_id: str, sensor_data: Optional[dict] = None) -> AnomalyReport:
        """Structured-only entry point (delegates to :meth:`run`)."""
        return self.run(machine_id, sensor_data)[0]

    def _analyze(self, machine_id: str, sensor_data: Optional[dict] = None) -> AnomalyReport:
        specs = self.get_machine_specs(machine_id)
        history = self.query_sensor_history(machine_id)
        reading = sensor_data or (history[-1] if history else {})

        vibration = float(reading.get("vibration_mm_s", reading.get("vibration", 0.0)))
        temperature = float(reading.get("temperature_c", reading.get("temp", 0.0)))

        iso_class = (specs or {}).get("iso_class", "Class II")
        zone_b, zone_c = ISO_ZONES.get(iso_class, ISO_ZONES["Class II"])

        # Deterministic severity from ISO zones.
        if vibration <= zone_b:
            severity, anomaly_type = Severity.NORMAL, "within acceptable vibration zone"
        elif vibration <= zone_c:
            severity, anomaly_type = Severity.WARNING, "elevated vibration (ISO zone C)"
        else:
            severity, anomaly_type = Severity.CRITICAL, "excessive vibration (ISO zone D)"

        # Temperature out of spec escalates a borderline reading.
        temp_range = (specs or {}).get("normal_ranges", {}).get("temperature_c", [0, 999])
        temp_exceeded = temperature > temp_range[1]
        if temp_exceeded and severity is Severity.NORMAL:
            severity, anomaly_type = Severity.WARNING, "temperature above spec"

        # Append deterministic symptom descriptors so downstream retrieval has
        # real signal to match against historical failure cases.
        symptoms = self._symptoms(vibration, zone_b, temperature, temp_exceeded, history)
        if symptoms and severity is not Severity.NORMAL:
            anomaly_type = f"{anomaly_type}; {symptoms}"

        # Trend confirmation increases confidence.
        confidence = self._confidence(vibration, zone_b, zone_c, history)

        standards = self.retrieve_vibration_standards((specs or {}).get("machine_type", ""))
        details = self._narrate(
            machine_id, vibration, temperature, severity, anomaly_type, iso_class, standards
        )

        return AnomalyReport(
            machine_id=machine_id,
            severity=severity,
            anomaly_type=anomaly_type,
            details=details,
            confidence=round(confidence, 2),
            citations=standards.sources,
        )

    def _symptoms(self, vibration, zone_b, temperature, temp_exceeded, history) -> str:
        """Human/retrieval-friendly symptom phrases derived from the data."""
        phrases: list[str] = []
        if vibration > zone_b:
            phrases.append("high vibration at running speed")
        if vibration > 2 * zone_b:
            phrases.append("bearing defect frequencies")
        if temp_exceeded:
            phrases.append("elevated bearing temperature")
        # Rising trend over the window suggests a developing fault.
        if len(history) >= 2:
            recent = [float(h.get("vibration_mm_s", 0)) for h in history[-10:]]
            if recent and recent[-1] > 1.25 * (sum(recent) / len(recent)):
                phrases.append("rising vibration trend")
        return ", ".join(phrases)

    def _confidence(self, vibration, zone_b, zone_c, history) -> float:
        # Distance from the nearest zone boundary → how unambiguous the call is.
        nearest = min(abs(vibration - zone_b), abs(vibration - zone_c))
        clarity = min(1.0, nearest / max(zone_b, 0.1))
        base = 0.6 + 0.3 * clarity
        if len(history) >= 24:  # enough data to trust the trend
            base += 0.1
        return min(0.99, base)

    def _narrate(self, machine_id, vib, temp, severity, atype, iso_class, standards) -> str:
        context = " ".join(standards.texts)
        prompt = (
            f"Machine {machine_id} ({iso_class}). Vibration {vib} mm/s, "
            f"temperature {temp} C. Deterministic assessment: {severity.value} — {atype}. "
            f"Reference standard excerpts: {context}\n"
            "Write a 1-2 sentence grounded summary for a maintenance engineer."
        )
        result = self.ctx.llm.complete(
            prompt, self.ctx.tenant, system=self.SYSTEM, task_type="analysis"
        )
        return result.text

    def run(self, machine_id: str, sensor_data: Optional[dict] = None) -> tuple[AnomalyReport, AgentResponse]:
        """Returns the structured report plus a guard-checkable AgentResponse.

        Retrieves exactly once; :meth:`analyze` delegates here.
        """
        report = self._analyze(machine_id, sensor_data)
        response = AgentResponse(
            text=report.details,
            structured=report.__dict__,
            sources=report.citations,
        )
        return report, response
