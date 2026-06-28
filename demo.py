"""Standalone end-to-end demo — no API, no external services required.

    python demo.py

Runs the full pipeline on a few machines and prints the trace + result, so the
whole system can be shown in one command during an interview.
"""
from __future__ import annotations

from agents.orchestrator import Pipeline
from sdk.models import Tenant

SCENARIOS = [
    ("pump_001", {"vibration_mm_s": 6.5, "temperature_c": 95}, "critical bearing"),
    ("motor_002", {"vibration_mm_s": 5.5, "temperature_c": 70}, "warning"),
    ("compressor_003", {"vibration_mm_s": 2.0, "temperature_c": 80}, "healthy"),
]


def main() -> None:
    tenant = Tenant(id="acme", name="ACME Manufacturing", llm_budget_usd=100.0)
    pipeline = Pipeline()
    print(f"LLM provider: {pipeline.services.llm.primary}\n")

    for machine_id, sensor, label in SCENARIOS:
        print("=" * 64)
        print(f"{machine_id}  ({label})  ->  {sensor}")
        print("-" * 64)
        state = pipeline.run(machine_id, tenant, sensor)
        for step in state["messages"]:
            print(f"  - {step}")
        report = state.get("anomaly_report")
        print(f"\n  Severity:  {report.severity.value}")
        if state.get("diagnosis"):
            d = state["diagnosis"]
            print(f"  Diagnosis: {d.root_cause}  (confidence {d.confidence}, RUL ~{d.rul_days}d)")
        if state.get("action_plan"):
            p = state["action_plan"]
            print(f"  Urgency:   {p.urgency.value}  (downtime ~{p.estimated_downtime_hours}h)")
            for a in p.actions:
                print(f"      - {a}")
        if state.get("escalated"):
            print("  ESCALATED to human expert.")
        print()


if __name__ == "__main__":
    main()
