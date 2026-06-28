"""Run the evaluation suite against the full pipeline.

Loads test cases, executes each through the LangGraph pipeline, runs every
evaluator, prints a LangSmith-style report, optionally uploads to LangSmith
(when configured), and exits non-zero if any aggregate metric misses its SLA —
so it doubles as a CI quality gate.

Usage:  python -m evaluation.run_eval
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from agents.orchestrator import Pipeline
from api.serialization import to_jsonable
from evaluation.evaluators.cost import CostEvaluator
from evaluation.evaluators.diagnosis_accuracy import DiagnosisAccuracyEvaluator
from evaluation.evaluators.groundedness import GroundednessEvaluator
from evaluation.evaluators.latency import LatencyEvaluator
from sdk.models import Tenant

TEST_CASES = Path(__file__).parent / "test_cases.json"
EVAL_TENANT = Tenant(id="acme", name="ACME Manufacturing", llm_budget_usd=100.0)

# Aggregate SLAs (the CI gate).
SLA = {
    "diagnosis_accuracy": 0.85,
    "groundedness": 0.80,
    "p95_latency_ms": 3000,
    "avg_cost_usd": 0.05,
    "severity_accuracy": 0.90,
}


def run_case(pipeline: Pipeline, case: dict[str, Any]) -> dict[str, Any]:
    inp = case["input"]
    spend_before = pipeline.services.llm.budget_manager.current_spend(EVAL_TENANT)
    t0 = time.perf_counter()
    state = pipeline.run(inp["machine_id"], EVAL_TENANT, inp.get("sensor_data"))
    latency_ms = (time.perf_counter() - t0) * 1000
    spend_after = pipeline.services.llm.budget_manager.current_spend(EVAL_TENANT)

    result = {
        "machine_id": state["machine_id"],
        "escalated": state.get("escalated", False),
        "anomaly_report": to_jsonable(state.get("anomaly_report")),
        "diagnosis": to_jsonable(state.get("diagnosis")),
        "action_plan": to_jsonable(state.get("action_plan")),
        "validated_response": to_jsonable(state.get("validated_response")),
        "total_tokens": state.get("total_tokens", 0),
        "_latency_ms": latency_ms,
        "_request_cost_usd": spend_after - spend_before,
    }
    return result


def main() -> int:
    cases = json.loads(TEST_CASES.read_text(encoding="utf-8"))
    pipeline = Pipeline()

    evaluators = [
        DiagnosisAccuracyEvaluator(),
        GroundednessEvaluator(),
        LatencyEvaluator(sla_ms=SLA["p95_latency_ms"]),
        CostEvaluator(max_usd=SLA["avg_cost_usd"]),
    ]

    scores: dict[str, list[float]] = {e.name: [] for e in evaluators}
    latencies: list[float] = []
    costs: list[float] = []
    severity_hits = severity_total = 0
    diag_considered = 0

    for case in cases:
        result = run_case(pipeline, case)
        latencies.append(result["_latency_ms"])
        costs.append(result["_request_cost_usd"])

        # Severity accuracy (deterministic, always checkable).
        if "expected_severity" in case:
            severity_total += 1
            actual_sev = (result.get("anomaly_report") or {}).get("severity")
            if actual_sev == case["expected_severity"]:
                severity_hits += 1

        for ev in evaluators:
            # Only score diagnosis accuracy on cases that have ground truth AND
            # actually produced a diagnosis (skip NORMAL/escalated).
            if ev.name == "diagnosis_accuracy":
                if not case.get("expected_diagnosis") or not result.get("diagnosis"):
                    continue
                diag_considered += 1
            s = ev.evaluate(case, result)
            scores[ev.name].append(s.score)

    # Aggregate.
    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 1.0

    p95 = LatencyEvaluator.percentile(latencies, 95)
    diag_acc = avg(scores["diagnosis_accuracy"])
    grounded = avg(scores["groundedness"])
    avg_cost = avg(costs)
    sev_acc = severity_hits / severity_total if severity_total else 1.0

    rows = [
        ("Diagnosis Accuracy", diag_acc, SLA["diagnosis_accuracy"], diag_acc >= SLA["diagnosis_accuracy"], "{:.1%}"),
        ("Severity Accuracy", sev_acc, SLA["severity_accuracy"], sev_acc >= SLA["severity_accuracy"], "{:.1%}"),
        ("Groundedness", grounded, SLA["groundedness"], grounded >= SLA["groundedness"], "{:.2f}"),
        ("P95 Latency (ms)", p95, SLA["p95_latency_ms"], p95 <= SLA["p95_latency_ms"], "{:.0f}"),
        ("Avg Cost (USD)", avg_cost, SLA["avg_cost_usd"], avg_cost <= SLA["avg_cost_usd"], "${:.4f}"),
    ]

    print("\nFactory Health Agent - Eval Report")
    print("=" * 52)
    print(f"Test Cases:   {len(cases)}")
    print(f"LLM provider: {pipeline.services.llm.primary}")
    print(f"{'Metric':<22}{'Score':>12}{'SLA':>10}  Status")
    print("-" * 52)
    all_pass = True
    for name, score, sla, ok, fmt in rows:
        all_pass &= ok
        sla_str = fmt.format(sla) if name != "P95 Latency (ms)" else f"{sla:.0f}"
        print(f"{name:<22}{fmt.format(score):>12}{sla_str:>10}  {'PASS' if ok else 'FAIL'}")
    print("=" * 52)
    print("RESULT:", "ALL METRICS PASS" if all_pass else "SLA REGRESSION DETECTED")

    _maybe_upload_langsmith(cases)
    return 0 if all_pass else 1


def _maybe_upload_langsmith(cases: list[dict]) -> None:
    from sdk.config import get_settings

    settings = get_settings()
    if not (settings.langchain_tracing_v2 and settings.langchain_api_key):
        print("(LangSmith upload skipped - set LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true)")
        return
    try:  # pragma: no cover - needs live LangSmith
        from langsmith import Client

        client = Client(api_key=settings.langchain_api_key)
        ds_name = f"{settings.langchain_project}-eval"
        dataset = client.create_dataset(ds_name)
        for c in cases:
            client.create_example(
                inputs=c["input"], outputs={k: v for k, v in c.items() if k.startswith("expected")},
                dataset_id=dataset.id,
            )
        print(f"(Uploaded {len(cases)} examples to LangSmith dataset '{ds_name}')")
    except Exception as exc:  # pragma: no cover
        print(f"(LangSmith upload failed: {exc})")


if __name__ == "__main__":
    sys.exit(main())
