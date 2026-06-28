"""Agent + pipeline behavior tests (offline, deterministic)."""
from __future__ import annotations

import pytest

from agents.analyzer import AnalyzerAgent
from agents.diagnostics import DiagnosticsAgent
from agents.orchestrator import Pipeline, build_services
from agents.recommender import RecommenderAgent
from sdk.models import AnomalyReport, Diagnosis, Severity, Urgency


@pytest.fixture(scope="module")
def services():
    from sdk.config import Settings

    return build_services(Settings(mongodb_uri="", redis_url="", chroma_host=""), seed=True)


@pytest.fixture(scope="module")
def pipeline(services):
    return Pipeline(services)


@pytest.fixture
def ctx(services, tenant_a):
    return services.context_for(tenant_a)


def test_analyzer_detects_critical_vibration(ctx):
    report = AnalyzerAgent(ctx).analyze("pump_001", {"vibration_mm_s": 6.5, "temperature_c": 95})
    assert report.severity is Severity.CRITICAL
    assert report.citations  # cited a standard


def test_analyzer_reports_normal_for_in_spec(ctx):
    report = AnalyzerAgent(ctx).analyze("pump_001", {"vibration_mm_s": 1.2, "temperature_c": 55})
    assert report.severity is Severity.NORMAL


def test_diagnostics_escalates_on_low_confidence(ctx):
    # An anomaly with no matching machine_type history → weak evidence → escalate.
    report = AnomalyReport(
        machine_id="unknown_x", severity=Severity.WARNING,
        anomaly_type="mystery signal", details="unrecognized", confidence=0.4,
    )
    diagnosis = DiagnosticsAgent(ctx).diagnose(report, machine_type="nonexistent_type")
    assert diagnosis.escalate is True


def test_recommender_critical_first_action_is_load_reduction(ctx):
    diagnosis = Diagnosis(root_cause="Bearing outer race defect", confidence=0.9, rul_days=7)
    plan = RecommenderAgent(ctx).recommend(diagnosis, Severity.CRITICAL)
    assert plan.urgency is Urgency.CRITICAL
    assert "reduce load" in plan.actions[0].lower()


def test_full_pipeline_end_to_end(pipeline, tenant_a):
    state = pipeline.run("pump_001", tenant_a, {"vibration_mm_s": 6.5, "temperature_c": 95})
    assert state["anomaly_report"].severity is Severity.CRITICAL
    assert state["diagnosis"] is not None
    # Critical + confident → produces a plan (not escalated).
    assert state.get("action_plan") is not None
    assert state["escalated"] is False


def test_full_pipeline_normal_short_circuits(pipeline, tenant_a):
    state = pipeline.run("pump_001", tenant_a, {"vibration_mm_s": 1.0, "temperature_c": 50})
    assert state["anomaly_report"].severity is Severity.NORMAL
    assert state.get("diagnosis") is None  # skipped diagnostics
    assert state.get("action_plan") is None


def test_pipeline_isolation_between_tenants(pipeline, tenant_a, tenant_b):
    # Both tenants seeded independently; each sees only its own machines.
    a = pipeline.run("pump_001", tenant_a, {"vibration_mm_s": 6.5, "temperature_c": 95})
    b = pipeline.run("pump_001", tenant_b, {"vibration_mm_s": 6.5, "temperature_c": 95})
    assert a["trace_id"] != b["trace_id"]
