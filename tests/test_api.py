"""API-level tests: auth enforcement, tenant header check, happy path, demo UI."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.security import decode_token, mint_token


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:  # triggers lifespan → builds + seeds services
        yield c


@pytest.fixture
def token():
    return mint_token("acme", name="ACME", budget_usd=100.0)


def test_health_is_open(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_requires_auth(client):
    r = client.post("/v1/analyze", json={"machine_id": "pump_001"})
    assert r.status_code == 401


def test_tenant_header_must_match_token(client, token):
    r = client.post(
        "/v1/analyze",
        json={"machine_id": "pump_001"},
        headers={"Authorization": f"Bearer {token}", "X-Tenant-ID": "globex"},
    )
    assert r.status_code == 403


def test_analyze_happy_path(client, token):
    r = client.post(
        "/v1/analyze",
        json={"machine_id": "pump_001", "sensor_data": {"vibration_mm_s": 6.5, "temperature_c": 95}},
        headers={"Authorization": f"Bearer {token}", "X-Tenant-ID": "acme"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["severity"] == "CRITICAL"
    assert body["trace"]  # pipeline step trace present


def test_metrics_reports_spend(client, token):
    r = client.get("/v1/metrics", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["tenant_id"] == "acme"


# --- Web UI / public demo ---------------------------------------------------
def test_root_serves_html_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Factory Health Agent" in r.text


def test_demo_token_is_open_and_valid(client):
    r = client.get("/v1/demo-token")
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == "acme"
    claims = decode_token(body["token"])
    assert claims["tenant_id"] == "acme"


def test_demo_token_authorizes_analyze(client):
    tok = client.get("/v1/demo-token").json()["token"]
    r = client.post(
        "/v1/analyze",
        json={"machine_id": "pump_001", "sensor_data": {"vibration_mm_s": 6.5, "temperature_c": 95}},
        headers={"Authorization": f"Bearer {tok}", "X-Tenant-ID": "acme"},
    )
    assert r.status_code == 200
    assert r.json()["severity"] == "CRITICAL"


def test_machines_endpoint_lists_catalog(client, token):
    r = client.get("/v1/machines", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    machines = r.json()
    assert len(machines) == 5
    ids = {m["machine_id"] for m in machines}
    assert "pump_001" in ids
    assert machines[0]["normal_ranges"]  # ranges present for UI presets


def test_machines_requires_auth(client):
    r = client.get("/v1/machines")
    assert r.status_code == 401


def test_analyze_includes_execution_trace(client, token):
    r = client.post(
        "/v1/analyze",
        json={"machine_id": "pump_001", "sensor_data": {"vibration_mm_s": 6.5, "temperature_c": 95}},
        headers={"Authorization": f"Bearer {token}", "X-Tenant-ID": "acme"},
    )
    assert r.status_code == 200
    trace = r.json()["execution_trace"]
    kinds = {e["kind"] for e in trace}
    # The full pipeline should expose agent, RAG, LLM, store, routing and guard steps.
    assert {"agent", "rag", "llm", "store", "route", "guard"} <= kinds
    rag = next(e for e in trace if e["kind"] == "rag")
    assert "hits" in rag["detail"]  # RAG events carry retrieved docs + scores
    llm = next(e for e in trace if e["kind"] == "llm")
    assert "model" in llm["detail"] and "tokens" in llm["detail"]
