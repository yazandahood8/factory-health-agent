"""API-level tests: auth enforcement, tenant header check, happy path."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.security import mint_token


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
