"""The most important tests in the codebase: cross-tenant data isolation.

A single leak here is the catastrophic failure mode for a multi-tenant B2B
platform, so these assert isolation at the infrastructure level — not via app
conventions that a future caller could forget.
"""
from __future__ import annotations

import pytest

from sdk.budget import BudgetManager
from sdk.exceptions import BudgetExceededException, TenantContextError
from sdk.tenant_store import TenantIsolatedStore


@pytest.fixture
def store(offline_settings):
    return TenantIsolatedStore(offline_settings)


def test_query_always_includes_tenant_id(store, tenant_a, monkeypatch):
    captured = {}
    real_find = store._db["machines"].find

    def spy(query):
        captured["query"] = query
        return real_find(query)

    monkeypatch.setattr(store._db["machines"], "find", spy)
    store.query("machines", {"machine_id": "pump_001"}, tenant_a)
    assert captured["query"]["tenant_id"] == tenant_a.id


def test_caller_cannot_override_tenant_id(store, tenant_a, tenant_b):
    # A caller maliciously passing another tenant_id is overridden, not honored:
    # the query is forced back to the caller's own namespace.
    store.insert("machines", {"machine_id": "m1"}, tenant_a)
    store.insert("machines", {"machine_id": "secret_b"}, tenant_b)
    rows = store.query("machines", {"tenant_id": tenant_b.id}, tenant_a)
    assert [r["machine_id"] for r in rows] == ["m1"]  # got acme's data, never globex's
    assert all(r["tenant_id"] == tenant_a.id for r in rows)


def test_cross_tenant_data_not_accessible(store, tenant_a, tenant_b):
    store.insert("machines", {"machine_id": "secret_a"}, tenant_a)
    store.insert("machines", {"machine_id": "secret_b"}, tenant_b)

    a_rows = store.query("machines", {}, tenant_a)
    b_rows = store.query("machines", {}, tenant_b)

    assert {r["machine_id"] for r in a_rows} == {"secret_a"}
    assert {r["machine_id"] for r in b_rows} == {"secret_b"}


def test_audit_log_written_on_every_access(store, tenant_a):
    store.insert("machines", {"machine_id": "m1"}, tenant_a)
    store.query("machines", {}, tenant_a)
    entries = store.audit_log.entries(tenant_a.id)
    ops = sorted(e["op"] for e in entries)
    assert ops == ["insert", "query"]
    assert all(e["tenant_id"] == tenant_a.id for e in entries)


def test_missing_tenant_is_rejected(store):
    with pytest.raises(TenantContextError):
        store.query("machines", {}, None)


def test_budget_exceeded_blocks_request(offline_settings, tenant_a):
    bm = BudgetManager(offline_settings)
    bm.record(tenant_a, tenant_a.llm_budget_usd + 1)
    assert bm.is_over_limit(tenant_a)
    with pytest.raises(BudgetExceededException):
        bm.check(tenant_a)
