"""Seed the document store with sample machine/sensor/failure data.

Run directly: ``python -m data.seed_mongodb``. Uses MongoDB when
``MONGODB_URI`` is configured, otherwise the in-memory store (in which case
seeding only persists for the life of the process — useful for a smoke test).
"""
from __future__ import annotations

from data.bootstrap import seed_store
from sdk.config import get_settings
from sdk.models import Tenant
from sdk.tenant_store import TenantIsolatedStore

DEMO_TENANTS = [
    Tenant(id="acme", name="ACME Manufacturing", llm_budget_usd=10.0),
    Tenant(id="globex", name="Globex Industrial", llm_budget_usd=10.0),
]


def main() -> None:
    settings = get_settings()
    store = TenantIsolatedStore(settings)
    backend = "MongoDB" if settings.mongodb_uri else "in-memory (ephemeral)"
    print(f"Seeding document store [{backend}]...")
    for tenant in DEMO_TENANTS:
        counts = seed_store(store, tenant)
        pretty = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  tenant '{tenant.id}': {pretty}")
    print("Done.")


if __name__ == "__main__":
    main()
