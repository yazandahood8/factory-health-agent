"""Public demo support: a self-serve token + the machine catalog.

``/v1/demo-token`` lets the bundled web UI (and curious visitors) exercise the
full pipeline without credentials. It does not bypass auth — it *issues* a real,
short-lived, budget-capped JWT for the seeded demo tenant, so the JWT machinery
is exactly the same one used in production.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import DemoTokenResponse, MachineInfo
from api.security import mint_token
from data.bootstrap import MACHINES
from data.seed_mongodb import DEMO_TENANTS

router = APIRouter(prefix="/v1", tags=["demo"])

# The demo tenant is the first seeded tenant (ACME); its machines/data exist.
DEMO_TENANT = DEMO_TENANTS[0]
DEMO_TTL_SECONDS = 15 * 60
DEMO_BUDGET_USD = 25.0  # generous; Gemini flash is sub-cent/request


@router.get("/demo-token", response_model=DemoTokenResponse)
async def demo_token() -> DemoTokenResponse:
    token = mint_token(
        DEMO_TENANT.id,
        name=DEMO_TENANT.name,
        tier="pro",
        budget_usd=DEMO_BUDGET_USD,
        ttl_seconds=DEMO_TTL_SECONDS,
    )
    return DemoTokenResponse(
        token=token, tenant_id=DEMO_TENANT.id, expires_in=DEMO_TTL_SECONDS
    )


@router.get("/machines", response_model=list[MachineInfo])
async def list_machines(request: Request) -> list[MachineInfo]:
    """Tenant-scoped machine catalog used to populate the UI dropdown."""
    services = request.app.state.services
    tenant = request.state.tenant
    rows = services.store.query(MACHINES, {}, tenant)
    rows.sort(key=lambda r: r.get("machine_id", ""))
    return [
        MachineInfo(
            machine_id=r.get("machine_id", ""),
            machine_type=r.get("machine_type", ""),
            manufacturer=r.get("manufacturer", ""),
            model=r.get("model", ""),
            iso_class=r.get("iso_class", ""),
            normal_ranges=r.get("normal_ranges", {}),
        )
        for r in rows
    ]
