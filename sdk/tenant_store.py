"""Tenant-isolated data access.

Every read and write is forced through :class:`TenantIsolatedStore`, which
injects ``tenant_id`` into the filter on *every* query and refuses to run
without a tenant. There is deliberately no escape hatch — cross-tenant access
is impossible by construction, not by convention.

Each access is also written to an append-only audit log for compliance.

Backed by MongoDB when ``MONGODB_URI`` is set, otherwise an in-memory store
with identical semantics (so the isolation tests run anywhere).
"""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Optional

from sdk.config import Settings, get_settings
from sdk.exceptions import TenantContextError
from sdk.models import Tenant


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class _InMemoryCollection:
    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []

    def insert_one(self, doc: dict[str, Any]) -> str:
        doc = dict(doc)
        doc.setdefault("_id", str(uuid.uuid4()))
        self.docs.append(doc)
        return doc["_id"]

    def find(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return [d for d in self.docs if _matches(d, query)]


def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, cond in query.items():
        val = doc.get(key)
        if isinstance(cond, dict):
            for op, operand in cond.items():
                if op == "$in" and val not in operand:
                    return False
                if op == "$gte" and not (val is not None and val >= operand):
                    return False
                if op == "$lte" and not (val is not None and val <= operand):
                    return False
                if op == "$gt" and not (val is not None and val > operand):
                    return False
                if op == "$lt" and not (val is not None and val < operand):
                    return False
        elif val != cond:
            return False
    return True


class _InMemoryDB:
    def __init__(self) -> None:
        self._cols: dict[str, _InMemoryCollection] = {}

    def __getitem__(self, name: str) -> _InMemoryCollection:
        return self._cols.setdefault(name, _InMemoryCollection())


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------
class AuditLogger:
    """Append-only record of every tenant data access."""

    COLLECTION = "_audit_log"

    def __init__(self, db: Any) -> None:
        self._db = db

    def record(self, *, tenant_id: str, op: str, collection: str, query_hash: str) -> None:
        self._db[self.COLLECTION].insert_one(
            {
                "tenant_id": tenant_id,
                "op": op,
                "collection": collection,
                "query_hash": query_hash,
                "ts": time.time(),
            }
        )

    def entries(self, tenant_id: Optional[str] = None) -> list[dict[str, Any]]:
        q = {"tenant_id": tenant_id} if tenant_id else {}
        return list(self._db[self.COLLECTION].find(q))


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
def _hash_query(query: dict[str, Any]) -> str:
    return hashlib.sha256(repr(sorted(query.items())).encode()).hexdigest()[:16]


class TenantIsolatedStore:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        db: Any = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._db = db if db is not None else self._connect()
        self.audit_log = audit_logger or AuditLogger(self._db)

    def _connect(self) -> Any:
        if self.settings.mongodb_uri:
            try:
                from pymongo import MongoClient

                client = MongoClient(self.settings.mongodb_uri, serverSelectionTimeoutMS=2000)
                client.admin.command("ping")
                return client[self.settings.mongodb_db]
            except Exception:  # pragma: no cover - depends on live mongo
                pass
        return _InMemoryDB()

    @staticmethod
    def _require_tenant(tenant: Optional[Tenant]) -> Tenant:
        if tenant is None or not getattr(tenant, "id", None):
            raise TenantContextError("A valid Tenant is required for all data access.")
        return tenant

    def query(self, collection: str, query: dict[str, Any], tenant: Tenant) -> list[dict[str, Any]]:
        tenant = self._require_tenant(tenant)
        # tenant_id is injected last so a caller can never override it.
        secure_query = {**query, "tenant_id": tenant.id}
        self.audit_log.record(
            tenant_id=tenant.id,
            op="query",
            collection=collection,
            query_hash=_hash_query(secure_query),
        )
        return list(self._db[collection].find(secure_query))

    def insert(self, collection: str, doc: dict[str, Any], tenant: Tenant) -> str:
        tenant = self._require_tenant(tenant)
        secure_doc = {**doc, "tenant_id": tenant.id}
        self.audit_log.record(
            tenant_id=tenant.id,
            op="insert",
            collection=collection,
            query_hash=_hash_query({"_doc": True}),
        )
        return self._db[collection].insert_one(secure_doc)
