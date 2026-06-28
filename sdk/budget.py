"""Per-tenant LLM spend tracking.

Backed by Redis when ``REDIS_URL`` is set, otherwise an in-process counter.
The interface is identical either way so callers never branch on the backend.
"""
from __future__ import annotations

import threading
from typing import Optional

from sdk.config import Settings, get_settings
from sdk.exceptions import BudgetExceededException
from sdk.models import Tenant


class _InMemoryBackend:
    """Process-local fallback. Thread-safe; resets on restart."""

    def __init__(self) -> None:
        self._spend: dict[str, float] = {}
        self._lock = threading.Lock()

    def incr(self, key: str, amount: float) -> float:
        with self._lock:
            self._spend[key] = self._spend.get(key, 0.0) + amount
            return self._spend[key]

    def get(self, key: str) -> float:
        with self._lock:
            return self._spend.get(key, 0.0)

    def reset(self, key: str) -> None:
        with self._lock:
            self._spend.pop(key, None)


class _RedisBackend:
    def __init__(self, url: str) -> None:
        import redis  # imported lazily so the dep is optional

        self._r = redis.Redis.from_url(url, decode_responses=True)
        # Fail fast if the URL is set but the server is unreachable.
        self._r.ping()

    @staticmethod
    def _k(key: str) -> str:
        return f"budget:spend:{key}"

    def incr(self, key: str, amount: float) -> float:
        return float(self._r.incrbyfloat(self._k(key), amount))

    def get(self, key: str) -> float:
        val = self._r.get(self._k(key))
        return float(val) if val is not None else 0.0

    def reset(self, key: str) -> None:
        self._r.delete(self._k(key))


class BudgetManager:
    """Tracks and enforces per-tenant cumulative LLM spend."""

    def __init__(self, settings: Optional[Settings] = None, backend=None) -> None:
        self.settings = settings or get_settings()
        if backend is not None:
            self.backend = backend
        elif self.settings.redis_url:
            try:
                self.backend = _RedisBackend(self.settings.redis_url)
            except Exception:  # pragma: no cover - depends on live redis
                self.backend = _InMemoryBackend()
        else:
            self.backend = _InMemoryBackend()

    def current_spend(self, tenant: Tenant) -> float:
        return self.backend.get(tenant.id)

    def is_over_limit(self, tenant: Tenant) -> bool:
        limit = tenant.llm_budget_usd or self.settings.default_tenant_budget_usd
        return self.current_spend(tenant) >= limit

    def check(self, tenant: Tenant) -> None:
        """Raise if the tenant is already at/over budget."""
        if self.is_over_limit(tenant):
            raise BudgetExceededException(
                f"Tenant '{tenant.id}' exceeded budget "
                f"(${self.current_spend(tenant):.4f} / ${tenant.llm_budget_usd:.2f})"
            )

    def record(self, tenant: Tenant, cost_usd: float) -> float:
        return self.backend.incr(tenant.id, cost_usd)

    def reset(self, tenant: Tenant) -> None:
        self.backend.reset(tenant.id)
