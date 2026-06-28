"""Lightweight per-tenant request metrics collected in memory.

Tracks request counts, error counts, and latency samples so the /v1/metrics
endpoint can report real SRE data rather than just billing info.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import NamedTuple


class TenantStats(NamedTuple):
    requests: int
    errors: int
    latency_samples: list[float]

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        s = sorted(self.latency_samples)
        idx = max(0, int(len(s) * 0.95) - 1)
        return round(s[idx] * 1000, 1)


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, list[float]] = defaultdict(list)

    def record(self, tenant_id: str, latency_s: float, *, error: bool = False) -> None:
        with self._lock:
            self._requests[tenant_id] += 1
            if error:
                self._errors[tenant_id] += 1
            samples = self._latencies[tenant_id]
            samples.append(latency_s)
            if len(samples) > 1000:
                del samples[:500]

    def stats(self, tenant_id: str) -> TenantStats:
        with self._lock:
            return TenantStats(
                requests=self._requests.get(tenant_id, 0),
                errors=self._errors.get(tenant_id, 0),
                latency_samples=list(self._latencies.get(tenant_id, [])),
            )
