"""Latency — per-request wall time against an SLA, plus suite percentiles."""
from __future__ import annotations

from typing import Any

from evaluation.evaluators import EvalScore


class LatencyEvaluator:
    name = "latency"

    def __init__(self, sla_ms: float = 3000) -> None:
        self.sla_ms = sla_ms

    def evaluate(self, case: dict[str, Any], result: dict[str, Any]) -> EvalScore:
        latency_ms = float(result.get("_latency_ms", 0.0))
        return EvalScore(
            self.name, round(latency_ms, 1), latency_ms <= self.sla_ms, f"SLA {self.sla_ms}ms"
        )

    @staticmethod
    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
        return s[idx]
