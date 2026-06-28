"""Cost — token spend per request against a budget SLA."""
from __future__ import annotations

from typing import Any

from evaluation.evaluators import EvalScore


class CostEvaluator:
    name = "cost"

    def __init__(self, max_usd: float = 0.05) -> None:
        self.max_usd = max_usd

    def evaluate(self, case: dict[str, Any], result: dict[str, Any]) -> EvalScore:
        cost = float(result.get("_request_cost_usd", result.get("total_cost_usd", 0.0)))
        return EvalScore(
            self.name, round(cost, 6), cost <= self.max_usd, f"max ${self.max_usd}"
        )
