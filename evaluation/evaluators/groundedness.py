"""Groundedness — was the narrative supported by retrieved sources?

Prefers the *production* groundedness number: the guard node already scored the
response against the actual retrieved document texts and stored it on
``validated_response.groundedness``. Reusing it means the eval metric is exactly
what the guardrail enforces in production — no measurement drift.
"""
from __future__ import annotations

from typing import Any

from evaluation.evaluators import EvalScore


class GroundednessEvaluator:
    name = "groundedness"

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def evaluate(self, case: dict[str, Any], result: dict[str, Any]) -> EvalScore:
        validated = result.get("validated_response") or {}
        if "groundedness" in validated:
            score = float(validated["groundedness"])
            return EvalScore(self.name, round(score, 3), score >= self.threshold)

        # No guard ran: either a deterministic NORMAL result or an escalation
        # before recommendation. Both are "safe" by construction.
        reason = "escalated (guarded)" if result.get("escalated") else "deterministic, no LLM claim"
        return EvalScore(self.name, 1.0, True, reason)
