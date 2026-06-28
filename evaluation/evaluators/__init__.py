"""Pluggable evaluators. Each takes (case, result) and returns an EvalScore."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class EvalScore:
    name: str
    score: float
    passed: bool
    detail: str = ""


class Evaluator(Protocol):
    name: str

    def evaluate(self, case: dict[str, Any], result: dict[str, Any]) -> EvalScore: ...
