"""Diagnosis accuracy — predicted vs. expert ground-truth root cause.

Uses token-overlap (fuzzy) matching so phrasing differences don't fail a
correct diagnosis, with an exact-substring fast path.
"""
from __future__ import annotations

import re
from typing import Any

from evaluation.evaluators import EvalScore

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "due", "and", "or", "in"}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP}


class DiagnosisAccuracyEvaluator:
    name = "diagnosis_accuracy"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def evaluate(self, case: dict[str, Any], result: dict[str, Any]) -> EvalScore:
        expected = case.get("expected_diagnosis", "")
        diagnosis = (result.get("diagnosis") or {})
        predicted = diagnosis.get("root_cause", "")

        if not expected:
            return EvalScore(self.name, 1.0, True, "no ground truth")

        if expected.lower() in predicted.lower() or predicted.lower() in expected.lower():
            return EvalScore(self.name, 1.0, True, "exact match")

        e, p = _tokens(expected), _tokens(predicted)
        jaccard = len(e & p) / len(e | p) if (e | p) else 0.0
        return EvalScore(
            self.name, round(jaccard, 3), jaccard >= self.threshold,
            f"'{predicted}' vs '{expected}'",
        )
