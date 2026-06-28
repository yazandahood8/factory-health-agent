"""Groundedness guardrail.

In an industrial setting a confident-but-wrong recommendation can damage
equipment or hurt someone. So the guard treats *ungrounded* output as a
failure state: if the response is not sufficiently supported by retrieved
sources, it does not return — it escalates to a human expert.

Groundedness is measured lexically (token cosine vs. sources) so it works
fully offline. The interface accepts a pluggable scorer, so a stronger
embedding/NLI-based scorer can be swapped in without touching callers.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from sdk.models import AgentResponse


@dataclass
class RetrievedContext:
    """Sources an agent grounded its answer in (kept here to avoid cycles)."""

    documents: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "at", "for", "with", "this", "that", "it", "as", "be", "by",
    "from", "all", "any", "its", "which", "you",
}


def _tokens(text: str) -> Counter:
    return Counter(t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def _split_claims(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


class GroundednessChecker:
    """Returns the fraction of claims supported by at least one source."""

    def __init__(self, per_claim_threshold: float = 0.18) -> None:
        self.per_claim_threshold = per_claim_threshold

    def check(self, response_text: str, source_docs: list[str]) -> float:
        claims = _split_claims(response_text)
        if not claims:
            return 1.0
        if not source_docs:
            return 0.0
        source_vecs = [_tokens(d) for d in source_docs]
        grounded = 0
        for claim in claims:
            cv = _tokens(claim)
            best = max((_cosine(cv, sv) for sv in source_vecs), default=0.0)
            if best >= self.per_claim_threshold:
                grounded += 1
        return grounded / len(claims)


class HallucinationGuard:
    def __init__(
        self,
        threshold: float = 0.8,
        checker: Optional[GroundednessChecker] = None,
        scorer: Optional[Callable[[str, list[str]], float]] = None,
    ) -> None:
        self.threshold = threshold
        self._score = scorer or (checker or GroundednessChecker()).check

    def validate(self, response: AgentResponse, context: RetrievedContext):
        from sdk.models import ValidatedResponse

        score = self._score(response.text, context.documents)
        if score < self.threshold:
            return ValidatedResponse(
                action="ESCALATE_TO_HUMAN",
                response=response,
                reason=f"Low groundedness: {score:.2f} < {self.threshold:.2f}",
                groundedness=score,
            )
        return ValidatedResponse(action="RETURN", response=response, groundedness=score)
