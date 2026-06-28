"""SDK unit tests: LLM router fallback, guard escalation, RAG retrieval."""
from __future__ import annotations

import pytest

from sdk.hallucination_guard import GroundednessChecker, HallucinationGuard, RetrievedContext
from sdk.llm_router import LLMResult, LLMRouter, MockProvider, RateLimitError
from sdk.models import AgentResponse, Tenant
from sdk.rag_engine import Document, RAGEngine


class _FlakyProvider:
    name = "flaky"

    def complete(self, prompt, system, task_type):
        raise RateLimitError("429 rate limit")


class _GoodProvider:
    name = "good"

    def complete(self, prompt, system, task_type):
        return LLMResult(text="ok", tokens=10, cost_usd=0.001, provider=self.name)


def test_router_fails_over_on_rate_limit(offline_settings, tenant_a):
    router = LLMRouter(offline_settings, providers=[_FlakyProvider(), _GoodProvider()])
    result = router.complete("hi", tenant_a)
    assert result.provider == "good"


def test_router_always_has_mock_fallback(offline_settings, tenant_a):
    router = LLMRouter(offline_settings)  # no creds → mock only
    result = router.complete("hi", tenant_a)
    assert result.provider == MockProvider.name


def test_router_records_spend(offline_settings, tenant_a):
    router = LLMRouter(offline_settings, providers=[_GoodProvider()])
    router.complete("hi", tenant_a)
    assert router.budget_manager.current_spend(tenant_a) == pytest.approx(0.001)


def test_groundedness_high_for_supported_text():
    checker = GroundednessChecker()
    sources = ["Bearing outer race defect causes high vibration and elevated temperature."]
    score = checker.check("High vibration indicates a bearing outer race defect.", sources)
    assert score >= 0.8


def test_groundedness_low_for_unsupported_text():
    checker = GroundednessChecker()
    sources = ["Routine lubrication schedule for centrifugal pumps."]
    score = checker.check("The reactor core temperature is approaching meltdown.", sources)
    assert score < 0.8


def test_guard_escalates_ungrounded_response():
    guard = HallucinationGuard(threshold=0.8)
    resp = AgentResponse(text="Completely unrelated fabricated claim about rockets.")
    ctx = RetrievedContext(documents=["pump bearing maintenance procedure"])
    validated = guard.validate(resp, ctx)
    assert validated.action == "ESCALATE_TO_HUMAN"


def test_rag_retrieves_relevant_docs():
    rag = RAGEngine()  # in-memory backend
    rag.index_documents(
        [
            Document("Bearing vibration rises with outer race defects.", "doc1"),
            Document("Cooling tower water treatment schedule.", "doc2"),
        ],
        "kb",
    )
    hits = rag.retrieve("bearing vibration", "kb", k=1)
    assert hits and hits[0].source == "doc1"
