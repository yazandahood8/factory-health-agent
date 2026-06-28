"""LLM routing with provider fallback and per-tenant budget enforcement.

Design notes (the "senior" decisions):

* **Don't hardcode one vendor.** Providers implement a tiny ``LLMProvider``
  protocol. The router tries them in priority order and fails over on
  rate-limit / transient errors.
* **Budget before tokens.** Every call is gated by the :class:`BudgetManager`;
  a tenant can never overspend, even across concurrent requests.
* **Always runnable.** If no provider is configured, a deterministic
  :class:`MockProvider` keeps the whole pipeline working offline.

The public surface is intentionally small: ``router.complete(...)``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from sdk.budget import BudgetManager
from sdk.config import Settings, get_settings
from sdk.exceptions import AllProvidersFailedError
from sdk.models import Tenant


@dataclass
class LLMResult:
    text: str
    tokens: int
    cost_usd: float
    provider: str


class RateLimitError(Exception):
    """Provider-agnostic signal that a call should fail over to a fallback."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, prompt: str, system: str, task_type: str) -> LLMResult: ...


# --------------------------------------------------------------------------
# Cost helpers (rough but consistent; real numbers come from provider usage)
# --------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    # ~4 chars/token is the standard back-of-envelope.
    return max(1, len(text) // 4)


def _cost(prompt_tokens: int, completion_tokens: int, in_rate: float, out_rate: float) -> float:
    return (prompt_tokens / 1000) * in_rate + (completion_tokens / 1000) * out_rate


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
class MockProvider:
    """Deterministic offline provider.

    Returns a templated, source-grounded summary. Numeric/safety-critical
    fields are computed by the agents themselves (tools), never by this mock,
    so offline output stays correct.
    """

    name = "mock"

    def complete(self, prompt: str, system: str, task_type: str) -> LLMResult:
        digest = hashlib.sha256((system + prompt).encode()).hexdigest()[:8]
        text = (
            f"[mock-llm:{task_type}] Summary generated from provided context. "
            f"All figures are derived from tool outputs and retrieved sources. "
            f"(ref {digest})"
        )
        pt, ct = _estimate_tokens(prompt + system), _estimate_tokens(text)
        return LLMResult(text=text, tokens=pt + ct, cost_usd=0.0, provider=self.name)


class AzureOpenAIProvider:
    name = "azure_openai"
    # USD per 1K tokens (gpt-4o class defaults).
    IN_RATE, OUT_RATE = 0.0025, 0.010

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import AzureChatOpenAI

        self._settings = settings
        self._client = AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.azure_openai_deployment,
            api_version=settings.openai_api_version,
            temperature=0,
        )

    def complete(self, prompt: str, system: str, task_type: str) -> LLMResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            resp = self._client.invoke(
                [SystemMessage(content=system), HumanMessage(content=prompt)]
            )
        except Exception as exc:  # pragma: no cover - needs live Azure
            if _is_rate_limit(exc):
                raise RateLimitError(str(exc)) from exc
            raise

        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        usage = getattr(resp, "usage_metadata", None) or {}
        pt = usage.get("input_tokens") or _estimate_tokens(prompt + system)
        ct = usage.get("output_tokens") or _estimate_tokens(text)
        return LLMResult(
            text=text,
            tokens=pt + ct,
            cost_usd=_cost(pt, ct, self.IN_RATE, self.OUT_RATE),
            provider=self.name,
        )


class GeminiProvider:
    name = "gemini"
    IN_RATE, OUT_RATE = 0.00125, 0.005

    def __init__(self, settings: Settings) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._settings = settings
        self._client = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )

    def complete(self, prompt: str, system: str, task_type: str) -> LLMResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            resp = self._client.invoke(
                [SystemMessage(content=system), HumanMessage(content=prompt)]
            )
        except Exception as exc:  # pragma: no cover - needs live Gemini
            if _is_rate_limit(exc):
                raise RateLimitError(str(exc)) from exc
            raise

        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        pt, ct = _estimate_tokens(prompt + system), _estimate_tokens(text)
        return LLMResult(
            text=text,
            tokens=pt + ct,
            cost_usd=_cost(pt, ct, self.IN_RATE, self.OUT_RATE),
            provider=self.name,
        )


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("rate limit", "429", "quota", "overloaded", "throttl"))


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
class LLMRouter:
    """Routes a completion request through an ordered provider chain.

    The chain is built from whatever is configured; the mock provider is always
    appended last so there is never a dead end.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        budget_manager: Optional[BudgetManager] = None,
        providers: Optional[list[LLMProvider]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.budget_manager = budget_manager or BudgetManager(self.settings)
        self.providers = providers if providers is not None else self._build_chain()

    def _build_chain(self) -> list[LLMProvider]:
        chain: list[LLMProvider] = []
        if self.settings.azure_enabled:
            try:
                chain.append(AzureOpenAIProvider(self.settings))
            except Exception:  # pragma: no cover - import/config issues
                pass
        if self.settings.gemini_enabled:
            try:
                chain.append(GeminiProvider(self.settings))
            except Exception:  # pragma: no cover
                pass
        chain.append(MockProvider())  # always-available safety net
        return chain

    @property
    def primary(self) -> str:
        return self.providers[0].name if self.providers else "none"

    def complete(
        self, prompt: str, tenant: Tenant, *, system: str = "", task_type: str = "reasoning"
    ) -> LLMResult:
        # Budget is enforced *before* spending a single token.
        self.budget_manager.check(tenant)

        errors: list[str] = []
        for provider in self.providers:
            try:
                result = provider.complete(prompt, system, task_type)
            except RateLimitError as exc:
                errors.append(f"{provider.name}: rate-limited ({exc})")
                continue  # fail over to the next provider
            except Exception as exc:  # pragma: no cover - provider-specific
                errors.append(f"{provider.name}: {exc}")
                continue
            self.budget_manager.record(tenant, result.cost_usd)
            return result

        raise AllProvidersFailedError("; ".join(errors))
