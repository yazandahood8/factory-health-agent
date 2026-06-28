"""SDK-wide exception hierarchy."""
from __future__ import annotations


class SDKError(Exception):
    """Base class for all SDK errors."""


class BudgetExceededException(SDKError):
    """Raised when a tenant has exhausted its LLM spend budget."""


class AllProvidersFailedError(SDKError):
    """Raised when every LLM provider (primary + fallbacks) failed."""


class TenantContextError(SDKError):
    """Raised when a tenant-scoped operation is attempted without a tenant."""
