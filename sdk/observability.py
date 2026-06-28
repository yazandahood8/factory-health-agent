"""LangSmith tracing wiring.

LangChain/LangGraph auto-trace when ``LANGCHAIN_*`` variables are present in the
process environment. We load configuration from ``.env`` via pydantic-settings,
which does *not* populate ``os.environ`` — so this module bridges the two:
call :func:`enable_tracing` once at startup and every LLM/graph call is traced.

No-op (and safe) when tracing is disabled or no API key is set.
"""
from __future__ import annotations

import os
from typing import Optional

from sdk.config import Settings, get_settings

_applied = False


def enable_tracing(settings: Optional[Settings] = None) -> bool:
    """Propagate LangSmith settings to os.environ. Returns True if tracing is on."""
    global _applied
    settings = settings or get_settings()

    if not (settings.langchain_tracing_v2 and settings.langchain_api_key):
        return False

    if not _applied:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        _applied = True
    return True
