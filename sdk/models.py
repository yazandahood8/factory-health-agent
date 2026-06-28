"""Shared domain models used across the SDK and agents.

Plain dataclasses / Pydantic models — no I/O, no side effects — so they are
trivially importable and testable from any layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, TypedDict


# --------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------
class Tier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass(frozen=True)
class Tenant:
    """A customer of the platform. The unit of isolation and budgeting."""

    id: str
    name: str = ""
    tier: Tier = Tier.PRO
    llm_budget_usd: float = 10.0


# --------------------------------------------------------------------------
# Severity / urgency vocabulary (shared by agents)
# --------------------------------------------------------------------------
class Severity(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Urgency(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# --------------------------------------------------------------------------
# Agent outputs
# --------------------------------------------------------------------------
@dataclass
class AnomalyReport:
    machine_id: str
    severity: Severity
    anomaly_type: str
    details: str
    confidence: float
    citations: list[str] = field(default_factory=list)


@dataclass
class Diagnosis:
    root_cause: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    escalate: bool = False
    rul_days: Optional[int] = None  # Remaining Useful Life estimate


@dataclass
class ActionPlan:
    actions: list[str]
    urgency: Urgency
    estimated_downtime_hours: float
    parts_needed: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Generic wrapper an agent returns before guardrail validation."""

    text: str
    structured: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class ValidatedResponse:
    action: Literal["RETURN", "ESCALATE_TO_HUMAN"]
    response: Optional[AgentResponse] = None
    reason: str = ""
    groundedness: float = 1.0


# --------------------------------------------------------------------------
# Orchestration state (LangGraph)
# --------------------------------------------------------------------------
class PipelineState(TypedDict, total=False):
    # Input
    machine_id: str
    sensor_data: dict[str, Any]
    tenant: Tenant
    # Intermediate
    anomaly_report: Optional[AnomalyReport]
    diagnosis: Optional[Diagnosis]
    action_plan: Optional[ActionPlan]
    validated_response: Optional[ValidatedResponse]
    # Output / bookkeeping
    escalated: bool
    trace_id: str
    total_tokens: int
    total_cost_usd: float
    messages: list[str]
    # Internal plumbing (not part of the public response)
    machine_type: str
    diag_response: Any
    recorder: Any
