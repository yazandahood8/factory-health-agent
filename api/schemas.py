"""Pydantic request/response models for the public API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SensorData(BaseModel):
    vibration_mm_s: Optional[float] = Field(None, description="RMS velocity, mm/s")
    temperature_c: Optional[float] = None
    pressure_bar: Optional[float] = None
    rpm: Optional[int] = None


class AnalyzeRequest(BaseModel):
    machine_id: str
    sensor_data: Optional[SensorData] = None


class AnalyzeResponse(BaseModel):
    machine_id: str
    trace_id: str
    severity: str
    escalated: bool
    anomaly_report: Optional[dict[str, Any]] = None
    diagnosis: Optional[dict[str, Any]] = None
    action_plan: Optional[dict[str, Any]] = None
    validated_response: Optional[dict[str, Any]] = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    trace: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]


class MetricsResponse(BaseModel):
    tenant_id: str
    spend_usd: float
    budget_usd: float
    llm_primary: str
    requests_total: int = 0
    error_rate: float = 0.0
    p95_latency_ms: float = 0.0


class DemoTokenResponse(BaseModel):
    token: str
    tenant_id: str
    expires_in: int


class MachineInfo(BaseModel):
    machine_id: str
    machine_type: str = ""
    manufacturer: str = ""
    model: str = ""
    iso_class: str = ""
    normal_ranges: dict[str, Any] = Field(default_factory=dict)
