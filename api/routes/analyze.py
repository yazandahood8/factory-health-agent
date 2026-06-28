"""Analysis endpoints — the full agent pipeline behind one POST."""
from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.schemas import AnalyzeRequest, AnalyzeResponse
from api.serialization import to_jsonable
from sdk.exceptions import BudgetExceededException

router = APIRouter(prefix="/v1", tags=["analyze"])


def _run(request: Request, body: AnalyzeRequest):
    pipeline = request.app.state.pipeline
    tenant = request.state.tenant
    sensor = body.sensor_data.model_dump(exclude_none=True) if body.sensor_data else None
    return pipeline.run(body.machine_id, tenant, sensor)


def _to_response(state) -> AnalyzeResponse:
    report = state.get("anomaly_report")
    recorder = state.get("recorder")
    execution_trace = recorder.snapshot() if recorder is not None else []
    # Per-request cost from this run's LLM calls (not cumulative tenant spend).
    request_cost = recorder.llm_cost() if recorder is not None else state.get("total_cost_usd", 0.0)
    return AnalyzeResponse(
        machine_id=state["machine_id"],
        trace_id=state["trace_id"],
        severity=report.severity.value if report else "UNKNOWN",
        escalated=state.get("escalated", False),
        anomaly_report=to_jsonable(report),
        diagnosis=to_jsonable(state.get("diagnosis")),
        action_plan=to_jsonable(state.get("action_plan")),
        validated_response=to_jsonable(state.get("validated_response")),
        total_tokens=state.get("total_tokens", 0),
        total_cost_usd=round(request_cost, 6),
        trace=state.get("messages", []),
        execution_trace=execution_trace,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, body: AnalyzeRequest):
    from fastapi import HTTPException

    try:
        state = _run(request, body)
    except BudgetExceededException as exc:
        raise HTTPException(status_code=402, detail=str(exc))
    return _to_response(state)


@router.post("/analyze/stream")
async def analyze_stream(request: Request, body: AnalyzeRequest):
    """SSE stream of pipeline progress then the final result."""

    def event_gen():
        try:
            state = _run(request, body)
        except BudgetExceededException as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return
        for msg in state.get("messages", []):
            yield f"event: step\ndata: {json.dumps({'message': msg})}\n\n"
        final = _to_response(state).model_dump()
        yield f"event: result\ndata: {json.dumps(final)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
