"""Analysis endpoints — the full agent pipeline behind one POST."""
from __future__ import annotations

import json
import queue
import threading

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.schemas import AnalyzeRequest, AnalyzeResponse
from api.serialization import to_jsonable
from sdk.exceptions import BudgetExceededException
from sdk.trace import TraceRecorder

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
    """Server-Sent Events: emits each pipeline step **as it happens**, then the
    final result. The pipeline runs in a worker thread; the trace recorder pushes
    every event onto a queue that this generator drains in real time.
    """
    pipeline = request.app.state.pipeline
    tenant = request.state.tenant
    sensor = body.sensor_data.model_dump(exclude_none=True) if body.sensor_data else None

    q: "queue.Queue" = queue.Queue()
    recorder = TraceRecorder(sink=lambda ev: q.put(("trace", ev)))

    def worker():
        try:
            state = pipeline.run(body.machine_id, tenant, sensor, recorder=recorder)
            q.put(("result", _to_response(state).model_dump()))
        except BudgetExceededException as exc:
            q.put(("error", {"detail": str(exc)}))
        except Exception as exc:  # surface unexpected failures to the client
            q.put(("error", {"detail": f"{type(exc).__name__}: {exc}"}))
        finally:
            q.put(None)  # sentinel: stream complete

    def event_gen():
        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
