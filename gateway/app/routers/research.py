from fastapi import APIRouter, HTTPException

from app.models.research import ExperimentSummary, MetricsResponse, ReturnsResponse
from app.services import qlib_reader

router = APIRouter(prefix="/research", tags=["research"])


@router.get("/experiments", response_model=list[ExperimentSummary])
async def list_experiments() -> list[ExperimentSummary]:
    return [ExperimentSummary(**item) for item in qlib_reader.list_experiments()]


@router.get("/{trace_id:path}/metrics", response_model=MetricsResponse)
async def get_metrics(trace_id: str) -> MetricsResponse:
    try:
        payload = qlib_reader.get_metrics(trace_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MetricsResponse(**payload)


@router.get("/{trace_id:path}/returns", response_model=ReturnsResponse)
async def get_returns(trace_id: str, loop_id: int | None = None) -> ReturnsResponse:
    try:
        payload = qlib_reader.get_returns(trace_id, loop_id=loop_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ReturnsResponse(**payload)
