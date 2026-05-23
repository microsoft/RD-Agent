from typing import Any

from pydantic import BaseModel


class ExperimentSummary(BaseModel):
    traceId: str
    scenario: str
    traceName: str
    loopCount: int
    messageCount: int
    lastTimestamp: str | None = None


class LoopMetrics(BaseModel):
    loopId: int
    metrics: dict[str, Any]
    hypothesis: str | None = None
    decision: bool | None = None


class MetricsResponse(BaseModel):
    traceId: str
    loops: list[LoopMetrics]


class ReturnPoint(BaseModel):
    time: str
    bench: float
    strategy: float
    excess: float


class ReturnMarker(BaseModel):
    time: str
    type: str


class ReturnsResponse(BaseModel):
    traceId: str
    loopId: int | None = None
    points: list[ReturnPoint]
    markers: list[ReturnMarker]
