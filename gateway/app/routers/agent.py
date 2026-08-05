import asyncio
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.models.agent import AgentControlRequest, AgentRunResponse, UserInteractionRequest
from app.services.agent_runner import agent_runner

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/scenarios")
async def list_scenarios() -> list[dict[str, Any]]:
    return agent_runner.list_scenarios()


@router.get("/traces")
async def list_traces() -> list[str]:
    return agent_runner.list_trace_ids()


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    scenario: str = Form(...),
    loops: int | None = Form(default=None),
    all_duration: str | None = Form(default=None),
    files: list[UploadFile] = File(default=[]),
) -> AgentRunResponse:
    try:
        trace_id = await agent_runner.start_run(scenario, loops, all_duration, files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AgentRunResponse(id=trace_id)


@router.get("/trace/{trace_id:path}")
async def get_trace(trace_id: str, offset: int = 0, limit: int = 50, all: bool = False) -> list[dict]:
    try:
        return agent_runner.get_trace_messages(trace_id, offset=offset, limit=limit, return_all=all)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/control")
async def control_agent(body: AgentControlRequest) -> dict[str, str]:
    if body.action != "stop":
        raise HTTPException(status_code=400, detail="Only stop action is supported")
    try:
        agent_runner.stop_trace(body.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "stopped"}


@router.post("/user-interaction/submit")
async def submit_user_interaction(body: UserInteractionRequest) -> dict[str, str]:
    try:
        agent_runner.submit_user_interaction(body.id, body.payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "success"}


@router.websocket("/ws/trace/{trace_id:path}")
async def trace_websocket(websocket: WebSocket, trace_id: str) -> None:
    await websocket.accept()
    seen = 0
    try:
        while True:
            messages = agent_runner.get_trace_messages(trace_id, return_all=True)
            if len(messages) > seen:
                for msg in messages[seen:]:
                    await websocket.send_json(msg)
                seen = len(messages)
            if messages and messages[-1].get("tag") == "END":
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
