import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.models.execution import OrderRequest, OrderResponse, PnLSnapshot, Position
from app.services.execution_service import execution_service

router = APIRouter(prefix="/execution", tags=["execution"])


@router.get("/status")
async def execution_status() -> dict:
    return await execution_service.get_status()


@router.post("/orders", response_model=OrderResponse)
async def place_order(body: OrderRequest) -> OrderResponse:
    try:
        return await execution_service.submit_order(body)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str, symbol: str, category: str = "linear") -> dict[str, str]:
    try:
        return await execution_service.cancel_order(symbol=symbol, order_id=order_id, category=category)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/positions", response_model=list[Position])
async def list_positions(category: str = "linear") -> list[Position]:
    try:
        return await execution_service.get_positions(category=category)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/pnl", response_model=PnLSnapshot)
async def get_pnl(category: str = "linear") -> PnLSnapshot:
    try:
        return await execution_service.get_pnl_snapshot(category=category)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/kill-switch/activate")
async def activate_kill_switch(reason: str = "Manual kill switch") -> dict[str, str]:
    execution_service.risk.activate_kill_switch(reason)
    await execution_service._broadcast_pnl()
    return {"status": "active", "reason": reason}


@router.post("/kill-switch/deactivate")
async def deactivate_kill_switch() -> dict[str, str]:
    execution_service.risk.deactivate_kill_switch()
    await execution_service._broadcast_pnl()
    return {"status": "inactive"}


@router.websocket("/ws/pnl")
async def pnl_websocket(websocket: WebSocket, category: str = "linear") -> None:
    await websocket.accept()
    queue = execution_service.subscribe_pnl()
    try:
        snapshot = await execution_service.get_pnl_snapshot(category=category)
        await websocket.send_json(snapshot.model_dump())
        while True:
            try:
                update = await asyncio.wait_for(queue.get(), timeout=2.0)
                await websocket.send_json(update.model_dump())
            except asyncio.TimeoutError:
                snapshot = await execution_service.get_pnl_snapshot(category=category)
                await websocket.send_json(snapshot.model_dump())
    except WebSocketDisconnect:
        pass
    finally:
        execution_service.unsubscribe_pnl(queue)
