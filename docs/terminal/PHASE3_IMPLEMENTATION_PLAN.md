# Phase 3 — Execution + Risk (Bybit)

> **Prerequisite:** Phase 2 merged (agent + research lab)  
> **Branch:** `feat/terminal-phase3-execution`  
> **Scope:** Bybit testnet first; Tiger Trade in Phase 4

---

## 1. Goals

| Goal | Description |
|------|-------------|
| Order placement | Market/limit orders via Bybit testnet |
| Risk gates | Max position, daily loss, order size limits |
| Paper trading | Simulated fills before live testnet |
| Manual approval | Research signal → user confirms → order |
| Live P&L | Positions + unrealized P&L in terminal |

---

## 2. Architecture

```
terminal/OrderTicket.tsx
    ↓ POST /api/v1/execution/orders
gateway/routers/execution.py
    ↓
RiskManager.check(order) → PaperAdapter | BybitAdapter.place_order()
    ↓
WebSocket /api/v1/execution/ws/pnl
```

---

## 3. Gateway deliverables

| File | Purpose |
|------|---------|
| `gateway/app/services/risk_manager.py` | Rules engine: max_qty, max_notional, daily_loss |
| `gateway/app/services/paper_adapter.py` | In-memory fills, slippage model |
| `gateway/app/brokers/bybit.py` | Extend: `place_order`, `cancel_order`, `get_positions` |
| `gateway/app/routers/execution.py` | REST orders + WS P&L |
| `gateway/app/models/execution.py` | OrderRequest, OrderResponse, Position |

---

## 4. Terminal deliverables

| File | Purpose |
|------|---------|
| `terminal/src/pages/ExecutionDesk.tsx` | Order ticket, positions, P&L |
| `terminal/src/hooks/useExecution.ts` | TanStack Query + WS |
| `CommandCenter.tsx` | Tab: Execution Desk |
| `terminal/src/components/execution/OrderForm.tsx` | Symbol, side, qty, type |
| `terminal/src/components/execution/RiskBanner.tsx` | Blocked reasons from RiskManager |

---

## 5. Prompt Sequence Phase 3

### P3-1 — RiskManager + models

```
Create gateway/app/services/risk_manager.py and models/execution.py.
Rules: MAX_ORDER_NOTIONAL, MAX_POSITION_USD, DAILY_LOSS_LIMIT from env.
Unit tests with pytest.
```

### P3-2 — Bybit order methods

```
Extend BybitAdapter: place_order, cancel_order, get_positions.
Testnet only; read API keys from config.
Mock tests in gateway/tests/test_bybit_orders.py.
```

### P3-3 — PaperAdapter

```
PaperAdapter implements same interface as live broker.
Simulated fill at mid price; store in memory dict.
```

### P3-4 — Execution router

```
POST /api/v1/execution/orders — validate → risk → paper|live.
GET /api/v1/execution/positions
WS /api/v1/execution/ws/pnl — push on fill/ticker.
```

### P3-5 — Order ticket UI

```
ExecutionDesk page: symbol from market store, side, qty, submit.
Show risk rejection message inline.
TanStack mutation for POST orders.
```

### P3-6 — Manual signal → order (MVP)

```
ResearchLab: "Use as signal" button → prefills ExecutionDesk symbol + side hint.
No auto-trade; user must confirm.
```

### P3-7 — Docker + docs

```
docker-compose.terminal.yml: execution env vars.
DEVELOPMENT.md: Bybit testnet API key setup.
```

### P3-8 — QA

```
pytest gateway/tests
npm run build
Manual: paper order BTCUSDT testnet
```

---

## 6. Phase 3 DoD

- [x] Paper order flow E2E on testnet symbol
- [x] RiskManager blocks oversize orders
- [x] Positions visible in UI
- [x] P&L updates via WebSocket
- [x] No changes to `rdagent/` core
- [x] Tests green; terminal build OK

---

## 7. Out of scope (Phase 4)

- Tiger Trade adapter
- Auto-trading from agent signals without approval
- Multi-account / portfolio optimization
- Production mainnet keys
