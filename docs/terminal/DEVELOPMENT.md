# RD-Agent Terminal — Development Guide

## Prerequisites

| Tool | Version |
|------|---------|
| Node.js | 20+ |
| Python | 3.10+ |
| npm | 10+ |
| Docker Desktop | optional (gateway container) |

## Quick Start (local)

### 1. Gateway (FastAPI)

```powershell
cd gateway
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 6900 --reload
```

Verify:

```powershell
curl http://localhost:6900/api/v1/health
curl "http://localhost:6900/api/v1/market/klines?symbol=BTCUSDT&interval=60&limit=10"
```

OpenAPI docs: http://localhost:6900/docs

### 2. Terminal (React)

```powershell
cd terminal
npm install
npm run dev
```

Open: http://localhost:5173

Vite proxies `/api` to `http://localhost:6900`.

## Environment

Copy root `.env.example` to `.env` and set:

```env
GATEWAY_PORT=6900
BYBIT_TESTNET=true
BYBIT_API_KEY=
BYBIT_API_SECRET=
```

Public klines/tickers work **without** API keys on Bybit testnet.

## Docker (gateway only)

```powershell
docker compose -f docker-compose.terminal.yml up --build gateway
```

## Port Map

| Service | Port |
|---------|------|
| Terminal (Vite) | 5173 |
| Gateway (FastAPI) | 6900 |
| Flask server_ui (legacy) | 19899 |

## Windows + qlib (Phase 2+)

RD-Agent quant scenarios (`fin_factor`, `fin_model`) use qlib via Docker/WSL2.
Terminal PR #1 does not modify qlib execution.

- Use WSL2 Ubuntu for `local_qlib` Docker image
- Set `MODEL_CoSTEER_env_type=docker` in `.env`
- Mount `~/.qlib` for CN market data

## Troubleshooting

### CORS errors
Ensure gateway `cors_origins` includes `http://localhost:5173`.

### Chart empty / API errors
1. Confirm gateway is running on port 6900
2. Check `/api/v1/health` returns `"status": "ok"`
3. Test klines endpoint directly in browser

### Bybit 502 errors
Upstream Bybit API may be rate-limited or unavailable. Retry after a few seconds.

## Phase 2 — Agent Console + Research Lab

Requires `pip install -e .` from repo root so gateway can import `rdagent`.

Agent runs are orchestrated by gateway (`/api/v1/agent/*`) with WebSocket trace streaming.
Research metrics are read from trace pickles via `/api/v1/research/*`.

Legacy Vue UI (`web/`, `rdagent server_ui`) remains available but terminal is the primary UI for Phase 2.

