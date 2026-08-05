# RD-Agent Terminal — Последовательность промптов (PR #1)

> **Версия:** 1.0  
> **Дата:** 2026-05-23  
> **Scope:** PR #1 — scaffold terminal + gateway + Bybit klines + chart  
> **План:** [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)  
> **Rules:** `.cursor/rules/terminal-*.mdc`, `gateway-python.mdc`  
> **Checkpoint:** `aa42a7b0`

---

## Как использовать

1. **Перед каждым промптом** — убедитесь, что Cursor rules активны (Project Rules в настройках).
2. **Выполняйте промпты строго по порядку** — каждый следующий зависит от предыдущего.
3. **После каждого промпта** — проверьте Gate (критерии перехода); не переходите, пока gate не пройден.
4. **Один промпт = один логический коммит** (или группа tightly related файлов).
5. **При отклонении от плана** — остановитесь и согласуйте; не расширяйте scope.

### Шаблон invocation

```
@docs/terminal/IMPLEMENTATION_PLAN.md @docs/terminal/PROMPT_SEQUENCE.md
[текст промпта из секции N]
```

---

## Фаза 0 — Подготовка (выполнено)

| # | Действие | Статус |
|---|----------|--------|
| 0.1 | Git checkpoint `aa42a7b0` | ✅ |
| 0.2 | IMPLEMENTATION_PLAN.md | ✅ |
| 0.3 | Cursor rules `.cursor/rules/` | ✅ |
| 0.4 | PROMPT_SEQUENCE.md (этот файл) | ✅ |

---

## Промпт 1 — Branch & Scaffold

### Цель
Создать ветку и каркас директорий без бизнес-логики.

### Промпт

```
Контекст: RD-Agent Terminal PR #1. Следуй .cursor/rules/terminal-project-scope.mdc и docs/terminal/IMPLEMENTATION_PLAN.md §14.2.

Задача:
1. Создай ветку feat/terminal-pr1-scaffold от текущего main.
2. Создай пустую структуру директорий:
   - gateway/app/{brokers,routers,models,services}
   - gateway/tests
   - terminal/src/{app,pages,components/{workspace,charts,ui},lib,stores}
3. Обнови .gitignore: terminal/node_modules, terminal/dist, gateway/.venv, gateway/__pycache__
4. НЕ создавай пока package.json, pyproject.toml, бизнес-код.
5. НЕ трогай rdagent/ и web/.

Gate: git status чистый; директории существуют; rdagent/ и web/ без изменений.
Коммит: chore(terminal): scaffold directory structure for PR #1
```

### Gate
- [ ] Branch `feat/terminal-pr1-scaffold` создан
- [ ] Директории на месте
- [ ] `.gitignore` обновлён
- [ ] `rdagent/`, `web/` не изменены

---

## Промпт 2 — Gateway Config & Models

### Цель
Конфигурация, Pydantic-модели, BrokerAdapter ABC.

### Промпт

```
Контекст: PR #1 Epic 2. Rules: gateway-python.mdc, terminal-api-contract.mdc, terminal-security.mdc.

Задача — gateway/ foundation:
1. gateway/pyproject.toml + requirements.txt (fastapi, uvicorn[standard], pybit, pydantic-settings, httpx, pytest)
2. gateway/app/config.py — Settings: gateway_host, gateway_port, cors_origins, bybit_testnet=True, bybit_api_key/secret
3. gateway/app/models/market.py — Symbol, OHLCVBar, Ticker, KlinesResponse, SymbolsResponse, HealthResponse
4. gateway/app/brokers/base.py — BrokerAdapter ABC + BrokerRegistry
5. gateway/app/services/risk_manager.py — stub class с docstrings (Phase 3), без logic
6. gateway/app/__init__.py, gateway/app/main.py — minimal FastAPI app + CORS, mount placeholder
7. Обнови .env.example: GATEWAY_PORT, BYBIT_TESTNET, BYBIT_API_KEY, BYBIT_API_SECRET

Сверь модели с docs/terminal/IMPLEMENTATION_PLAN.md §12.1.
НЕ реализуй BybitAdapter и routers пока.

Gate: pip install -r requirements.txt OK; python -c "from app.config import Settings" OK.
Коммит: feat(gateway): config, pydantic models, and broker adapter interface
```

### Gate
- [ ] Settings загружает env
- [ ] Модели соответствуют API contract
- [ ] BrokerAdapter ABC определён
- [ ] risk_manager — stub only

---

## Промпт 3 — BybitAdapter

### Цель
Полная реализация read-only Bybit market data.

### Промпт

```
Контекст: PR #1 Epic 2.4. Rules: gateway-python.mdc, terminal-security.mdc.

Задача — gateway/app/brokers/bybit.py:
1. BybitAdapter(BrokerAdapter) — pybit HTTP, category=linear, testnet from Settings
2. get_symbols() — get_instruments_info, filter status=Trading
3. get_klines(symbol, interval, limit) — get_kline, normalize OHLCVBar, time in seconds, sort asc
4. get_ticker(symbol) — get_tickers
5. Error handling: Bybit fail→502, bad symbol→404, rate limit→429
6. Register in BrokerRegistry as "bybit"
7. gateway/tests/test_bybit.py — unit tests с mocked pybit (без live API в CI)

НЕ добавляй place_order, positions, WebSocket.
BYBIT_TESTNET=true по умолчанию.

Gate: pytest gateway/tests/test_bybit.py pass.
Коммит: feat(gateway): implement BybitAdapter for read-only market data
```

### Gate
- [ ] Все три метода реализованы
- [ ] Tests pass с mocks
- [ ] Нет order/trade methods

---

## Промпт 4 — Gateway Routers & Health

### Цель
REST API endpoints + OpenAPI.

### Промпт

```
Контекст: PR #1 Epic 2.5. Rules: gateway-python.mdc, terminal-api-contract.mdc.

Задача:
1. gateway/app/routers/health.py — GET /api/v1/health
2. gateway/app/routers/market.py — GET /api/v1/market/symbols, /klines, /ticker
   - Query params: broker (default bybit), symbol, interval, limit, category
   - Resolve broker via BrokerRegistry
3. gateway/app/main.py — include routers under /api/v1, OpenAPI /docs
4. gateway/tests/test_market.py — test health + klines schema (mocked broker)

Запуск: uvicorn app.main:app --host 0.0.0.0 --port 6900 --reload

Gate (manual):
- curl http://localhost:6900/api/v1/health → 200
- curl "http://localhost:6900/api/v1/market/klines?symbol=BTCUSDT&interval=60&limit=10" → valid JSON bars
- http://localhost:6900/docs открывается

Коммит: feat(gateway): add health and market REST endpoints
```

### Gate
- [ ] 4 endpoints работают
- [ ] OpenAPI docs доступны
- [ ] Tests pass

---

## Промпт 5 — Terminal Scaffold & Theme

### Цель
Vite + React + TS + Tailwind + shadcn + providers.

### Промпт

```
Контекст: PR #1 Epic 3.1–3.2. Rules: terminal-react.mdc, terminal-project-scope.mdc.

Задача — terminal/ scaffold:
1. npm create vite@latest (react-ts) в terminal/
2. Dependencies: react-router-dom, @tanstack/react-query, zustand, react-grid-layout, lightweight-charts
3. Tailwind v4 + shadcn/ui (button, select, badge, separator)
4. Dark fintech theme: bg #0a0e17, amber accent, tabular-nums
5. src/app/providers.tsx — QueryClientProvider
6. src/app/router.tsx — / → CommandCenter placeholder
7. vite.config.ts — proxy /api → http://localhost:6900
8. terminal/.env.development — VITE_GATEWAY_URL=http://localhost:6900

НЕ реализуй chart и workspace пока — только «RD-Agent Terminal» placeholder page.

Gate: npm run dev → http://localhost:5173 loads, dark theme visible.
Коммит: feat(terminal): vite react scaffold with tailwind and shadcn
```

### Gate
- [ ] Dev server starts
- [ ] Theme applied
- [ ] Proxy configured

---

## Промпт 6 — API Client & Types

### Цель
Typed client, TanStack Query hooks.

### Промпт

```
Контекст: PR #1 Epic 3.3. Rules: terminal-api-contract.mdc.

Задача:
1. terminal/src/lib/types.ts — mirror gateway models (OHLCVBar, Symbol, Ticker, etc.)
2. terminal/src/lib/api.ts — fetchHealth, fetchSymbols, fetchKlines, fetchTicker
3. terminal/src/lib/format.ts — price formatting, percent, volume abbrev
4. terminal/src/hooks/useMarket.ts — useKlines, useTicker, useSymbols (TanStack Query)
   - useTicker staleTime: 5000
   - useKlines: refetch on symbol/interval change

Field names MUST match gateway Pydantic models exactly.
Handle fetch errors with typed ApiError.

Gate: with gateway running, hooks return data in React DevTools / test page.
Коммит: feat(terminal): typed API client and market data hooks
```

### Gate
- [ ] Types match contract
- [ ] Hooks fetch from gateway
- [ ] Error handling present

---

## Промпт 7 — CandlestickChart Component

### Цель
Lightweight Charts OHLC + volume, lifecycle-safe.

### Промпт

```
Контекст: PR #1 Epic 3.5. Rules: terminal-react.mdc. Q4=A Lightweight Charts only.

Задача — terminal/src/components/charts/CandlestickChart.tsx:
1. Props: bars: OHLCVBar[], loading, error, onRetry
2. createChart() on mount; candlestick + histogram (volume) series
3. Dark theme matching terminal (#0a0e17 background)
4. setData() when bars change; fitContent optional
5. ResizeObserver → chart.resize()
6. cleanup: remove chart on unmount
7. Loading skeleton, error state with retry button

НЕ используй ECharts/Recharts для свечей.
НЕ добавляй signal markers (Phase 2).

Gate: render with mock bars — chart visible, resize works, no console errors on unmount/remount.
Коммит: feat(terminal): candlestick chart with lightweight-charts
```

### Gate
- [ ] Chart renders mock data
- [ ] Resize works
- [ ] No memory leak on remount

---

## Промпт 8 — Workspace Shell & Store

### Цель
Grid layout, panels, persistence.

### Промпт

```
Контекст: PR #1 Epic 3.4. Rules: terminal-react.mdc.

Задача:
1. terminal/src/stores/workspaceStore.ts — layout, activeSymbol (BTCUSDT), activeInterval (60), save/load localStorage
2. terminal/src/components/workspace/Panel.tsx — title, collapse, children
3. terminal/src/components/workspace/StatusBar.tsx — gateway status, testnet badge, symbol, last price
4. terminal/src/components/workspace/WorkspaceShell.tsx — react-grid-layout default:
   - chart (large), ticker-info, agent-placeholder, execution-placeholder
5. Placeholder text: "Agent Console — Phase 2", "Execution Monitor — Phase 3"

Gate: panels drag/resize; layout persists after F5.
Коммит: feat(terminal): workspace shell with grid layout
```

### Gate
- [ ] Layout draggable/resizable
- [ ] localStorage persistence
- [ ] Placeholders visible

---

## Промпт 9 — CommandCenter Integration

### Цель
Собрать полную страницу: symbol/interval selectors + chart + ticker.

### Промпт

```
Контекст: PR #1 Epic 3.6. Rules: all terminal rules.

Задача — terminal/src/pages/CommandCenter.tsx:
1. Top bar: title, symbol Select (from useSymbols), interval Select (1m/5m/15m/1h/4h/1D)
2. WorkspaceShell with panels:
   - Chart → CandlestickChart(useKlines)
   - Ticker → 24h stats from useTicker
   - Agent/Execution placeholders
3. StatusBar with gateway health check (useQuery fetchHealth)
4. Default: BTCUSDT, 1h

End-to-end: gateway + terminal together.

Gate (manual QA §14.6.2 items 1-6):
- Chart shows live BTCUSDT klines
- Switch ETHUSDT → chart updates
- Switch 1h→4h → chart updates
- Gateway down → error state

Коммит: feat(terminal): command center with live bybit chart
```

### Gate
- [ ] E2E chart works with live gateway
- [ ] Symbol/interval switching works
- [ ] Error state on gateway down

---

## Промпт 10 — Docker & DevOps

### Цель
docker-compose, Dockerfile, DEVELOPMENT.md.

### Промпт

```
Контекст: PR #1 Epic 4. Rules: terminal-security.mdc.

Задача:
1. gateway/Dockerfile — python:3.11-slim, uvicorn
2. docker-compose.terminal.yml — gateway service port 6900, env_file .env; redis profile optional
3. docs/terminal/DEVELOPMENT.md:
   - Prerequisites (Node 20+, Python 3.10+)
   - Gateway setup (venv + uvicorn)
   - Terminal setup (npm run dev)
   - Bybit testnet key guide (optional for public klines)
   - Windows notes, port map, troubleshooting CORS
4. README.md — секция "RD-Agent Terminal" со ссылкой на DEVELOPMENT.md

Gate: docker compose -f docker-compose.terminal.yml up gateway — health OK.
Коммит: chore(terminal): docker compose and development guide
```

### Gate
- [ ] Docker gateway starts
- [ ] DEVELOPMENT.md complete
- [ ] README updated

---

## Промпт 11 — Final QA & PR Prep

### Цель
Полный QA, regression check, PR description.

### Промпт

```
Контекст: PR #1 Epic 5–6. Rules: terminal-testing-dod.mdc.

Задача:
1. Прогони gateway/tests/ — все pass
2. Manual QA checklist IMPLEMENTATION_PLAN.md §14.6.2 (items 1-9)
3. Verify rdagent/ и web/ — zero diff
4. Verify no secrets in git diff
5. Self-review: scope matches PR #1 only (§14.11 out-of-scope absent)
6. Prepare PR description:
   - Title: feat(terminal): PR #1 scaffold — React terminal + FastAPI gateway + Bybit klines
   - Summary bullets, test plan, screenshots note, breaking changes: none
7. Fix any issues found

НЕ создавай PR автоматически без запроса пользователя.
НЕ push без запроса.

Gate: all QA items pass; ready for review.
```

### Gate
- [ ] All tests pass
- [ ] Full manual QA pass
- [ ] No scope creep
- [ ] PR description drafted

---

## Сводная таблица промптов

| # | Промпт | Epic | Коммит prefix | Зависит от |
|---|--------|------|---------------|------------|
| 1 | Branch & Scaffold | 1 | chore | — |
| 2 | Gateway Config & Models | 2.1–2.3 | feat(gateway) | 1 |
| 3 | BybitAdapter | 2.4 | feat(gateway) | 2 |
| 4 | Gateway Routers | 2.5 | feat(gateway) | 3 |
| 5 | Terminal Scaffold | 3.1–3.2 | feat(terminal) | 1 |
| 6 | API Client & Types | 3.3 | feat(terminal) | 4, 5 |
| 7 | CandlestickChart | 3.5 | feat(terminal) | 5 |
| 8 | Workspace Shell | 3.4 | feat(terminal) | 5 |
| 9 | CommandCenter | 3.6 | feat(terminal) | 4, 6, 7, 8 |
| 10 | Docker & DevOps | 4 | chore | 4, 9 |
| 11 | Final QA & PR Prep | 5–6 | fix/chore | 10 |

**Параллелизация:** Промпты 5–7 могут начаться после 4 (не зависят от 9). Оптимально: 1→2→3→4, параллельно 5→7→8, затем 6→9→10→11.

---

## Anti-patterns — STOP if agent suggests

| Anti-pattern | Правильное действие |
|--------------|---------------------|
| Modify `rdagent/scenarios/qlib/` | Out of scope PR #1 |
| Modify `web/` Vue app | Out of scope PR #1 |
| Add Flask proxy now | Phase 2 |
| Implement place_order | Phase 3 |
| Use ECharts for candles | Lightweight Charts only |
| Eager Redis/Celery | Optional Phase 2 |
| Full auth system | Phase 3 |
| Tiger adapter | Phase 4 |
| Rewrite IMPLEMENTATION_PLAN mid-PR | Update only if API contract changes |

---

## Phase 2+ — Preview prompts (не выполнять сейчас)

<details>
<summary>Phase 2 — Agent Console (reference)</summary>

```
Bridge Flask server_ui via gateway/app/services/agent_bridge.py.
WebSocket /ws/agent/trace/{id}. Migrate Playground flow to AgentConsole.tsx.
qlib_reader for ret.pkl and qlib_res.csv. Deprecate Vue agent views.
```
</details>

<details>
<summary>Phase 3 — Execution + Risk (reference)</summary>

```
Order ticket UI. RiskManager enforcement. PaperAdapter.
POST /execution/orders with manual approval gate. Bybit WS for live P&L.
```
</details>

<details>
<summary>Phase 4 — Tiger + Multi-market (reference)</summary>

```
TigerAdapter for US/HK/CN equities. OpenBB optional widgets.
```
</details>

---

## Cursor Rules Index

| Rule file | Scope | alwaysApply |
|-----------|-------|-------------|
| `terminal-project-scope.mdc` | Global scope & boundaries | ✅ |
| `gateway-python.mdc` | `gateway/**` | — |
| `terminal-react.mdc` | `terminal/**` | — |
| `terminal-api-contract.mdc` | API models & lib | — |
| `terminal-security.mdc` | Secrets, testnet | — |
| `terminal-testing-dod.mdc` | QA, commits | — |

---

*Обновляйте версию при добавлении Phase 2+ prompt sequences.*
