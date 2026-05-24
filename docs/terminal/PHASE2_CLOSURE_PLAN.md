# Phase 2 — Закрытие DoD: анализ, промпты, план

> **Ветка:** `feat/terminal-phase2-agent-research`  
> **Базовый план:** [PHASE2_IMPLEMENTATION_PLAN.md](./PHASE2_IMPLEMENTATION_PLAN.md)

---

## 1. Анализ оставшихся пунктов DoD

| # | Item | Текущий статус | Блокер | Решение | Приоритет |
|---|------|----------------|--------|---------|-----------|
| 1 | **Live fin_factor E2E** | Agent API готов, UI готов | LLM keys + qlib Docker/WSL2 | Чеклист + smoke script; ручной E2E 1 loop | P0 |
| 2 | **Signal overlay на chart** | API `markers[]` есть | Qlib dates ≠ Bybit unix time | Markers на **EquityCurveChart** (Recharts); Market chart — без overlay | P1 |
| 3 | **Vue deprecation redirect** | Только docs | — | Banner в `web/` + README link | P1 |
| 4 | **Chart markers из returns** | 1 marker (last rebalance) | Partial parser | Расширить qlib_reader: rebalance + drawdown markers | P1 |
| 5 | **PR Phase 2** | Код на ветке | — | `gh pr create` | P0 |
| 6 | **Phase 3** | Не начат | — | Отдельный plan + prompts | P2 |

### 1.1 Live fin_factor E2E — детали

**Что уже работает:**
- `POST /api/v1/agent/run` → subprocess `fin_factor`
- `POST /receive` ← WebStorage
- `WS /api/v1/agent/ws/trace/{id}`
- AgentConsole UI: scenario, loops, stop

**Что нужно для E2E:**
```env
OPENAI_API_KEY=...
CHAT_MODEL=gpt-4o
MODEL_CoSTEER_env_type=docker   # WSL2/Linux
# ~/.qlib/qlib_data/cn_data
pip install -e .                # из корня RD-Agent
```

**Критерий успеха E2E:**
1. Run `Finance Data Building`, loops=1
2. Trace ID появляется в Agent Console
3. WS/poll получает tags: `research.hypothesis`, `feedback.metric`, `END`
4. Research Lab показывает metrics + equity curve

### 1.2 Signal overlay — архитектурное ограничение

| Chart | Data source | Marker time format | Overlay feasible? |
|-------|-------------|-------------------|-------------------|
| Bybit CandlestickChart | pybit klines | Unix seconds | ❌ для qlib CSI300 |
| EquityCurveChart | ret.pkl via qlib_reader | Date strings (T0, 2020-…) | ✅ |

**Решение:** Phase 2 DoD для overlay = markers на **Research equity curve**, не на crypto market chart. Phase 3+ может добавить unified symbol mapping.

### 1.3 Vue deprecation

Минимальный professional approach:
- Banner в Vue Playground: «New Terminal UI → http://localhost:5173»
- README: terminal = primary UI для quant scenarios
- Flask `server_ui` остаётся для backward compat

---

## 2. Prompt Sequence — закрытие Phase 2 (P2-C)

### P2-C1 — Equity curve markers (Recharts)

```
Добавь markers на EquityCurveChart из returnsQuery.markers.
ReferenceDot или Scatter на time/strategy.
ResearchLab передаёт markers в chart.
Gate: при наличии trace с ret.pkl видны точки rebalance.
Коммит: feat(terminal): equity curve markers from research returns
```

### P2-C2 — Расширить qlib_reader markers

```
В gateway/app/services/qlib_reader.py:
- markers для каждого 5-го rebalance point в report index
- marker type: rebalance | period_end
- unit test с mock DataFrame
Коммит: feat(gateway): enrich returns markers from qlib report
```

### P2-C3 — Vue deprecation banner

```
web/src/views/Playground.vue — banner вверху:
"RD-Agent Terminal (React) is the primary UI → http://localhost:5173"
Не ломать existing flow.
Коммит: chore(web): deprecation banner pointing to React terminal
```

### P2-C4 — E2E checklist

```
Создай docs/terminal/E2E_AGENT_RUN.md:
- prerequisites (.env, docker, qlib data)
- step-by-step Finance Data Building 1 loop
- expected tags timeline
- troubleshooting
Коммит: docs(terminal): E2E agent run checklist
```

### P2-C5 — PR Phase 2

```
gh pr create из feat/terminal-phase2-agent-research → main
Title: feat(terminal): Phase 2 agent console and research lab
Body: summary, test plan, breaking changes none
```

---

## 3. Phase 3 — Preview (Execution + Risk)

| Epic | Deliverable |
|------|-------------|
| 3.1 | Order ticket UI (Bybit testnet) |
| 3.2 | RiskManager enforcement |
| 3.3 | PaperAdapter |
| 3.4 | Manual approval gate signal→order |
| 3.5 | Live P&L WebSocket |

**Branch:** `feat/terminal-phase3-execution`  
**Prompt doc:** `PROMPT_SEQUENCE_PHASE3.md` (создать при старте Phase 3)

---

## 4. Definition of Done — финальный чеклист Phase 2

- [ ] P2-C1 Equity markers UI
- [ ] P2-C2 Gateway markers enriched
- [ ] P2-C3 Vue banner
- [ ] P2-C4 E2E doc
- [ ] Manual E2E fin_factor 1 loop (user + keys)
- [ ] P2-C5 PR merged or open
- [ ] Gateway tests 11/11
- [ ] Terminal build OK

---

## 5. Рекомендуемый порядок выполнения

```
P2-C2 → P2-C1 → P2-C3 → P2-C4 → (manual E2E) → P2-C5 → Phase 3 planning
```
