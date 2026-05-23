# RD-Agent Terminal — Phase 2: Agent Console + Research Lab

> **Версия:** 1.0  
> **Дата:** 2026-05-23  
> **Предусловие:** Phase 1 (PR #1) завершён на ветке `feat/terminal-pr1-scaffold`  
> **Branch:** `feat/terminal-phase2-agent-research`

---

## 1. Executive Summary

Phase 2 соединяет **RD-Agent LLM loop** с React Terminal и добавляет **Research Lab** для qlib-метрик — без изменений `rdagent/` core и `web/` Vue.

| Зона | Phase 1 | Phase 2 |
|------|---------|---------|
| Market Chart (Bybit) | ✅ | ✅ сохраняется |
| Agent Console | placeholder | **live trace, run/stop, scenarios** |
| Research Lab | placeholder | **metrics, equity curve, loop table** |
| Execution | placeholder | placeholder (Phase 3) |

**Ключевое архитектурное решение Phase 2:** встроить agent orchestration в **gateway** (порт 6900), а не требовать отдельный Flask `server_ui`. WebStorage `/receive` обслуживается gateway для совместимости с rdagent logging.

---

## 2. As-Is после Phase 1

```
terminal/  → CommandCenter, Bybit chart, workspace
gateway/   → FastAPI, BybitAdapter, /api/v1/market/*
rdagent/   → RDAgentTask в Flask app.py (19899), FileStorage traces
web/       → Vue Playground (legacy)
```

**Trace flow (Flask today):**
```
POST /upload → RDAgentTask subprocess → WebStorage POST /receive → task.messages
POST /trace  → incremental poll → Vue PlaygroundPage
```

**Qlib artifacts в trace:**
| Tag / source | Content |
|--------------|---------|
| `feedback.metric` | `result` JSON (IC, ARR, MDD, …) |
| `feedback.hypothesis_feedback` | decision, reason, hypothesis |
| `feedback.return_chart` | Plotly HTML (legacy) |
| Pickle: `Quantitative Backtesting Chart` | pandas DataFrame (ret report) |
| Workspace files | `qlib_res.csv`, `ret.pkl` |

---

## 3. To-Be Architecture (Phase 2)

```mermaid
flowchart TB
    subgraph Terminal
        CC[CommandCenter]
        AC[AgentConsole]
        RL[ResearchLab]
        CH[CandlestickChart + markers]
    end

    subgraph Gateway6900
        AR[agent_runner]
        TR[trace_reader]
        QR[qlib_reader]
        WS["/ws/agent/trace"]
        REC["/receive WebStorage"]
    end

    subgraph RDAgent
        Loop[fin_factor | fin_model | fin_quant]
        FS[FileStorage pickles]
    end

    AC -->|REST + WS| Gateway6900
    RL -->|REST| QR
    CC --> AC
    CC --> RL
    AR --> Loop
    Loop -->|WebStorage| REC
    TR --> FS
    QR --> TR
    CH -->|overlay| QR
```

---

## 4. Scope

### 4.1 In Scope

- Native agent runner в gateway (порт RDAgentTask, без обязательного Flask)
- `POST /api/v1/agent/run`, `GET /traces`, `GET /trace/{id}`, `POST /control`, `POST /user-interaction/submit`
- `POST /receive` — WebStorage compat
- `WS /ws/agent/trace/{trace_id}` — push новых сообщений
- `GET /api/v1/research/experiments`, `/research/{trace_id}/metrics`, `/research/{trace_id}/returns`
- Terminal: AgentConsole, ResearchLab, navigation, Recharts analytics
- Signal overlay markers на Lightweight Charts из returns (упрощённо: rebalance points)
- Docs + Phase 2 prompt sequence

### 4.2 Out of Scope (Phase 3+)

- Order placement / Bybit execution
- Full Vue removal (только deprecation notice в docs)
- Data Science scenario в terminal (Streamlit only)
- Auth / multi-user
- Изменения `rdagent/`, `web/`

---

## 5. API Specification (Phase 2)

### 5.1 Agent

```
GET  /api/v1/agent/scenarios
POST /api/v1/agent/run          multipart: scenario, loops, all_duration, files[]
GET  /api/v1/agent/traces
GET  /api/v1/agent/trace/{id}?offset=0&limit=50
POST /api/v1/agent/control      { id, action: "stop" }
POST /api/v1/agent/user-interaction/submit  { id, payload }
WS   /ws/agent/trace/{trace_id}
POST /receive                   WebStorage (rdagent compat)
```

### 5.2 Research

```
GET /api/v1/research/experiments
    → [{ traceId, scenario, traceName, loopCount, lastTimestamp }]

GET /api/v1/research/{trace_id}/metrics
    → { loops: [{ loopId, metrics: {...}, decision, hypothesis }] }

GET /api/v1/research/{trace_id}/returns?loop_id=0
    → { points: [{ time, bench, strategy, excess }], markers: [{ time, type }] }
```

---

## 6. Gateway Modules

```
gateway/app/
  services/
    agent_runner.py    # RDAgentTask, processes dict, scenario map
    trace_reader.py    # FileStorage → normalized messages
    qlib_reader.py     # metrics + returns from trace
  routers/
    agent.py
    research.py
    ws.py
  models/
    agent.py
    research.py
```

**Repo path bootstrap:** `sys.path.insert(0, repo_root)` в `main.py` для импорта `rdagent.*`.

**New dependencies:** `pandas`, `aiofiles` (optional)

---

## 7. Terminal Modules

```
terminal/src/
  pages/
    AgentConsole.tsx
    ResearchLab.tsx
  components/agent/
    ScenarioPicker.tsx
    LoopTimeline.tsx
    TraceMessageList.tsx
  components/research/
    MetricsTable.tsx
    EquityCurveChart.tsx
  hooks/
    useAgent.ts
    useAgentTrace.ts      # WebSocket
    useResearch.ts
  lib/
    scenarios.ts
    agentApi.ts
```

**Router:**
- `/` — CommandCenter (tabs: Market | Agent | Research)
- `/agent` — AgentConsole full page
- `/research` — ResearchLab

---

## 8. Epics & Tasks

### Epic 1 — Gateway Agent Runner (3d)
- 1.1 Config: `trace_folder`, `ui_server_port`, `workspace_path`
- 1.2 Port `RDAgentTask` + scenario mapping from Flask
- 1.3 `/receive`, load persisted traces on startup
- 1.4 Agent REST routes + tests

### Epic 2 — WebSocket Trace (1d)
- 2.1 WS manager, subscribe by trace_id
- 2.2 Push on new /receive messages + END detection

### Epic 3 — Qlib Research Reader (2d)
- 3.1 `trace_reader` via FileStorage + WebStorage._obj_to_json
- 3.2 Extract feedback.metric, hypothesis, ret DataFrame
- 3.3 Research REST routes + tests

### Epic 4 — Terminal Agent UI (3d)
- 4.1 scenarios.ts, agentApi, useAgent hooks
- 4.2 AgentConsole: scenario select, run, stop, live timeline
- 4.3 WebSocket integration

### Epic 5 — Terminal Research UI (2d)
- 5.1 MetricsTable, EquityCurveChart (Recharts)
- 5.2 ResearchLab page, trace picker
- 5.3 Chart markers overlay

### Epic 6 — Integration & Docs (1d)
- 6.1 CommandCenter navigation tabs
- 6.2 DEVELOPMENT.md update, Vue deprecation note
- 6.3 QA checklist

---

## 9. Definition of Done

- [ ] Agent run fin_factor from terminal (loops=1) shows live trace
- [ ] Stop works via API
- [ ] Historical trace loads metrics + equity curve
- [ ] WebSocket delivers messages without polling
- [ ] Bybit chart still works (no regression)
- [ ] `rdagent/`, `web/` zero diff
- [ ] Gateway tests pass

---

## 10. Risk & Mitigations

| Risk | Mitigation |
|------|------------|
| rdagent import fails in gateway | sys.path + document `pip install -e .` from repo root |
| Long agent runs block gateway | subprocess (existing RDAgentTask pattern) |
| ret.pkl path unknown | Parse from trace pickles, not filesystem scan |
| Windows qlib docker | Document WSL2; agent UI works without local qlib run |

---

## 11. Rollback

Checkpoint before Phase 2: last commit on `feat/terminal-pr1-scaffold` (`a462f7ec`).

```bash
git checkout feat/terminal-pr1-scaffold
# or
git checkout -b rollback-phase1 a462f7ec
```
