# RD-Agent Terminal — Phase 2 Prompt Sequence

> **Plan:** [PHASE2_IMPLEMENTATION_PLAN.md](./PHASE2_IMPLEMENTATION_PLAN.md)  
> **Branch:** `feat/terminal-phase2-agent-research`  
> **Base:** `feat/terminal-pr1-scaffold` @ `a462f7ec`

---

## Prompt P2-1 — Branch & Gateway Config

```
Создай ветку feat/terminal-phase2-agent-research от feat/terminal-pr1-scaffold.
Расширь gateway/app/config.py: trace_folder, ui_server_port (=gateway_port), workspace_path, repo_root.
Добавь sys.path bootstrap в main.py для импорта rdagent.
Обнови .env.example: UI_TRACE_FOLDER, GATEWAY_PORT.
Коммит: chore(terminal): phase 2 branch and gateway config
```

## Prompt P2-2 — Agent Runner Service

```
Реализуй gateway/app/services/agent_runner.py — порт RDAgentTask из rdagent/log/server/app.py:
- scenario → target_name mapping (Finance Data Building → fin_factor, etc.)
- in-memory processes, /receive append, load persisted traces
- БЕЗ изменений rdagent/
Коммит: feat(gateway): native agent runner service
```

## Prompt P2-3 — Agent REST + /receive

```
gateway/app/models/agent.py, routers/agent.py:
GET /api/v1/agent/scenarios, POST /run (multipart), GET /traces, GET /trace/{id}, POST /control, POST /user-interaction/submit
POST /receive (WebStorage compat)
Tests: test_agent.py
Коммит: feat(gateway): agent REST endpoints
```

## Prompt P2-4 — WebSocket Trace

```
gateway/app/routers/ws.py — WS /ws/agent/trace/{trace_id}
Push new messages from agent_runner, detect END tag
Коммит: feat(gateway): websocket agent trace streaming
```

## Prompt P2-5 — Qlib Research Reader

```
gateway/app/services/trace_reader.py + qlib_reader.py + routers/research.py
GET /api/v1/research/experiments, /research/{id}/metrics, /research/{id}/returns
Tests: test_research.py
Коммит: feat(gateway): qlib research reader API
```

## Prompt P2-6 — Terminal Agent API & Hooks

```
terminal/src/lib/scenarios.ts, agentApi.ts, hooks/useAgent.ts, useAgentTrace.ts (WebSocket)
Коммит: feat(terminal): agent API client and websocket hook
```

## Prompt P2-7 — AgentConsole UI

```
terminal/src/components/agent/*, pages/AgentConsole.tsx
Scenario picker, run/stop, LoopTimeline, TraceMessageList
Коммит: feat(terminal): agent console UI
```

## Prompt P2-8 — Research Lab UI

```
recharts dependency, components/research/*, pages/ResearchLab.tsx, hooks/useResearch.ts
MetricsTable, EquityCurveChart
Коммit: feat(terminal): research lab UI
```

## Prompt P2-9 — Integration & Overlay

```
CommandCenter tabs (Market | Agent | Research), chart markers from returns API
Router updates, DEVELOPMENT.md Phase 2 section
Коммит: feat(terminal): integrate agent and research into command center
```

## Prompt P2-10 — QA

```
pytest gateway/tests/, npm run build, manual checklist PHASE2 §9
Verify rdagent/ web/ unchanged
Коммит: docs(terminal): phase 2 completion notes (if needed)
```

---

## Gate Matrix

| Prompt | Gate |
|--------|------|
| P2-1 | Settings load trace_folder |
| P2-2 | Process starts fin_factor subprocess |
| P2-3 | POST /run returns trace id |
| P2-4 | WS receives messages |
| P2-5 | GET /metrics returns loops array |
| P2-6 | Hooks compile |
| P2-7 | AgentConsole renders |
| P2-8 | Equity curve renders mock/live data |
| P2-9 | Navigation works, chart overlay |
| P2-10 | All tests pass |
