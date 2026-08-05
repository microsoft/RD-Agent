# E2E Agent Run — Finance Data Building (1 loop)

Manual end-to-end validation for Phase 2 agent integration.

## Prerequisites

| Requirement | Check |
|-------------|-------|
| Python 3.10+ | `python --version` |
| RD-Agent installed | `pip install -e .` from repo root |
| LLM keys in `.env` | `OPENAI_API_KEY`, `CHAT_MODEL` |
| Qlib data (cn) | `~/.qlib/qlib_data/cn_data` or Docker |
| Qlib env | `MODEL_CoSTEER_env_type=docker` (WSL2/Linux) |
| Gateway deps | `pip install -r gateway/requirements.txt` |

## Start services

```powershell
# Terminal 1 — Gateway
cd gateway
uvicorn app.main:app --host 0.0.0.0 --port 6900 --reload

# Terminal 2 — React UI
cd terminal
npm run dev
```

Open http://localhost:5173 → **Agent Console** tab.

## Run test

1. Scenario: **Finance Data Building**
2. Loops: **1**
3. Click **Run Agent**
4. Observe trace ID in history panel
5. WebSocket / poll should show tags in order:
   - `research.hypothesis`
   - `research.experiment`
   - `feedback.metric`
   - `END` (or loop completion)

## Verify Research Lab

1. Switch to **Research Lab** tab
2. Select the new trace in Experiments list
3. **Qlib Metrics** table shows loop metrics (IC, annualized return, etc.)
4. **Equity Curve** shows bench/strategy/excess lines
5. **Amber dots** = rebalance markers from `ret.pkl`

## API smoke (optional)

```powershell
# List experiments
curl http://localhost:6900/api/v1/research/experiments

# Returns + markers (replace TRACE_ID)
curl http://localhost:6900/api/v1/research/experiments/TRACE_ID/returns
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: rdagent` | `pip install -e .` from repo root |
| Agent subprocess exits immediately | Check `.env` LLM keys; gateway logs |
| No experiments in Research Lab | Wait for trace to finish; check `trace_folder` in gateway config |
| Empty equity curve | Trace may lack `Quantitative Backtesting Chart` pickle |
| Qlib docker errors | WSL2 + Docker; see main README qlib section |

## Success criteria

- [ ] Agent run completes 1 loop without crash
- [ ] Trace visible in Agent Console history
- [ ] Research Lab shows metrics for trace
- [ ] Equity curve renders with markers
- [ ] Gateway tests pass: `pytest gateway/tests -q`
