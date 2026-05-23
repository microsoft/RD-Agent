from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.agent_runner import agent_runner, read_trace_from_disk


def list_experiments() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for trace_id in agent_runner.list_trace_ids():
        messages = agent_runner.get_trace_messages(trace_id, return_all=True)
        parts = trace_id.split("/", 1)
        scenario = parts[0] if parts else trace_id
        trace_name = parts[1] if len(parts) > 1 else trace_id
        loop_ids = {m.get("loop_id") for m in messages if m.get("loop_id") is not None}
        last_ts = messages[-1].get("timestamp") if messages else None
        experiments.append(
            {
                "traceId": trace_id,
                "scenario": scenario,
                "traceName": trace_name,
                "loopCount": len(loop_ids),
                "messageCount": len(messages),
                "lastTimestamp": last_ts,
            }
        )
    return experiments


def get_metrics(trace_id: str) -> dict[str, Any]:
    messages = _load_messages(trace_id)
    loops: dict[int, dict[str, Any]] = {}

    for msg in messages:
        loop_id = msg.get("loop_id")
        if loop_id is None:
            continue
        entry = loops.setdefault(loop_id, {"loopId": loop_id, "metrics": {}, "hypothesis": None, "decision": None})

        if msg.get("tag") == "feedback.metric":
            raw = msg.get("content", {}).get("result")
            if raw:
                try:
                    entry["metrics"] = json.loads(raw)
                except json.JSONDecodeError:
                    entry["metrics"] = {"raw": raw}
        if msg.get("tag") == "feedback.hypothesis_feedback":
            content = msg.get("content", {})
            entry["decision"] = content.get("decision")
            entry["hypothesis"] = content.get("new_hypothesis") or content.get("reason")
        if msg.get("tag") == "research.hypothesis":
            entry["hypothesis"] = msg.get("content", {}).get("hypothesis")

    return {"traceId": trace_id, "loops": sorted(loops.values(), key=lambda x: x["loopId"])}


def get_returns(trace_id: str, loop_id: int | None = None) -> dict[str, Any]:
    trace_dir = agent_runner.trace_root / trace_id
    points: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []

    from rdagent.log.storage import FileStorage

    fs = FileStorage(trace_dir)
    for msg in fs.iter_msg():
        if "Quantitative Backtesting Chart" not in msg.tag:
            continue
        if loop_id is not None and msg.tag and f"Loop_{loop_id}" not in msg.tag and f"loop_{loop_id}" not in msg.tag.lower():
            continue
        df = msg.content
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        report = _normalize_report_df(df)
        for idx, row in report.iterrows():
            points.append(
                {
                    "time": str(idx),
                    "bench": float(row.get("cum_bench", 0)),
                    "strategy": float(row.get("cum_return_w_cost", row.get("cum_return_wo_cost", 0))),
                    "excess": float(row.get("cum_ex_return_w_cost", row.get("cum_ex_return_wo_cost", 0))),
                }
            )
        if len(report.index) > 1:
            markers.append({"time": str(report.index[-1]), "type": "rebalance"})
        break

    return {"traceId": trace_id, "loopId": loop_id, "points": points, "markers": markers}


def _load_messages(trace_id: str) -> list[dict]:
    full_id = str(agent_runner.trace_root / trace_id)
    if full_id not in agent_runner.processes or not agent_runner.processes[full_id].messages:
        read_trace_from_disk(agent_runner.trace_root / trace_id, full_id, agent_runner.processes)
    return agent_runner.get_trace_messages(trace_id, return_all=True)


def _normalize_report_df(df: pd.DataFrame) -> pd.DataFrame:
    from rdagent.log.ui.qlib_report_figure import _calculate_report_data

    return _calculate_report_data(df)
