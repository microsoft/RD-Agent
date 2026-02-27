"""
AutoRL-Bench Results Dashboard

Usage:
    streamlit run rdagent/scenarios/rl/autorl_bench/core/ui.py --server.port=8510 --server.address=0.0.0.0
"""
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------- 页面配置 ----------
st.set_page_config(page_title="AutoRL-Bench", page_icon="🧪", layout="wide")

CSV_PATH = Path(__file__).resolve().parent.parent / "results.csv"

# ---------- 自定义样式 ----------
st.markdown("""
<style>
    /* 指标卡片 */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px 14px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.7;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
        font-weight: 700;
    }
    /* 表格行高亮 */
    .stDataFrame td {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 标题 ----------
st.markdown("# 🧪 AutoRL-Bench Results")
st.divider()

# ---------- 加载数据 ----------
if not CSV_PATH.exists():
    st.info("No results yet. Run an experiment first.")
    st.stop()

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["duration_min"] = (df["duration_s"] / 60).round(1)

# ---------- 侧栏 ----------
with st.sidebar:
    st.markdown("### Filters")
    agents = ["All"] + sorted(df["agent"].unique().tolist())
    sel_agent = st.selectbox("Agent", agents)

    tasks = ["All"] + sorted(df["task"].unique().tolist())
    sel_task = st.selectbox("Task", tasks)

    st.divider()
    st.markdown("### About")
    st.markdown(
        "Evaluating LLM-driven agents that optimize smaller LLMs "
        "via RL post-training."
    )

filtered = df.copy()
if sel_agent != "All":
    filtered = filtered[filtered["agent"] == sel_agent]
if sel_task != "All":
    filtered = filtered[filtered["task"] == sel_task]

# ---------- Agent 对比 ----------
if len(filtered) > 1:
    st.markdown("#### Agent Summary")
    summary = (
        filtered.groupby(["agent", "task", "base_model"])
        .agg(
            runs=("agent", "size"),
            success=("success", "sum"),
            best=("best_score", "max"),
            best_improve=("improvement", "max"),
            subs=("submissions", "sum"),
        )
        .round(2)
        .reset_index()
        .sort_values("best", ascending=False)
    )
    summary.columns = ["Agent", "Task", "Base Model", "Runs", "Success", "Best", "Best Impr.", "Submissions"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()

# ---------- 结果表格 ----------
st.markdown("#### Run History")
display = filtered[[
    "timestamp", "agent", "driver_model", "base_model", "task",
    "baseline", "best_score", "improvement", "submissions",
    "duration_min", "success", "workspace",
]].sort_values("timestamp", ascending=False)

display.columns = [
    "Time", "Agent", "Driver LLM", "Base Model", "Task",
    "Baseline", "Best Score", "Improvement", "Submissions",
    "Duration(min)", "Success", "Workspace",
]

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        "Best Score": st.column_config.NumberColumn(format="%.2f"),
        "Baseline": st.column_config.NumberColumn(format="%.2f"),
        "Improvement": st.column_config.NumberColumn(format="%.2f"),
        "Duration(min)": st.column_config.NumberColumn(format="%.0f"),
        "Success": st.column_config.CheckboxColumn(),
    },
)
