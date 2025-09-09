"""
Please refer to rdagent/log/ui/utils.py:get_summary_df for more detailed documents about metrics
"""

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit import session_state as state

from rdagent.log.ui.utils import (
    ALL,
    HIGH,
    LITE,
    MEDIUM,
    curve_figure,
    get_statistics_df,
    get_summary_df,
    lite_curve_figure,
    percent_df,
)
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction


def curves_win(summary: dict):
    # draw curves
    cbwin1, cbwin2 = st.columns(2)
    if cbwin1.toggle("Show Curves", key="show_curves"):
        for k, v in summary.items():
            with st.container(border=True):
                st.markdown(f"**:blue[{k}] - :violet[{v['competition']}]**")
                try:
                    tscores = {k: v for k, v in v["test_scores"].items()}
                    tscores = pd.Series(tscores)
                    vscores = {}
                    for k, vs in v["valid_scores"].items():
                        if not vs.index.is_unique:
                            st.warning(
                                f"Loop {k}'s valid scores index are not unique, only the last one will be kept to show."
                            )
                            st.write(vs)
                        vscores[k] = vs[~vs.index.duplicated(keep="last")].iloc[:, 0]
                    if len(vscores) > 0:
                        metric_name = list(vscores.values())[0].name
                    else:
                        metric_name = "None"
                    vscores = pd.DataFrame(vscores)
                    if "ensemble" in vscores.index:
                        ensemble_row = vscores.loc[["ensemble"]]
                        vscores = pd.concat([ensemble_row, vscores.drop("ensemble")])
                    vscores = vscores.T
                    vscores["test"] = tscores
                    vscores.index = [f"L{i}" for i in vscores.index]
                    vscores.columns.name = metric_name

                    st.plotly_chart(curve_figure(vscores))
                except Exception as e:
                    import traceback

                    st.markdown("- Error: " + str(e))
                    st.code(traceback.format_exc())
                    st.markdown("- Valid Scores: ")
                    # st.write({k: type(v) for k, v in v["valid_scores"].items()})
                    st.json(v["valid_scores"])
    if cbwin2.toggle("Show Curves (Lite)", key="show_curves_lite"):
        st.pyplot(lite_curve_figure(summary))


def all_summarize_win():
    def shorten_folder_name(folder: str) -> str:
        if "amlt" in folder:
            return folder[folder.rfind("amlt") + 5 :].split("/")[0]
        if "ep" in folder:
            return folder[folder.rfind("ep") :]
        return folder

    selected_folders = st.multiselect(
        "Show these folders",
        state.log_folders,
        state.log_folders,
        format_func=shorten_folder_name,
    )
    for lf in selected_folders:
        if not (Path(lf) / "summary.pkl").exists():
            st.warning(
                f"summary.pkl not found in **{lf}**\n\nRun:`dotenv run -- python rdagent/log/mle_summary.py grade_summary --log_folder={lf} --hours=<>`"
            )
    summary = {}
    dfs = []
    for lf in selected_folders:
        s, df = get_summary_df(lf)
        df.index = [f"{shorten_folder_name(lf)} - {idx}" for idx in df.index]

        dfs.append(df)
        summary.update({f"{shorten_folder_name(lf)} - {k}": v for k, v in s.items()})
    base_df = pd.concat(dfs)

    valid_rate = float(base_df.get("Valid Improve", pd.Series()).mean())
    test_rate = float(base_df.get("Test Improve", pd.Series()).mean())
    submit_merge_rate = float(base_df.get("Submit Merge", pd.Series()).mean())
    merge_sota_avg = float(base_df.get("Merge Sota", pd.Series()).mean())
    base_df = percent_df(base_df)
    base_df.insert(0, "Select", True)
    bt1, bt2 = st.columns(2)
    select_lite_level = bt2.selectbox(
        "Select MLE-Bench Competitions Level",
        options=["ALL", "HIGH", "MEDIUM", "LITE"],
        index=0,
        key="select_lite_level",
    )
    if select_lite_level != "ALL":
        if select_lite_level == "HIGH":
            lite_set = set(HIGH)
        elif select_lite_level == "MEDIUM":
            lite_set = set(MEDIUM)
        elif select_lite_level == "LITE":
            lite_set = set(LITE)
        else:
            lite_set = set()
        base_df["Select"] = base_df["Competition"].isin(lite_set)
    else:
        base_df["Select"] = True  # select all if ALL is chosen

    if bt1.toggle("Select Best", key="select_best"):

        def apply_func(cdf: pd.DataFrame):
            cp = base_df.loc[cdf.index[0], "Competition"]
            md = get_metric_direction(cp)
            # If SOTA Exp Score (valid, to_submit) column is empty, return the first index
            if cdf["SOTA Exp Score (valid, to_submit)"].dropna().empty:
                return cdf.index[0]
            if md:
                best_idx = cdf["SOTA Exp Score (valid, to_submit)"].idxmax()
            else:
                best_idx = cdf["SOTA Exp Score (valid, to_submit)"].idxmin()
            return best_idx

        best_idxs = base_df.groupby("Competition").apply(apply_func, include_groups=False)
        base_df["Select"] = base_df.index.isin(best_idxs.values)

    base_df = st.data_editor(
        base_df,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Stat this trace.", disabled=False),
        },
        disabled=(col for col in base_df.columns if col not in ["Select"]),
    )
    st.markdown("Ours vs Base: `math.exp(abs(math.log(sota_exp_score / baseline_score)))`")

    # 统计选择的比赛
    base_df = base_df[base_df["Select"]]
    st.markdown(f"**统计的比赛数目: :red[{base_df.shape[0]}]**")
    stat_win_left, stat_win_right = st.columns(2)
    with stat_win_left:
        stat_df = get_statistics_df(base_df)
        st.dataframe(stat_df.round(2))
        markdown_table = f"""
| xxx | {stat_df.iloc[0,1]:.1f} | {stat_df.iloc[1,1]:.1f} | {stat_df.iloc[2,1]:.1f} | {stat_df.iloc[3,1]:.1f} | {stat_df.iloc[4,1]:.1f} | {stat_df.iloc[5,1]:.1f} | {stat_df.iloc[6,1]:.1f}   |
| Valid Improve {valid_rate * 100:.2f}% | Test Improve {test_rate * 100:.2f}% | Submit Merge {submit_merge_rate * 100:.2f}% | Merge Sota {merge_sota_avg * 100:.2f}% |
"""
        st.text(markdown_table)
    with stat_win_right:
        Loop_counts = base_df["Total Loops"]

        # Create histogram
        fig = px.histogram(
            Loop_counts, nbins=15, title="Distribution of Total Loops", color_discrete_sequence=["#3498db"]
        )
        fig.update_layout(title_font_size=16, title_font_color="#2c3e50")

        # Calculate statistics
        mean_value = Loop_counts.mean()
        median_value = Loop_counts.median()

        # Add mean and median lines
        fig.add_vline(x=mean_value, line_color="#e74c3c", line_width=3)
        fig.add_vline(x=median_value, line_color="#f39c12", line_width=3)

        fig.add_annotation(
            x=0.02,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"<span style='color:#e74c3c; font-weight:bold'>Mean: {mean_value:.1f}</span><br><span style='color:#f39c12; font-weight:bold'>Median: {median_value:.1f}</span>",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(128,128,128,0.5)",
            borderwidth=1,
            font=dict(size=12, color="#333333"),
        )

        st.plotly_chart(fig, use_container_width=True)

    # write curve
    st.subheader("Curves", divider="rainbow")
    curves_win(summary)


with st.container(border=True):
    try:
        all_summarize_win()
    except Exception as e:
        import traceback

        st.error(f"Error occurred when show summary:\n{e}")
        st.code(traceback.format_exc())
