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
    percent_df,
)
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction


def days_summarize_win():
    lfs1 = [re.sub(r"log\.srv\d*", "log.srv", folder) for folder in state.log_folders]
    lfs2 = [re.sub(r"log\.srv\d*", "log.srv2", folder) for folder in state.log_folders]
    lfs3 = [re.sub(r"log\.srv\d*", "log.srv3", folder) for folder in state.log_folders]

    _, df1 = get_summary_df(lfs1)
    _, df2 = get_summary_df(lfs2)
    _, df3 = get_summary_df(lfs3)

    df = pd.concat([df1, df2, df3], axis=0)

    def mean_func(x: pd.DataFrame):
        numeric_cols = x.select_dtypes(include=["int", "float"]).mean()
        string_cols = x.select_dtypes(include=["object"]).agg(lambda col: ", ".join(col.fillna("none").astype(str)))
        return pd.concat([numeric_cols, string_cols], axis=0).reindex(x.columns).drop("Competition")

    df = df.groupby("Competition").apply(mean_func)
    if st.toggle("Show Percent", key="show_percent"):
        st.dataframe(percent_df(df, show_origin=False))
    else:
        st.dataframe(df)


def curves_win(summary: dict):
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
    summary, base_df = get_summary_df(selected_folders)
    if not summary:
        return

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
            cp = cdf["Competition"].values[0]
            md = get_metric_direction(cp)
            # If SOTA Exp Score (valid) column is empty, return the first index
            if cdf["SOTA Exp Score (valid)"].dropna().empty:
                return cdf.index[0]
            if md:
                best_idx = cdf["SOTA Exp Score (valid)"].idxmax()
            else:
                best_idx = cdf["SOTA Exp Score (valid)"].idxmin()
            return best_idx

        best_idxs = base_df.groupby("Competition").apply(apply_func)
        base_df["Select"] = base_df.index.isin(best_idxs.values)

    base_df = st.data_editor(
        base_df.style.apply(
            lambda col: col.map(lambda val: "background-color: #F0F8FF"),
            subset=[
                "Baseline Score",
                "Bronze Threshold",
                "Silver Threshold",
                "Gold Threshold",
                "Medium Threshold",
            ],
            axis=0,
        )
        .apply(
            lambda col: col.map(lambda val: "background-color: #FFFFE0"),
            subset=[
                "Ours - Base",
                "Ours vs Base",
                "Ours vs Bronze",
                "Ours vs Silver",
                "Ours vs Gold",
            ],
            axis=0,
        )
        .apply(
            lambda col: col.map(lambda val: "background-color: #E6E6FA"),
            subset=[
                "Script Time",
                "Exec Time",
                "Exp Gen",
                "Coding",
                "Running",
            ],
            axis=0,
        )
        .apply(
            lambda col: col.map(lambda val: "background-color: #F0FFF0"),
            subset=[
                "Best Result",
                "SOTA Exp",
                "SOTA Exp (_to_submit)",
                "SOTA Exp Score",
                "SOTA Exp Score (valid)",
            ],
            axis=0,
        ),
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
"""
        st.text(markdown_table)
    with stat_win_right:
        Loop_counts = base_df["Total Loops"]
        fig = px.histogram(Loop_counts, nbins=10, title="Total Loops Histogram (nbins=10)")
        mean_value = Loop_counts.mean()
        median_value = Loop_counts.median()
        fig.add_vline(
            x=mean_value,
            line_color="orange",
            annotation_text="Mean",
            annotation_position="top right",
            line_width=3,
        )
        fig.add_vline(
            x=median_value,
            line_color="red",
            annotation_text="Median",
            annotation_position="top right",
            line_width=3,
        )
        st.plotly_chart(fig)

    # write curve
    st.subheader("Curves", divider="rainbow")
    if st.toggle("Show Curves", key="show_curves"):
        curves_win(summary)


with st.container(border=True):
    if st.toggle("近3天平均", key="show_3days"):
        days_summarize_win()
with st.container(border=True):
    all_summarize_win()
