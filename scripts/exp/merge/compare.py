import pickle
import re
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger
import typer

app = typer.Typer()

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.ui.conf import UI_SETTING


def get_script_time(stdout_p: Path):
    with stdout_p.open("r") as f:
        first_line = next(f).strip()
        last_line = deque(f, maxlen=1).pop().strip()

        # Extract timestamps from the lines
        first_time_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})", first_line)
        last_time_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})", last_line)

        if first_time_match and last_time_match:
            first_time = datetime.fromisoformat(first_time_match.group(1))
            last_time = datetime.fromisoformat(last_time_match.group(1))
            return pd.Timedelta(last_time - first_time)

    return None


def get_final_sota_exp(log_path: Path):
    sota_exp_paths = [i for i in log_path.rglob(f"**/SOTA experiment/**/*.pkl")]
    if len(sota_exp_paths) == 0:
        return None
    final_sota_exp_path = max(sota_exp_paths, key=lambda x: int(re.match(r".*Loop_(\d+).*", str(x))[1]))
    with final_sota_exp_path.open("rb") as f:
        final_sota_exp = pickle.load(f)
    return final_sota_exp


def load_times(log_path: Path):
    """加载时间数据"""
    try:
        session_path = log_path / "__session__"
        max_li = max(int(p.name) for p in session_path.iterdir() if p.is_dir() and p.name.isdigit())
        max_step = max(int(p.name.split("_")[0]) for p in (session_path / str(max_li)).iterdir() if p.is_file())
        rdloop_obj_p = next((session_path / str(max_li)).glob(f"{max_step}_*"))

        rd_times = DataScienceRDLoop.load(rdloop_obj_p, do_truncate=False).loop_trace
    except Exception as e:
        rd_times = {}
    return rd_times


def get_summary_df(log_folders: list[str], sn: str = "summary.pkl") -> tuple[dict, pd.DataFrame]:
    summarys = {}
    for lf in log_folders:
        if not (Path(lf) / sn).exists():
            logger.warning(
                f"{sn} not found in **{lf}**\n\nRun:`dotenv run -- python rdagent/log/mle_summary.py grade_summary --log_folder={lf} --hours=<>`"
            )
        else:
            summarys[lf] = pd.read_pickle(Path(lf) / sn)

    if len(summarys) == 0:
        return {}, pd.DataFrame()

    summary = {}
    for lf, s in summarys.items():
        for k, v in s.items():
            stdout_p = Path(lf) / f"{k}.stdout"
            if stdout_p.exists():
                v["script_time"] = get_script_time(stdout_p)
            else:
                v["script_time"] = None

            exp_gen_time = timedelta()
            coding_time = timedelta()
            running_time = timedelta()
            all_time = timedelta()
            times_info = load_times(Path(lf) / k)
            for time_info in times_info.values():
                all_time += sum((ti.end - ti.start for ti in time_info), timedelta())
                exp_gen_time += time_info[0].end - time_info[0].start
                if len(time_info) > 1:
                    coding_time += time_info[1].end - time_info[1].start
                if len(time_info) > 2:
                    running_time += time_info[2].end - time_info[2].start
            v["exec_time"] = str(all_time).split(".")[0]
            v["exp_gen_time"] = str(exp_gen_time).split(".")[0]
            v["coding_time"] = str(coding_time).split(".")[0]
            v["running_time"] = str(running_time).split(".")[0]

            final_sota_exp = get_final_sota_exp(Path(lf) / k)
            if final_sota_exp is not None and final_sota_exp.result is not None:
                v["sota_exp_score_valid"] = final_sota_exp.result.loc["ensemble"].iloc[0]
            else:
                v["sota_exp_score_valid"] = None
            v["sota_exp_stat_new"] = None
            # 调整实验名字
            if "combined_logs" in lf:
                # get name like deciding-cod; use a strategy relative to "combined_logs"
                # Example lf: '/Data/home/xiaoyang/repos/JobAndExp/amlt_project/amlt_processed/12h/deciding-cod/combined_logs'
                parts = Path(lf).parts
                try:
                    idx = parts.index("combined_logs")
                    # The exp name is the part immediately before "combined_logs"
                    exp_name = parts[idx - 1] if idx > 0 else "unknown-exp"
                except ValueError:
                    exp_name = "unknown-exp"
                summary[f"{exp_name} - {k}"] = v
            elif "amlt" in lf:
                summary[f"{lf[lf.rfind('amlt')+5:].split('/')[0]} - {k}"] = v
            elif "ep" in lf:
                summary[f"{lf[lf.rfind('ep'):]} - {k}"] = v
            else:
                summary[f"{lf} - {k}"] = v

    summary = {k: v for k, v in summary.items() if "competition" in v}
    base_df = pd.DataFrame(
        columns=[
            "Competition",
            "Script Time",
            "Exec Time",
            "Exp Gen",
            "Coding",
            "Running",
            "Total Loops",
            "Successful Final Decision",
            "Made Submission",
            "Valid Submission",
            "V/M",
            "Above Median",
            "Bronze",
            "Silver",
            "Gold",
            "Any Medal",
            "Best Result",
            "SOTA Exp",
            "SOTA Exp (_to_submit)",
            "SOTA Exp Score (valid)",
            "SOTA Exp Score",
            "Baseline Score",
            "Ours - Base",
            "Ours vs Base",
            "Ours vs Bronze",
            "Ours vs Silver",
            "Ours vs Gold",
            "Bronze Threshold",
            "Silver Threshold",
            "Gold Threshold",
            "Medium Threshold",
        ],
        index=summary.keys(),
    )

    # Read baseline results
    baseline_result_path = UI_SETTING.baseline_result_path
    if Path(baseline_result_path).exists():
        baseline_df = pd.read_csv(baseline_result_path)

    def compare_score(s1, s2):
        if s1 is None or s2 is None:
            return None
        try:
            import math
            c_value = math.exp(abs(math.log(s1 / s2)))
        except Exception:
            c_value = None
        return c_value

    for k, v in summary.items():
        loop_num = v["loop_num"]
        base_df.loc[k, "Competition"] = v["competition"]
        base_df.loc[k, "Script Time"] = v["script_time"]
        base_df.loc[k, "Exec Time"] = v["exec_time"]
        base_df.loc[k, "Exp Gen"] = v["exp_gen_time"]
        base_df.loc[k, "Coding"] = v["coding_time"]
        base_df.loc[k, "Running"] = v["running_time"]
        base_df.loc[k, "Total Loops"] = loop_num
        if loop_num == 0:
            base_df.loc[k] = "N/A"
        else:
            base_df.loc[k, "Successful Final Decision"] = v["success_loop_num"]
            base_df.loc[k, "Made Submission"] = v["made_submission_num"]
            if v["made_submission_num"] > 0:
                base_df.loc[k, "Best Result"] = "made_submission"
            base_df.loc[k, "Valid Submission"] = v["valid_submission_num"]
            if v["valid_submission_num"] > 0:
                base_df.loc[k, "Best Result"] = "valid_submission"
            base_df.loc[k, "Above Median"] = v["above_median_num"]
            if v["above_median_num"] > 0:
                base_df.loc[k, "Best Result"] = "above_median"
            base_df.loc[k, "Bronze"] = v["bronze_num"]
            if v["bronze_num"] > 0:
                base_df.loc[k, "Best Result"] = "bronze"
            base_df.loc[k, "Silver"] = v["silver_num"]
            if v["silver_num"] > 0:
                base_df.loc[k, "Best Result"] = "silver"
            base_df.loc[k, "Gold"] = v["gold_num"]
            if v["gold_num"] > 0:
                base_df.loc[k, "Best Result"] = "gold"
            base_df.loc[k, "Any Medal"] = v["get_medal_num"]

            baseline_score = None
            if Path(baseline_result_path).exists():
                baseline_score = baseline_df.loc[baseline_df["competition_id"] == v["competition"], "score"].item()

            base_df.loc[k, "SOTA Exp"] = v.get("sota_exp_stat", None)
            base_df.loc[k, "SOTA Exp (_to_submit)"] = v["sota_exp_stat_new"]
            if baseline_score is not None and v.get("sota_exp_score", None) is not None:
                base_df.loc[k, "Ours - Base"] = v["sota_exp_score"] - baseline_score
            base_df.loc[k, "Ours vs Base"] = compare_score(v["sota_exp_score"], baseline_score)
            base_df.loc[k, "Ours vs Bronze"] = compare_score(v["sota_exp_score"], v.get("bronze_threshold", None))
            base_df.loc[k, "Ours vs Silver"] = compare_score(v["sota_exp_score"], v.get("silver_threshold", None))
            base_df.loc[k, "Ours vs Gold"] = compare_score(v["sota_exp_score"], v.get("gold_threshold", None))
            base_df.loc[k, "SOTA Exp Score"] = v.get("sota_exp_score", None)
            base_df.loc[k, "SOTA Exp Score (valid)"] = v.get("sota_exp_score_valid", None)
            base_df.loc[k, "Baseline Score"] = baseline_score
            base_df.loc[k, "Bronze Threshold"] = v.get("bronze_threshold", None)
            base_df.loc[k, "Silver Threshold"] = v.get("silver_threshold", None)
            base_df.loc[k, "Gold Threshold"] = v.get("gold_threshold", None)
            base_df.loc[k, "Medium Threshold"] = v.get("median_threshold", None)

    base_df["SOTA Exp"] = base_df["SOTA Exp"].replace("", pd.NA)

    base_df["SOTA Exp Score (valid)"] = base_df["SOTA Exp Score (valid)"].replace("Not Calculated", 0)
    base_df["SOTA Exp Score (valid)"] = base_df["SOTA Exp Score (valid)"].replace("Not Computed", 0)
    base_df = base_df.astype(
        {
            "Total Loops": int,
            "Successful Final Decision": int,
            "Made Submission": int,
            "Valid Submission": int,
            "Above Median": int,
            "Bronze": int,
            "Silver": int,
            "Gold": int,
            "Any Medal": int,
            "Ours - Base": float,
            "Ours vs Base": float,
            "SOTA Exp Score": float,
            "SOTA Exp Score (valid)": float,
            "Baseline Score": float,
            "Bronze Threshold": float,
            "Silver Threshold": float,
            "Gold Threshold": float,
            "Medium Threshold": float,
        }
    )
    return summary, base_df


def main(
    exp_list: List[str] = typer.Option(..., "--exp-list", help="List of experiment names.", show_default=False),
    output: str = typer.Option("merge_base_df.h5", help="Output summary file name."),
    summary_name: str = typer.Option("summary.pkl", help="Summary pickle file name in log folders."),
):
    """
    Generate summary and base dataframe for given experiment list, and save to a summary file.
    """
    print(f"exp_list: {exp_list}")
    log_folders = [
        f"/home/xiaoyang/repos/JobAndExp/amlt_project/amlt/{exp}/combined_logs" for exp in exp_list
    ]
    summary, base_df = get_summary_df(log_folders, sn=summary_name)
    print("Summary keys:", list(summary.keys()))
    print("Summary DataFrame:")
    print(base_df)
    base_df.to_hdf(output, "data")
    print(f"Summary saved to {output}")

if __name__ == "__main__":
    app.command()(main)
    app()

