import math
import pickle
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import typer
from matplotlib import pyplot as plt

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.core.proposal import Trace
from rdagent.core.utils import cache_with_pickle
from rdagent.log.storage import FileStorage
from rdagent.log.ui.conf import UI_SETTING
from rdagent.log.utils import extract_json, extract_loopid_func_name
from rdagent.oai.llm_utils import md5_hash
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.select.submit import (
    BestValidSelector,
)
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction

LITE = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
]

HIGH = [
    "3d-object-detection-for-autonomous-vehicles",
    "bms-molecular-translation",
    "google-research-identify-contrails-reduce-global-warming",
    "hms-harmful-brain-activity-classification",
    "iwildcam-2019-fgvc6",
    "nfl-player-contact-detection",
    "predict-volcanic-eruptions-ingv-oe",
    "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "siim-covid19-detection",
    "smartphone-decimeter-2022",
    "stanford-covid-vaccine",
    "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection",
]

MEDIUM = [
    "AI4Code",
    "alaska2-image-steganalysis",
    "billion-word-imputation",
    "cassava-leaf-disease-classification",
    "cdiscount-image-classification-challenge",
    "chaii-hindi-and-tamil-question-answering",
    "champs-scalar-coupling",
    "facebook-recruiting-iii-keyword-extraction",
    "freesound-audio-tagging-2019",
    "google-quest-challenge",
    "h-and-m-personalized-fashion-recommendations",
    "herbarium-2020-fgvc7",
    "herbarium-2021-fgvc8",
    "herbarium-2022-fgvc9",
    "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation",
    "icecube-neutrinos-in-deep-ice",
    "imet-2020-fgvc7",
    "inaturalist-2019-fgvc6",
    "iwildcam-2020-fgvc7",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "kuzushiji-recognition",
    "learning-agency-lab-automated-essay-scoring-2",
    "lmsys-chatbot-arena",
    "multi-modal-gesture-recognition",
    "osic-pulmonary-fibrosis-progression",
    "petfinder-pawpularity-score",
    "plant-pathology-2021-fgvc8",
    "seti-breakthrough-listen",
    "statoil-iceberg-classifier-challenge",
    "tensorflow-speech-recognition-challenge",
    "tensorflow2-question-answering",
    "tgs-salt-identification-challenge",
    "tweet-sentiment-extraction",
    "us-patent-phrase-to-phrase-matching",
    "uw-madison-gi-tract-image-segmentation",
    "ventilator-pressure-prediction",
    "whale-categorization-playground",
]

ALL = HIGH + MEDIUM + LITE


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


def _log_path_hash_func(log_path: Path) -> str:
    hash_str = str(log_path) + str(log_path.stat().st_mtime)
    session_p = log_path / "__session__"
    if session_p.exists():
        for ld in session_p.iterdir():
            if ld.is_dir():
                hash_str += str(ld.name) + str(ld.stat().st_mtime)
    else:
        hash_str += "no session now"
    return md5_hash(hash_str)


def map_stat(sota_mle_score: dict | None) -> str:
    sota_exp_stat = None
    if sota_mle_score:  # sota exp's grade output
        if sota_mle_score["gold_medal"]:
            sota_exp_stat = "gold"
        elif sota_mle_score["silver_medal"]:
            sota_exp_stat = "silver"
        elif sota_mle_score["bronze_medal"]:
            sota_exp_stat = "bronze"
        elif sota_mle_score["above_median"]:
            sota_exp_stat = "above_median"
        elif sota_mle_score["valid_submission"]:
            sota_exp_stat = "valid_submission"
        elif sota_mle_score["submission_exists"]:
            sota_exp_stat = "made_submission"
    return sota_exp_stat


def get_best_report(log_path: Path) -> dict | None:
    log_storage = FileStorage(log_path)
    mle_reports = [extract_json(i.content) for i in log_storage.iter_msg(pattern="**/running/mle_score/*/*.pkl")]
    mle_reports = [report for report in mle_reports if report is not None and not pd.isna(report["score"])]
    if mle_reports:
        lower_better = mle_reports[0]["is_lower_better"]
        if lower_better:
            mle_reports.sort(key=lambda report: report["score"])
        else:
            mle_reports.sort(key=lambda report: report["score"], reverse=True)
        return mle_reports[0]
    return None


if UI_SETTING.enable_cache:
    get_best_report = cache_with_pickle(_log_path_hash_func, force=True)(get_best_report)


def _get_sota_exp_stat_hash_func(log_path: Path, selector: Literal["auto", "best_valid"] = "auto") -> str:
    return _log_path_hash_func(log_path) + selector


def get_sota_exp_stat(
    log_path: Path, selector: Literal["auto", "best_valid"] = "auto"
) -> tuple[DSExperiment | None, int | None, dict | None, str | None]:
    """
    Get the SOTA experiment and its statistics from the log path.

    Parameters
    ----------
    log_path : Path
        Path to the experiment log directory.
    selector : Literal["auto", "best_valid"], default "auto"
        If "auto", returns sota_exp_to_submit; if "best_valid", returns sota selected by best valid score.

    Returns
    -------
    tuple[DSExperiment | None, int | None, dict | None, str | None]
        A tuple containing:
        - sota_exp : DSExperiment or None
            The SOTA experiment object or None if not found.
        - sota_loop_id : int or None
            The loop ID of the SOTA experiment or None if not found.
        - sota_mle_score : dict or None
            The MLE score dictionary of the SOTA experiment or None if not found.
        - sota_exp_stat : str or None
            The medal status string ("gold", "silver", "bronze", etc.) or None if not found.
    """
    log_storage = FileStorage(log_path)

    # get sota exp
    sota_exp = None
    if selector == "auto":
        sota_exp_list = [i.content for i in log_storage.iter_msg(tag="sota_exp_to_submit")]
        sota_exp = sota_exp_list[-1] if sota_exp_list else None
    elif selector == "best_valid":
        trace_list = [i.content for i in log_storage.iter_msg(tag="trace")]
        if trace_list:
            final_trace = trace_list[-1]
            final_trace.scen.metric_direction = get_metric_direction(
                final_trace.scen.competition
            )  # FIXME: remove this later.
            bvs = BestValidSelector()
            sota_exp = bvs.get_sota_exp_to_submit(final_trace)

    if sota_exp is None:
        return None, None, None, None

    # find sota exp's loop id
    sota_loop_id = None
    running_exps: list[tuple[DSExperiment, int]] = [
        (i.content, int(re.search(r".*Loop_(\d+).*", str(i.tag))[1]))
        for i in log_storage.iter_msg(pattern="**/running/*/*.pkl")
    ]
    running_exps.sort(key=lambda x: x[1], reverse=True)
    for exp, loop_id in running_exps:
        if exp.experiment_workspace.all_codes == sota_exp.experiment_workspace.all_codes and "".join(
            str(i) for i in exp.hypothesis.__dict__.values()
        ) == "".join(str(i) for i in sota_exp.hypothesis.__dict__.values()):
            sota_loop_id = loop_id
            break

    # get sota exp's mle score
    try:
        sota_mle_score = extract_json(
            [i.content for i in log_storage.iter_msg(tag=f"Loop_{sota_loop_id}.running.mle_score")][0]
        )
    except Exception as e:
        # sota exp is not tested yet
        return sota_exp, sota_loop_id, None, None

    return sota_exp, sota_loop_id, sota_mle_score, map_stat(sota_mle_score)


if UI_SETTING.enable_cache:
    get_sota_exp_stat = cache_with_pickle(_get_sota_exp_stat_hash_func, force=True)(get_sota_exp_stat)


def _get_score_stat_hash_func(log_path: Path, sota_loop_id: int) -> str:
    return _log_path_hash_func(log_path) + str(sota_loop_id)


def get_score_stat(log_path: Path, sota_loop_id: int) -> tuple[float | None, float | None, bool | None, float | None]:
    """
    Get the scores before and after merge period.

    Parameters
    ----------
    log_path : Path
        Path to the experiment log directory.
    sota_loop_id : int
        The loop ID of the SOTA experiment to check for merge status.

    Returns
    -------
    tuple[float | None, float | None]
        A tuple containing:
        - valid_improve : bool
            True if valid score is improved during merge period.
        - test_improve : bool
            True if test score is improved during merge period.
        - submit_is_merge : bool
            True if the sota loop is a merge loop.
        - merge_sota_rate : float | None
            The merge sota rate.
    """
    valid_before_merge = []
    test_before_merge = []
    valid_after_merge = []
    test_after_merge = []
    submit_is_merge = False
    is_lower_better = False
    valid_improve = False
    test_improve = False
    total_merge_loops = 0
    log_storage = FileStorage(log_path)
    all_trace = list(log_storage.iter_msg(tag="trace"))
    if all_trace:
        final_trace = all_trace[-1].content
    else:
        return None, None, None, None
    for loop_index, (exp, fb) in enumerate(final_trace.hist):
        if hasattr(final_trace, "idx2loop_id"):
            loop_id = final_trace.idx2loop_id[loop_index]
        else:
            loop_id = int(re.search(r"\d+", all_trace[loop_index].tag).group())

        is_merge = False
        direct_exp_gen = log_storage.iter_msg(pattern=f"Loop_{loop_id}/direct_exp_gen/debug_tpl/*/*.pkl")
        for tr in direct_exp_gen:
            uri = tr.content.get("uri") if isinstance(tr.content, dict) else getattr(tr.content, "uri", None)
            if isinstance(uri, str) and "scenarios.data_science.proposal.exp_gen.merge" in uri:
                is_merge = True
                total_merge_loops += 1
                if sota_loop_id == loop_id:
                    submit_is_merge = True
                break
        if not fb.decision:
            continue

        try:
            mle_score = extract_json(
                [i.content for i in log_storage.iter_msg(tag=f"Loop_{loop_id}.running.mle_score")][0]
            )
        except Exception:
            continue

        if not mle_score:
            continue

        is_lower_better = mle_score.get("is_lower_better", False)
        valid_score = pd.DataFrame(exp.result).loc["ensemble"].iloc[0]

        if is_merge:
            valid_after_merge.append(valid_score)
            if mle_score["score"] is not None:
                test_after_merge.append(mle_score["score"])
        else:
            valid_before_merge.append(valid_score)
            if mle_score["score"] is not None:
                test_before_merge.append(mle_score["score"])

    if is_lower_better:
        if valid_after_merge:
            valid_improve = not valid_before_merge or min(valid_after_merge) < min(valid_before_merge)
        if test_after_merge:
            test_improve = not test_before_merge or min(test_after_merge) < min(test_before_merge)
    else:
        if valid_after_merge:
            valid_improve = not valid_before_merge or max(valid_after_merge) > max(valid_before_merge)
        if test_after_merge:
            test_improve = not test_before_merge or max(test_after_merge) > max(test_before_merge)

    merge_sota_rate = 0 if not total_merge_loops else len(test_after_merge) / total_merge_loops
    return valid_improve, test_improve, submit_is_merge, merge_sota_rate


if UI_SETTING.enable_cache:
    get_score_stat = cache_with_pickle(_get_score_stat_hash_func, force=True)(get_score_stat)


def load_times_deprecated(log_path: Path):
    try:
        session_path = log_path / "__session__"
        max_li = max(int(p.name) for p in session_path.iterdir() if p.is_dir() and p.name.isdigit())
        max_step = max(int(p.name.split("_")[0]) for p in (session_path / str(max_li)).iterdir() if p.is_file())
        rdloop_obj_p = next((session_path / str(max_li)).glob(f"{max_step}_*"))

        rd_times = DataScienceRDLoop.load(rdloop_obj_p).loop_trace
    except Exception as e:
        rd_times = {}
    return rd_times


if UI_SETTING.enable_cache:
    load_times_deprecated = cache_with_pickle(_log_path_hash_func, force=True)(load_times_deprecated)


def load_times_info(log_path: Path) -> dict[int, dict[str, dict[Literal["start_time", "end_time"], datetime]]]:
    """
    Load timing information for each loop and step.

    Returns
    -------
    dict[int, dict[str, dict[Literal["start_time", "end_time"], datetime]]]
        Dictionary with loop IDs as keys, where each value contains step names
        mapping to their start and end times.

        Example:
            {
                1: {
                    "exp_gen": {
                        "start_time": datetime(2024, 1, 1, 10, 0, 0),
                        "end_time": datetime(2024, 1, 1, 10, 15, 30)
                    },
                    "coding": {
                        "start_time": datetime(2024, 1, 1, 10, 15, 30),
                        "end_time": datetime(2024, 1, 1, 10, 45, 12)
                    }
                },
            }
    """
    log_storage = FileStorage(log_path)
    time_msgs = list(log_storage.iter_msg(tag="time_info"))
    exp_gen_time_msgs = list(log_storage.iter_msg(tag="exp_gen_time_info"))
    times_info = defaultdict(dict)
    for msg in time_msgs:
        li, fn = extract_loopid_func_name(msg.tag)
        times_info[int(li)][fn] = msg.content
    for msg in exp_gen_time_msgs:
        li, fn = extract_loopid_func_name(msg.tag)
        times_info[int(li)]["exp_gen"] = msg.content
    return times_info


if UI_SETTING.enable_cache:
    load_times_info = cache_with_pickle(_log_path_hash_func, force=True)(load_times_info)


def _log_folders_summary_hash_func(log_folder: str | Path, hours: int | None = None):
    summary_p = Path(log_folder) / (f"summary.pkl" if hours is None else f"summary_{hours}h.pkl")
    if summary_p.exists():
        hash_str = str(summary_p) + str(summary_p.stat().st_mtime)
    else:
        hash_str = f"{summary_p} not exists"
    return md5_hash(hash_str)


def get_summary_df(log_folder: str | Path, hours: int | None = None) -> tuple[dict, pd.DataFrame]:
    """Process experiment logs and generate summary DataFrame.

    Several key metrics that need explanation:

    * Successful Final Decision: Percentage of experiment loops where code executed correctly
      and produced expected output, as determined by evaluation feedback

    * Best Result: The highest achievement level reached by any experiment throughout the entire
      process, ranging from lowest to highest: made_submission, valid_submission, above_median,
      bronze, silver, gold

    * SOTA Exp: Version found by working backward from the last attempt to find the most recent
      successful experiment

    * SOTA Exp (to_submit): Version selected by LLM from all successful experiments for
      competition submission, considering not only scores but also generalization ability
      and overfitting risk, totally decided by LLM

    """
    log_folder = Path(log_folder)
    sn = "summary.pkl" if hours is None else f"summary_{hours}h.pkl"
    if (log_folder / sn).exists():
        summary: dict = pd.read_pickle(log_folder / sn)
    else:
        return {}, pd.DataFrame()

    for k, v in summary.items():
        stdout_p = log_folder / f"{k}.stdout"
        if stdout_p.exists():
            v["script_time"] = get_script_time(stdout_p)
        else:
            v["script_time"] = None

        times_info = load_times_info(log_folder / k)

        exp_gen_time = coding_time = running_time = timedelta()
        start_times, end_times = [], []

        for loop_times in times_info.values():
            for step_name, step_time in loop_times.items():
                duration = step_time["end_time"] - step_time["start_time"]
                start_times.append(step_time["start_time"])
                end_times.append(step_time["end_time"])

                if step_name == "exp_gen":
                    exp_gen_time += duration
                elif step_name == "coding":
                    coding_time += duration
                elif step_name == "running":
                    running_time += duration

        all_time = (max(end_times) - min(start_times)) if start_times else timedelta()
        v["exec_time"] = str(all_time).split(".")[0]
        v["exp_gen_time"] = str(exp_gen_time).split(".")[0]
        v["coding_time"] = str(coding_time).split(".")[0]
        v["running_time"] = str(running_time).split(".")[0]

        # overwrite sota_exp_stat in summary.pkl because it may not be correct in multi-trace
        sota_exp_submit, v["sota_loop_id_new"], sota_submit_report, v["sota_exp_stat_new"] = get_sota_exp_stat(
            log_folder / k, selector="auto"
        )
        sota_exp_bv, v["sota_loop_id"], sota_bv_report, v["sota_exp_stat"] = get_sota_exp_stat(
            log_folder / k, selector="best_valid"
        )
        (
            v["valid_improve"],
            v["test_improve"],
            v["submit_is_merge"],
            v["merge_sota_rate"],
        ) = get_score_stat(log_folder / k, v["sota_loop_id_new"])

        if sota_exp_submit is not None:
            try:
                sota_submit_result = sota_exp_submit.result
            except AttributeError:  # Compatible with old versions
                sota_submit_result = sota_exp_submit.__dict__["result"]
            v["sota_exp_score_valid_new"] = (
                sota_submit_result.loc["ensemble"].iloc[0] if sota_submit_result is not None else None
            )
        v["sota_exp_score"] = sota_bv_report["score"] if sota_bv_report else None
        v["sota_exp_score_new"] = sota_submit_report["score"] if sota_submit_report else None

    summary = {k: v for k, v in summary.items() if "competition" in v}
    base_df = pd.DataFrame(
        columns=[
            "Competition",
            "Total Loops",
            "Best Result",
            "SOTA Exp (to_submit)",
            "SOTA LID (to_submit)",
            "SOTA Exp Score (to_submit)",
            "SOTA Exp Score (valid, to_submit)",
            "SOTA Exp",
            "SOTA Exp Score",
            "Successful Final Decision",
            "Made Submission",
            "Valid Submission",
            "V/M",
            "Above Median",
            "Bronze",
            "Silver",
            "Gold",
            "Any Medal",
            "Script Time",
            "Exec Time",
            "Exp Gen",
            "Coding",
            "Running",
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
            c_value = math.exp(abs(math.log(s1 / s2)))
        except Exception as e:
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
            base_df.loc[k, "SOTA Exp Score"] = v.get("sota_exp_score", None)
            base_df.loc[k, "Valid Improve"] = v.get("valid_improve", None)
            base_df.loc[k, "Test Improve"] = v.get("test_improve", None)
            base_df.loc[k, "Submit Merge"] = v.get("submit_is_merge", None)
            base_df.loc[k, "Merge Sota"] = v.get("merge_sota_rate", None)
            base_df.loc[k, "SOTA Exp (to_submit)"] = v["sota_exp_stat_new"]
            base_df.loc[k, "SOTA Exp Score (to_submit)"] = v.get("sota_exp_score_new", None)
            base_df.loc[k, "SOTA LID (to_submit)"] = v.get("sota_loop_id_new", None)
            base_df.loc[k, "SOTA Exp Score (valid, to_submit)"] = v.get("sota_exp_score_valid_new", None)

            if baseline_score is not None and v.get("sota_exp_score", None) is not None:
                base_df.loc[k, "Ours - Base"] = v["sota_exp_score"] - baseline_score
            base_df.loc[k, "Ours vs Base"] = compare_score(v["sota_exp_score"], baseline_score)
            base_df.loc[k, "Ours vs Bronze"] = compare_score(v["sota_exp_score"], v.get("bronze_threshold", None))
            base_df.loc[k, "Ours vs Silver"] = compare_score(v["sota_exp_score"], v.get("silver_threshold", None))
            base_df.loc[k, "Ours vs Gold"] = compare_score(v["sota_exp_score"], v.get("gold_threshold", None))
            base_df.loc[k, "Baseline Score"] = baseline_score
            base_df.loc[k, "Bronze Threshold"] = v.get("bronze_threshold", None)
            base_df.loc[k, "Silver Threshold"] = v.get("silver_threshold", None)
            base_df.loc[k, "Gold Threshold"] = v.get("gold_threshold", None)
            base_df.loc[k, "Medium Threshold"] = v.get("median_threshold", None)

    base_df["SOTA Exp"] = base_df["SOTA Exp"].replace("", pd.NA)

    base_df.loc[
        base_df["SOTA Exp Score (valid, to_submit)"].apply(lambda x: isinstance(x, str)),
        "SOTA Exp Score (valid, to_submit)",
    ] = 0.0
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
            "SOTA Exp Score (valid, to_submit)": float,
            "Baseline Score": float,
            "Bronze Threshold": float,
            "Silver Threshold": float,
            "Gold Threshold": float,
            "Medium Threshold": float,
            "Valid Improve": bool,
            "Test Improve": bool,
            "Submit Merge": bool,
            "Merge Sota": float,
        }
    )
    return summary, base_df


if UI_SETTING.enable_cache:
    get_summary_df = cache_with_pickle(_log_folders_summary_hash_func, force=True)(get_summary_df)


def percent_df(summary_df: pd.DataFrame, show_origin=True) -> pd.DataFrame:
    """
    Convert the summary DataFrame to a percentage format.
    """
    new_df = summary_df.copy(deep=True)

    # Convert columns to object dtype so we can store strings like "14 (53.85%)" without warnings
    columns_to_convert = [
        "Successful Final Decision",
        "Made Submission",
        "Valid Submission",
        "Above Median",
        "Bronze",
        "Silver",
        "Gold",
        "Any Medal",
    ]

    # Filter columns_to_convert to only include columns that exist in new_df
    existing_columns = [col for col in columns_to_convert if col in new_df.columns]
    new_df[existing_columns] = new_df[existing_columns].astype(object)

    def num2percent(num: int, total: int, show_origin=True) -> str:
        num = int(num)
        total = int(total)
        if show_origin:
            return f"{num} ({round(num / total * 100, 2)}%)"
        return f"{round(num / total * 100, 2)}%"

    for k in new_df.index:
        loop_num = int(new_df.loc[k, "Total Loops"])
        if loop_num != 0:
            if new_df.loc[k, "Made Submission"] != 0:
                new_df.loc[k, "V/M"] = (
                    f"{round(new_df.loc[k, 'Valid Submission'] / new_df.loc[k, 'Made Submission'] * 100, 2)}%"
                )
            else:
                new_df.loc[k, "V/M"] = "N/A"
            for col in existing_columns:
                new_df.loc[k, col] = num2percent(new_df.loc[k, col], loop_num, show_origin)

    return new_df


def get_statistics_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df["Any Medal"].dtype == int:
        check_value = 0
    else:
        sample_val = summary_df["Any Medal"].dropna().iloc[0]
        if "(" in sample_val:
            check_value = "0 (0.0%)"
        else:
            check_value = "0.0%"
    total_stat = (
        summary_df[
            [
                "Made Submission",
                "Valid Submission",
                "Above Median",
                "Bronze",
                "Silver",
                "Gold",
                "Any Medal",
            ]
        ]
        != check_value
    ).sum()
    total_stat.name = "总体统计(%)"
    total_stat.loc["Bronze"] = summary_df["Best Result"].value_counts().get("bronze", 0)
    total_stat.loc["Silver"] = summary_df["Best Result"].value_counts().get("silver", 0)
    total_stat.loc["Gold"] = summary_df["Best Result"].value_counts().get("gold", 0)
    total_stat = total_stat / summary_df.shape[0] * 100

    # SOTA Exp 统计
    se_counts = summary_df["SOTA Exp"].value_counts(dropna=True)
    se_counts.loc["made_submission"] = se_counts.sum()
    se_counts.loc["Any Medal"] = se_counts.get("gold", 0) + se_counts.get("silver", 0) + se_counts.get("bronze", 0)
    se_counts.loc["above_median"] = se_counts.get("above_median", 0) + se_counts.get("Any Medal", 0)
    se_counts.loc["valid_submission"] = se_counts.get("valid_submission", 0) + se_counts.get("above_median", 0)

    sota_exp_stat = pd.Series(index=total_stat.index, dtype=int, name="SOTA Exp 统计(%)")
    sota_exp_stat.loc["Made Submission"] = se_counts.get("made_submission", 0)
    sota_exp_stat.loc["Valid Submission"] = se_counts.get("valid_submission", 0)
    sota_exp_stat.loc["Above Median"] = se_counts.get("above_median", 0)
    sota_exp_stat.loc["Bronze"] = se_counts.get("bronze", 0)
    sota_exp_stat.loc["Silver"] = se_counts.get("silver", 0)
    sota_exp_stat.loc["Gold"] = se_counts.get("gold", 0)
    sota_exp_stat.loc["Any Medal"] = se_counts.get("Any Medal", 0)
    sota_exp_stat = sota_exp_stat / summary_df.shape[0] * 100

    # SOTA Exp (trace.sota_exp_to_submit) 统计
    se_counts_new = summary_df["SOTA Exp (to_submit)"].value_counts(dropna=True)
    se_counts_new.loc["made_submission"] = se_counts_new.sum()
    se_counts_new.loc["Any Medal"] = (
        se_counts_new.get("gold", 0) + se_counts_new.get("silver", 0) + se_counts_new.get("bronze", 0)
    )
    se_counts_new.loc["above_median"] = se_counts_new.get("above_median", 0) + se_counts_new.get("Any Medal", 0)
    se_counts_new.loc["valid_submission"] = se_counts_new.get("valid_submission", 0) + se_counts_new.get(
        "above_median", 0
    )

    sota_exp_stat_new = pd.Series(index=total_stat.index, dtype=int, name="SOTA Exp (to_submit) 统计(%)")
    sota_exp_stat_new.loc["Made Submission"] = se_counts_new.get("made_submission", 0)
    sota_exp_stat_new.loc["Valid Submission"] = se_counts_new.get("valid_submission", 0)
    sota_exp_stat_new.loc["Above Median"] = se_counts_new.get("above_median", 0)
    sota_exp_stat_new.loc["Bronze"] = se_counts_new.get("bronze", 0)
    sota_exp_stat_new.loc["Silver"] = se_counts_new.get("silver", 0)
    sota_exp_stat_new.loc["Gold"] = se_counts_new.get("gold", 0)
    sota_exp_stat_new.loc["Any Medal"] = se_counts_new.get("Any Medal", 0)
    sota_exp_stat_new = sota_exp_stat_new / summary_df.shape[0] * 100

    stat_df = pd.concat([total_stat, sota_exp_stat, sota_exp_stat_new], axis=1)
    return stat_df


def curve_figure(scores: pd.DataFrame) -> go.Figure:
    """
    scores.columns.name is the metric name, e.g., "accuracy", "f1", etc.
    scores.index is the loop index, e.g., ["L1", "L2", "L3", ...]
    scores["test"] is the test score, other columns are valid scores for different loops.
    The "ensemble" column is the ensemble score.
    The "Test scores" and "ensemble" lines are visible, while other valid scores are hidden by default.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scores.index,
            y=scores["test"],
            mode="lines+markers",
            name="Test scores",
            marker=dict(symbol="diamond"),
            line=dict(shape="linear", dash="dash"),
        )
    )
    for column in scores.columns:
        if column != "test":
            fig.add_trace(
                go.Scatter(
                    x=scores.index,
                    y=scores[column],
                    mode="lines+markers",
                    name=f"{column}",
                    visible=("legendonly" if column != "ensemble" else None),
                )
            )
    fig.update_layout(title=f"Test and Valid scores (metric: {scores.columns.name})")

    return fig


def lite_curve_figure(summary):
    cols = 3  # 每行几个图，可调整
    rows = math.ceil(len(summary) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
    axes = axes.flatten()  # 💡 扁平化 axes 结构，确保 ax.plot 不报错
    colors = {"Bronze": "#cd7f32", "Silver": "#c0c0c0", "Gold": "#ffd700", "Median": "gray"}

    for idx, competition in enumerate(summary.keys()):
        data = summary[competition]
        test_scores_df = pd.DataFrame.from_dict(data["test_scores"], orient="index", columns=["Test Score"])
        test_scores_df.index.name = "Loop"
        valid_scores_dict = data["valid_scores"]

        # 提取 ensemble 验证分数
        ensemble_scores = {}
        for loop_id, df in valid_scores_dict.items():
            if "ensemble" in df.index:
                ensemble_scores[loop_id] = df.loc["ensemble"].iloc[0]

        ensemble_valid_df = pd.DataFrame.from_dict(ensemble_scores, orient="index", columns=["Ensemble Valid Score"])
        ensemble_valid_df.index.name = "Loop"

        combined_df = pd.merge(ensemble_valid_df, test_scores_df, left_index=True, right_index=True, how="outer")
        combined_df.sort_index(inplace=True)

        bronze_threshold = data["bronze_threshold"]
        silver_threshold = data["silver_threshold"]
        gold_threshold = data["gold_threshold"]
        sota_loop_id = data["sota_loop_id_new"]

        # 当前 subplot
        ax = axes[idx]
        ax.plot(combined_df.index, combined_df["Ensemble Valid Score"], marker="o", markersize=4, label="Valid Score")
        ax.plot(combined_df.index, combined_df["Test Score"], marker="s", markersize=4, label="Test Score")
        ax.axhline(y=bronze_threshold, color=colors["Bronze"], linestyle="--", linewidth=2)
        ax.axhline(y=silver_threshold, color=colors["Silver"], linestyle="--", linewidth=2)
        ax.axhline(y=gold_threshold, color=colors["Gold"], linestyle="--", linewidth=2)

        # 标记 SOTA loop
        if sota_loop_id is not None and sota_loop_id in combined_df.index:
            ax.axvline(x=sota_loop_id, color="red", linestyle=":", linewidth=2, alpha=0.7)
            # 添加文本标注
            ax.text(
                sota_loop_id,
                ax.get_ylim()[1] * 0.95,
                f"L{sota_loop_id}",
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
            )

        ax.set_title(f"{competition}")
        ax.set_xlabel("Loop")
        ax.set_ylabel("Score")
        ax.grid(True)
        ax.legend()

    # 删除多余 subplot（如果有）
    for j in range(len(summary), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


def trace_figure(trace: Trace, merge_loops: list = []):
    G = nx.DiGraph()

    # Calculate the number of ancestors for each node (root node is 0, more ancestors means lower level)
    levels = {}
    for i in range(len(trace.dag_parent)):
        levels[i] = len(trace.get_parents(i))

    def get_display_name(idx: int):
        """
        Convert to index in the queue (enque id) to loop_idx for easier understanding.
        """
        if hasattr(trace, "idx2loop_id") and idx in trace.idx2loop_id:
            # FIXME: only keep me after it is stable. Just for compatibility.
            return f"L{trace.idx2loop_id[idx]} ({idx})"
        return f"L{idx}"

    # Add nodes and edges
    edges = []
    parents_record = {}
    for i, parents in enumerate(trace.dag_parent):
        for parent in parents:
            edges.append((get_display_name(parent), get_display_name(i)))
        if len(parents) == 0:
            G.add_node(get_display_name(i))
        parents_record[get_display_name(i)] = [get_display_name(parent) for parent in parents]
    G.add_edges_from(edges)

    # Check if G is a path (a single line)
    is_path = nx.is_path(G, list(nx.topological_sort(G)))
    if is_path:
        # Arrange nodes in a square spiral
        n = len(G.nodes())
        pos = {}
        x, y = 0, 0
        dx, dy = 1, 0
        step = 1
        steps_taken = 0
        steps_in_dir = 1
        dir_changes = 0
        for i, node in enumerate(G.nodes()):
            pos[node] = (x, y)
            x += dx
            y += dy
            steps_taken += 1
            if steps_taken == steps_in_dir:
                steps_taken = 0
                # Change direction: right -> up -> left -> down -> right ...
                dx, dy = -dy, dx
                dir_changes += 1
                if dir_changes % 2 == 0:
                    steps_in_dir += 1
    else:
        # Group nodes by number of ancestors, fewer ancestors are higher up
        layer_nodes = {}
        for idx, lvl in levels.items():
            layer_nodes.setdefault(lvl, []).append(get_display_name(idx))

        # Layout by level: y axis is -lvl, x axis is evenly distributed
        pos = {}

        def parent_avg_pos(node):
            parent_nodes = parents_record.get(node, [])
            parent_xs = [pos[p][0] for p in parent_nodes if p in pos]
            return sum(parent_xs) / len(parent_xs) if parent_xs else 0

        for lvl in sorted(layer_nodes):
            nodes = layer_nodes[lvl]
            # For root nodes, sort directly by index
            if lvl == min(layer_nodes):
                sorted_nodes = sorted(nodes, key=lambda n: int(n[1:].split(" ")[0]))
            else:
                # Sort by average parent x, so children are below their parents
                sorted_nodes = sorted(nodes, key=parent_avg_pos)
            y = -lvl  # y decreases as level increases (children below parents)
            for i, node in enumerate(sorted_nodes):
                if lvl == min(layer_nodes):
                    x = i
                else:
                    # Place child directly below average parent x, offset if multiple at same y
                    avg_x = parent_avg_pos(node)
                    # To avoid overlap, spread siblings a bit if needed
                    x = avg_x + (i - (len(sorted_nodes) - 1) / 2) * 0.5
                pos[node] = (x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = ["tomato" if node in [get_display_name(idx) for idx in merge_loops] else "skyblue" for node in G]
    nx.draw(G, pos, with_labels=True, arrows=True, node_color=color_map, node_size=100, font_size=5, ax=ax)
    return fig


def timeline_figure(times_dict: dict[int, dict[str, dict[Literal["start_time", "end_time"], datetime]]]) -> go.Figure:
    # Prepare data for px.timeline
    timeline_data = []
    step_names = ["exp_gen", "coding", "running", "feedback", "record"]

    # Beautiful color palette with gradients
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA726", "#5A0069"]
    color_map = {step: color for step, color in zip(step_names, colors)}

    for loop_id, steps in times_dict.items():
        for step_name, timing in steps.items():
            if step_name in step_names:
                duration = timing["end_time"] - timing["start_time"]
                timeline_data.append(
                    {
                        "Start": timing["start_time"],
                        "Finish": timing["end_time"],
                        "Step": step_name,
                        "Loop_ID": f"Loop {loop_id}",
                        "Duration": str(duration).split(".")[0],  # Remove microseconds
                    }
                )

    # Create DataFrame and sort by loop ID in descending order
    df = pd.DataFrame(timeline_data)
    df["loop_sort"] = df["Loop_ID"].str.extract("(\d+)").astype(int)
    df = df.sort_values("loop_sort", ascending=False)

    # Create timeline with enhanced styling
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Loop_ID",
        color="Step",
        color_discrete_map=color_map,
        title="🚀 Data Science Loop Timeline",
        hover_data={"Duration": True, "Loop_ID": False, "Step": False},
        hover_name="Step",
    )

    # Enhanced styling and layout
    fig.update_traces(
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.8)"), opacity=0.85),
        width=0.9,  # Increased from 0.8 to make bars thicker and reduce spacing
        hovertemplate="<b>%{hovertext}</b><br>"
        + "Start: %{base}<br>"
        + "End: %{x}<br>"
        + "Duration: %{customdata[0]}<br>"
        + "<extra></extra>",
    )

    # Beautiful layout with gradients and shadows
    fig.update_layout(
        title=dict(text="Data Science Loop Timeline", x=0.0, font=dict(size=24, color="#2C3E50", family="Arial Black")),
        xaxis=dict(
            title="⏰ Time",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(176, 196, 222, 0.4)",
            zeroline=False,
            tickfont=dict(size=12, color="#34495E"),
            title_font=dict(size=14, color="#2C3E50", family="Arial"),
        ),
        yaxis=dict(
            title="🔄 Loop ID",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(176, 196, 222, 0.4)",
            zeroline=False,
            tickfont=dict(size=12, color="#34495E"),
            title_font=dict(size=14, color="#2C3E50", family="Arial"),
        ),
        plot_bgcolor="rgba(248, 249, 250, 0.8)",
        paper_bgcolor="white",
        height=max(200, len(times_dict) * 25),  # Reduced from 300 and 30 to 200 and 25
        margin=dict(l=100, r=60, t=80, b=60),
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            title_font=dict(size=12, color="#2C3E50"),
            font=dict(size=11, color="#34495E"),
            traceorder="normal",
        ),
        font=dict(family="Arial, sans-serif"),
        template="plotly_white",
    )

    # Reorder legend to match step_names order
    fig.data = sorted(
        fig.data, key=lambda trace: step_names.index(trace.name) if trace.name in step_names else len(step_names)
    )

    # Add subtle shadow effect
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="rgba(0,0,0,0.1)", width=2),
        fillcolor="rgba(0,0,0,0.02)",
    )

    return fig


def compare(
    exp_list: list[str] = typer.Option(..., "--exp-list", help="List of experiment names.", show_default=False),
    output: str = typer.Option("merge_base_df.h5", help="Output summary file name."),
    hours: int | None = typer.Option(None, help="if None, use summary.pkl, else summary_{hours}h.pkl"),
    select_best: bool = typer.Option(False, help="Select best experiment for each competition."),
):
    """
    Generate summary and base dataframe for given experiment list, and save to a summary file.
    """
    typer.secho(f"exp_list: {exp_list}", fg=typer.colors.GREEN)
    log_folders = [f"{UI_SETTING.amlt_path}/{exp}/combined_logs" for exp in exp_list]
    summary, base_df = get_summary_df(log_folders, hours=hours)
    if select_best:

        def apply_func(cdf: pd.DataFrame):
            cp = cdf["Competition"].values[0]
            md = get_metric_direction(cp)
            # If SOTA Exp Score (valid, to_submit) column is empty, return the first index
            if cdf["SOTA Exp Score (valid, to_submit)"].dropna().empty:
                return cdf.index[0]
            if md:
                best_idx = cdf["SOTA Exp Score (valid, to_submit)"].idxmax()
            else:
                best_idx = cdf["SOTA Exp Score (valid, to_submit)"].idxmin()
            return best_idx

        best_idxs = base_df.groupby("Competition").apply(apply_func)
        base_df = base_df[base_df.index.isin(best_idxs.values)]
        summary = {k: v for k, v in summary.items() if k in best_idxs.values.tolist()}
    typer.secho(f"Summary keys: {list(summary.keys())}", fg=typer.colors.CYAN)
    typer.secho("Summary DataFrame:", fg=typer.colors.MAGENTA)
    typer.secho(str(base_df), fg=typer.colors.YELLOW)
    base_df.to_hdf(output, "data")
    typer.secho(f"Summary saved to {output}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(compare)
    app()
