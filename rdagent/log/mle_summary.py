import pickle
import traceback
from collections import defaultdict
from pathlib import Path

import fire
import pandas as pd

from rdagent.core.experiment import FBWorkspace
from rdagent.core.proposal import ExperimentFeedback
from rdagent.log.storage import FileStorage
from rdagent.log.utils import extract_json, extract_loopid_func_name, is_valid_session
from rdagent.log.utils.folder import get_first_session_file_after_duration
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.test_eval import (
    MLETestEval,
    NoTestEvalError,
    get_test_eval,
)
from rdagent.scenarios.kaggle.kaggle_crawler import score_rank
from rdagent.utils.workflow import LoopBase

test_eval = get_test_eval()

is_mle = isinstance(test_eval, MLETestEval)


def save_grade_info(log_trace_path: Path):
    trace_storage = FileStorage(log_trace_path)
    for msg in trace_storage.iter_msg():
        if "competition" in msg.tag:
            competition = msg.content

        if "running" in msg.tag:
            if isinstance(msg.content, DSExperiment):
                # TODO:  mle_score.txt is not a general name now.
                # Please use a more general name like test_score.txt
                try:
                    mle_score_str = test_eval.eval(competition, msg.content.experiment_workspace)
                    trace_storage.log(
                        mle_score_str, tag=f"{msg.tag}.mle_score.pid", save_type="pkl", timestamp=msg.timestamp
                    )
                except Exception as e:
                    print(f"Error in {log_trace_path}: {e}", traceback.format_exc())


def save_all_grade_info(log_folder):
    for log_trace_path in Path(log_folder).iterdir():
        if is_valid_session(log_trace_path):
            try:
                save_grade_info(log_trace_path)
            except NoTestEvalError as e:
                print(f"Error in {log_trace_path}: {e}", traceback.format_exc())


def _get_loop_and_fn_after_hours(log_folder: Path, hours: int):
    stop_session_fp = get_first_session_file_after_duration(log_folder, f"{hours}h")

    with stop_session_fp.open("rb") as f:
        session_obj: LoopBase = pickle.load(f)

    loop_trace = session_obj.loop_trace
    stop_li = max(loop_trace.keys())
    last_loop = loop_trace[stop_li]
    last_step = last_loop[-1]
    stop_fn = session_obj.steps[last_step.step_idx]
    print(f"Stop Loop: {stop_li=}, {stop_fn=}")
    files = sorted(
        (log_folder / "__session__").glob("*/*_*"), key=lambda f: (int(f.parent.name), int(f.name.split("_")[0]))
    )

    print(f"Max Session: {files[-1:]=}")
    return stop_li, stop_fn


def summarize_folder(log_folder: Path, hours: int | None = None):
    """
    Summarize the log folder and save the summary as a pickle file.
    Args:
        log_folder (Path): The path to the log folder (contains many log traces).
        hours (int | None): The number of hours to stat. If None, stat all.
    """
    log_folder = Path(log_folder)
    stat = defaultdict(dict)
    for log_trace_path in log_folder.iterdir():  # One log trace
        if not is_valid_session(log_trace_path):
            continue
        loop_num = 0
        made_submission_num = 0
        valid_submission_num = 0
        above_median_num = 0
        get_medal_num = 0
        bronze_num = 0
        silver_num = 0
        gold_num = 0
        test_scores = {}
        test_ranks = {}
        valid_scores = {}
        bronze_threshold = 0.0
        silver_threshold = 0.0
        gold_threshold = 0.0
        median_threshold = 0.0
        success_loop_num = 0

        sota_exp_stat = ""
        sota_exp_score = None
        sota_exp_rank = None
        grade_output = None

        if hours:
            stop_li, stop_fn = _get_loop_and_fn_after_hours(log_trace_path, hours)
        msgs = [(msg, extract_loopid_func_name(msg.tag)) for msg in FileStorage(log_trace_path).iter_msg()]
        msgs = [(msg, int(loop_id) if loop_id else loop_id, fn) for msg, (loop_id, fn) in msgs]
        msgs.sort(key=lambda m: m[1] if m[1] else -1)  # sort by loop id
        for msg, loop_id, fn in msgs:  # messages in log trace
            if loop_id:
                loop_num = max(loop_id + 1, loop_num)
            if hours and loop_id == stop_li and fn == stop_fn:
                break
            if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
                if "competition" in msg.tag:
                    stat[log_trace_path.name]["competition"] = msg.content

                    # get threshold scores
                    workflowexp = FBWorkspace()
                    if is_mle:
                        stdout = workflowexp.execute(
                            env=test_eval.env,
                            entry=f"mlebench grade-sample None {stat[log_trace_path.name]['competition']} --data-dir /mle/data",
                        )
                        grade_output = extract_json(stdout)
                        if grade_output:
                            bronze_threshold = grade_output["bronze_threshold"]
                            silver_threshold = grade_output["silver_threshold"]
                            gold_threshold = grade_output["gold_threshold"]
                            median_threshold = grade_output["median_threshold"]

                if "running" in msg.tag:
                    if isinstance(msg.content, DSExperiment):
                        if msg.content.result is not None:
                            valid_scores[loop_id] = msg.content.result
                    elif "mle_score" in msg.tag:
                        grade_output = extract_json(msg.content)
                        if grade_output:
                            if grade_output["submission_exists"]:
                                made_submission_num += 1
                            if grade_output["score"] is not None:
                                test_scores[loop_id] = grade_output["score"]
                                if is_mle:
                                    _, test_ranks[loop_id] = score_rank(
                                        stat[log_trace_path.name]["competition"], grade_output["score"]
                                    )
                            if grade_output["valid_submission"]:
                                valid_submission_num += 1
                            if grade_output["above_median"]:
                                above_median_num += 1
                            if grade_output["any_medal"]:
                                get_medal_num += 1
                            if grade_output["bronze_medal"]:
                                bronze_num += 1
                            if grade_output["silver_medal"]:
                                silver_num += 1
                            if grade_output["gold_medal"]:
                                gold_num += 1

                if "feedback" in msg.tag and "evolving" not in msg.tag:
                    if isinstance(msg.content, ExperimentFeedback) and bool(msg.content):
                        success_loop_num += 1

                        if grade_output:  # sota exp's grade output
                            if grade_output["gold_medal"]:
                                sota_exp_stat = "gold"
                            elif grade_output["silver_medal"]:
                                sota_exp_stat = "silver"
                            elif grade_output["bronze_medal"]:
                                sota_exp_stat = "bronze"
                            elif grade_output["above_median"]:
                                sota_exp_stat = "above_median"
                            elif grade_output["valid_submission"]:
                                sota_exp_stat = "valid_submission"
                            elif grade_output["submission_exists"]:
                                sota_exp_stat = "made_submission"
                            if grade_output["score"] is not None:
                                sota_exp_score = grade_output["score"]
                                if is_mle:
                                    _, sota_exp_rank = score_rank(
                                        stat[log_trace_path.name]["competition"], grade_output["score"]
                                    )

        stat[log_trace_path.name].update(
            {
                "loop_num": loop_num,
                "made_submission_num": made_submission_num,
                "valid_submission_num": valid_submission_num,
                "above_median_num": above_median_num,
                "get_medal_num": get_medal_num,
                "bronze_num": bronze_num,
                "silver_num": silver_num,
                "gold_num": gold_num,
                "test_scores": test_scores,
                "test_ranks": test_ranks,
                "valid_scores": valid_scores,
                "success_loop_num": success_loop_num,
                "sota_exp_stat": sota_exp_stat,
                "sota_exp_score": sota_exp_score,
                "sota_exp_rank": sota_exp_rank,
                "bronze_threshold": bronze_threshold,
                "silver_threshold": silver_threshold,
                "gold_threshold": gold_threshold,
                "median_threshold": median_threshold,
            }
        )

    # Save the summary
    save_name = f"summary_{hours}h.pkl" if hours else "summary.pkl"
    save_p = log_folder / save_name
    if save_p.exists():
        save_p.unlink()
        print(f"Old {save_name} removed.")
    pd.to_pickle(stat, save_p)


# {
#     "competition_id": "stanford-covid-vaccine",
#     "score": null,
#     "gold_threshold": 0.34728,
#     "silver_threshold": 0.35175,
#     "bronze_threshold": 0.3534,
#     "median_threshold": 0.363095,
#     "any_medal": false,
#     "gold_medal": false,
#     "silver_medal": false,
#     "bronze_medal": false,
#     "above_median": false,
#     "submission_exists": true,
#     "valid_submission": false,
#     "is_lower_better": true,
#     "created_at": "2025-01-21T11:59:33.788201",
#     "submission_path": "submission.csv"
# }


def grade_summary(log_folder):
    log_folder = Path(log_folder)
    save_all_grade_info(log_folder)
    summarize_folder(log_folder)


if __name__ == "__main__":
    fire.Fire(
        {
            "grade": save_all_grade_info,
            "summary": summarize_folder,
            "grade_summary": grade_summary,
        }
    )
