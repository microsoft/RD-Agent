import json
import re
from collections import defaultdict
from pathlib import Path

import fire
import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.core.experiment import FBWorkspace
from rdagent.core.proposal import ExperimentFeedback
from rdagent.log.storage import FileStorage
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.kaggle.kaggle_crawler import score_rank
from rdagent.utils.env import DockerEnv, MLEBDockerConf

de = get_ds_env("mlebench")
de.conf.extra_volumes = {f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data"}
de.prepare()


def extract_mle_json(log_content: str) -> dict | None:
    match = re.search(r"\{.*\}", log_content, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return None


def save_grade_info(log_trace_path: Path):
    trace_storage = FileStorage(log_trace_path)
    for msg in trace_storage.iter_msg():
        if "competition" in msg.tag:
            competition = msg.content

        if "running" in msg.tag:
            if isinstance(msg.content, DSExperiment):
                mle_score_str = msg.content.experiment_workspace.execute(
                    env=de,
                    entry=f"mlebench grade-sample submission.csv {competition} --data-dir /mle/data | tee mle_score.txt",
                )
                msg.content.experiment_workspace.execute(env=de, entry="chmod 777 mle_score.txt")
                trace_storage.log(mle_score_str, name=f"{msg.tag}.mle_score")


def is_valid_session(p: Path) -> bool:
    return p.is_dir() and p.joinpath("__session__").exists()


def save_all_grade_info(log_folder):
    for log_trace_path in log_folder.iterdir():
        if is_valid_session(log_trace_path):
            save_grade_info(log_trace_path)


def summarize_folder(log_folder: Path):
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
        for msg in FileStorage(log_trace_path).iter_msg():  # messages in log trace
            if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
                if "competition" in msg.tag:
                    stat[log_trace_path.name]["competition"] = msg.content

                    # get threshold scores
                    workflowexp = FBWorkspace()
                    stdout = workflowexp.execute(
                        env=de,
                        entry=f"mlebench grade-sample None {stat[log_trace_path.name]['competition']} --data-dir /mle/data",
                    )
                    grade_output = extract_mle_json(stdout)
                    if grade_output:
                        bronze_threshold = grade_output["bronze_threshold"]
                        silver_threshold = grade_output["silver_threshold"]
                        gold_threshold = grade_output["gold_threshold"]
                        median_threshold = grade_output["median_threshold"]

                if "direct_exp_gen" in msg.tag and isinstance(msg.content, DSExperiment):
                    loop_num += 1

                if "running" in msg.tag:
                    if isinstance(msg.content, DSExperiment):
                        submission_path = msg.content.experiment_workspace.workspace_path / "submission.csv"
                        if submission_path.exists():
                            made_submission_num += 1
                            scores_path = msg.content.experiment_workspace.workspace_path / "scores.csv"
                            valid_scores[loop_num - 1] = pd.read_csv(scores_path, index_col=0)
                    elif "mle_score" in msg.tag:
                        grade_output = extract_mle_json(msg.content)
                        if grade_output:
                            if grade_output["score"] is not None:
                                test_scores[loop_num - 1] = grade_output["score"]
                                _, test_ranks[loop_num - 1] = score_rank(
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
    if (log_folder / "summary.pkl").exists():
        (log_folder / "summary.pkl").unlink()
        print("Old summary file removed.")
    pd.to_pickle(stat, log_folder / "summary.pkl")


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
