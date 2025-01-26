import json
import re
from collections import defaultdict
from pathlib import Path

import fire
import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.log.storage import FileStorage
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.utils.env import DockerEnv, MLEBDockerConf

mle_de_conf = MLEBDockerConf()
mle_de_conf.extra_volumes = {
    f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data",
}
de = DockerEnv(conf=mle_de_conf)
de.prepare()


def extract_mle_json(log_content):
    match = re.search(r"\{.*\}", log_content, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return None


def save_grade_info(log_trace_path: Path):
    for msg in FileStorage(log_trace_path).iter_msg():
        if "competition" in msg.tag:
            competition = msg.content

        if "running" in msg.tag:
            if isinstance(msg.content, DSExperiment):
                msg.content.experiment_workspace.execute(
                    env=de,
                    entry=f"mlebench grade-sample submission.csv {competition} --data-dir /mle/data > mle_score.txt 2>&1",
                )
                msg.content.experiment_workspace.execute(env=de, entry="chmod 777 mle_score.txt")


def save_all_grade_info(log_folder):
    for log_trace_path in log_folder.iterdir():
        if log_trace_path.is_dir():
            save_grade_info(log_trace_path)


def summarize_folder(log_folder: Path):
    log_folder = Path(log_folder)
    stat = defaultdict(dict)
    for log_trace_path in log_folder.iterdir():  # One log trace
        if not log_trace_path.is_dir():
            continue
        loop_num = 0
        made_submission_num = 0
        test_scores = {}
        valid_scores = {}
        medal = "None"
        success_loop_num = 0

        for msg in FileStorage(log_trace_path).iter_msg():  # messages in log trace
            if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
                if "competition" in msg.tag:
                    stat[log_trace_path.name]["competition"] = msg.content

                if "direct_exp_gen" in msg.tag and isinstance(msg.content, DSExperiment):
                    loop_num += 1

                if "running" in msg.tag:
                    if isinstance(msg.content, DSExperiment):
                        submission_path = msg.content.experiment_workspace.workspace_path / "submission.csv"
                        if submission_path.exists():
                            made_submission_num += 1
                            scores_path = msg.content.experiment_workspace.workspace_path / "scores.csv"
                            valid_scores[loop_num - 1] = pd.read_csv(scores_path, index_col=0)
                            grade_output_path = msg.content.experiment_workspace.workspace_path / "mle_score.txt"
                            if not grade_output_path.exists():
                                raise FileNotFoundError(
                                    f"mle_score.txt in {grade_output_path} not found, genarate it first!"
                                )
                            grade_output = extract_mle_json(grade_output_path.read_text())
                            if grade_output and grade_output["score"] is not None:
                                test_scores[loop_num - 1] = grade_output["score"]
                                if grade_output["any_medal"]:
                                    medal = (
                                        "gold"
                                        if grade_output["gold_medal"]
                                        else "silver" if grade_output["silver_medal"] else "bronze"
                                    )

                if "feedback" in msg.tag and "evolving" not in msg.tag:
                    if bool(msg.content):
                        success_loop_num += 1

        stat[log_trace_path.name].update(
            {
                "loop_num": loop_num,
                "made_submission_num": made_submission_num,
                "test_scores": test_scores,
                "valid_scores": valid_scores,
                "medal": medal,
                "success_loop_num": success_loop_num,
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
