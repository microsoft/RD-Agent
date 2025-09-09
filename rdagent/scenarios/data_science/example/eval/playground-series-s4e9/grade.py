import datetime
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


class InvalidSubmissionError(Exception):
    """
    A custom exception for when the agent submission cannot be graded.
    """

    pass


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:

    if "id" not in submission.columns or "id" not in answers.columns:
        raise InvalidSubmissionError("Both submission and answers DataFrames must contain an 'id' column.")

    if "price" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must contain 'price' columns.")

    assert "price" in answers.columns, "Answers DataFrame must contain 'price' columns."

    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission must be the same length as the answers.")

    answers_sorted = answers.sort_values("id")
    submission_sorted = submission.sort_values("id")

    if (submission_sorted["id"].values != answers_sorted["id"].values).any():
        raise InvalidSubmissionError("Submission and answers have mismatched 'id' columns")

    y_true = answers_sorted[["price"]].to_numpy()
    y_score = submission_sorted[["price"]].to_numpy()

    return {"y_true": y_true, "y_score": y_score}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    metric_inputs = prepare_for_metric(submission, answers)
    return np.sqrt(mean_squared_error(metric_inputs["y_true"], metric_inputs["y_score"]))


if __name__ == "__main__":
    submission_path = "submission.csv"
    gt_submission_path = "submission_test.csv"
    submission = pd.read_csv(submission_path)
    answers = pd.read_csv(gt_submission_path)
    score = grade(submission=submission, answers=answers)

    # This `thresholds` can be customized according to the leaderboard page of the Kaggle website and your own needs.
    # Refs: https://www.kaggle.com/competitions/playground-series-s4e9/leaderboard
    thresholds = {
        "gold": 62917.05988,
        "silver": 62945.91714,
        "bronze": 62958.13747,
        "median": 63028.69429,
    }

    # The output must be in json format. To configure the full output,
    # you can run the command `rdagent grade_summary --log-folder` to summarize the scores at the end of the program.
    # If you don't need it, you can just provide the `competition_id`` and `score``.
    print(
        json.dumps(
            {
                "competition_id": "arf-12-hours-prediction-task",
                "score": score,
                "gold_threshold": thresholds["gold"],
                "silver_threshold": thresholds["silver"],
                "bronze_threshold": thresholds["bronze"],
                "median_threshold": thresholds["median"],
                "any_medal": bool(score >= thresholds["bronze"]),
                "gold_medal": bool(score >= thresholds["gold"]),
                "silver_medal": bool(score >= thresholds["silver"]),
                "bronze_medal": bool(score >= thresholds["bronze"]),
                "above_median": bool(score >= thresholds["median"]),
                "submission_exists": True,
                "valid_submission": True,
                "is_lower_better": False,
                "created_at": str(datetime.datetime.now().isoformat()),
                "submission_path": submission_path,
            }
        )
    )
