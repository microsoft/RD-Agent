import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
import json
import numpy as np


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
    # metric_inputs = prepare_for_metric(submission, answers)
    # return mean_absolute_error(metric_inputs["y_true"], metric_inputs["y_score"])


if __name__ == "__main__":
    submission_path = "submission.csv"
    gt_submission_path = "submission_test.csv"
    submission = pd.read_csv(submission_path)
    answers = pd.read_csv(gt_submission_path)
    score = grade(submission=submission, answers=answers)

    # must json,
    print(
        json.dumps(
            {
                "competition_id": "playground-series-s4e9",
                "score": score,
            }
        )
    )
