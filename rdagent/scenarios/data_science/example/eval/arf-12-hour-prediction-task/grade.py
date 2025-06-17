import json

import pandas as pd
from sklearn.metrics import roc_auc_score


class InvalidSubmissionError(Exception):
    """
    A custom exception for when the agent submission cannot be graded.
    """

    pass


def prepare_for_auroc_metric(submission: pd.DataFrame, answers: pd.DataFrame, id_col: str, target_col: str) -> dict:

    # Answers checks
    assert id_col in answers.columns, f"answers dataframe should have an {id_col} column"
    assert target_col in answers.columns, f"answers dataframe should have a {target_col} column"

    # Submission checks
    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission should have an {id_col} column")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission should have a {target_col} column")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(f"Submission and answers should have the same number of rows")
    try:
        pd.to_numeric(submission[target_col])
    except ValueError:
        raise InvalidSubmissionError(
            f"Expected {target_col} column to be numeric, got {submission[target_col].dtype} instead"
        )
    if submission[target_col].min() < 0 or submission[target_col].max() > 1:
        raise InvalidSubmissionError(
            f"Submission {target_col} column should contain probabilities,"
            " and therefore contain values between 0 and 1 inclusive"
        )
    # Sort
    submission = submission.sort_values(id_col)
    answers = answers.sort_values(id_col)

    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError(f"Submission and answers should have the same {id_col} values")

    roc_auc_inputs = {
        "y_true": answers[target_col].to_numpy(),
        "y_score": submission[target_col].to_numpy(),
    }

    return roc_auc_inputs


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    roc_auc_inputs = prepare_for_auroc_metric(
        submission=submission, answers=answers, id_col="ID", target_col="ARF_LABEL"
    )
    return roc_auc_score(y_true=roc_auc_inputs["y_true"], y_score=roc_auc_inputs["y_score"])


if __name__ == "__main__":
    submission_path = "submission.csv"
    gt_submission_path = "submission_test.csv"
    submission = pd.read_csv(submission_path)
    answers = pd.read_csv(gt_submission_path)
    score = grade(submission=submission, answers=answers)

    print(
        json.dumps(
            {
                "competition_id": "arf-12-hour-prediction-task",
                "score": score,
            }
        )
    )
