import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ens_and_decision(test_pred_l: list[np.ndarray], val_pred_l: list[np.ndarray], val_label: np.ndarray) -> np.ndarray:
    """
    Handle the following:
    1) Ensemble predictions using a simple average.
    2) Make final decision after ensemble (convert the predictions to final binary form).

    Parameters
    ----------
    test_pred_l : list[np.ndarray]
        List of predictions on the test data.
    val_pred_l : list[np.ndarray]
        List of predictions on the validation data.
    val_label : np.ndarray
        True labels of the validation data.

    Returns
    -------
    np.ndarray
        Binary predictions on the test data.
    """

    scores = []
    for id, val_pred in enumerate(val_pred_l):
        scores.append(roc_auc_score(val_label, val_pred))

    # Normalize the scores to get weights
    total_score = sum(scores)
    weights = [score / total_score for score in scores]

    # Weighted average of test predictions
    weighted_test_pred = np.zeros_like(test_pred_l[0])
    for weight, test_pred in zip(weights, test_pred_l):
        weighted_test_pred += weight * test_pred

    weighted_valid_pred = np.zeros_like(val_pred_l[0])
    for weight, val_pred in zip(weights, val_pred_l):
        weighted_valid_pred += weight * val_pred

    weighted_valid_pred_score = roc_auc_score(val_label, weighted_valid_pred)

    scores_df = pd.DataFrame(
        {
            "Model": list(range(len(val_pred_l))) + ["weighted_average_ensemble"],
            "AUROC": scores + [weighted_valid_pred_score],
        }
    )
    scores_df.to_csv("scores.csv", index=False)

    pred_binary_l = [0 if value < 0.50 else 1 for value in weighted_test_pred]
    return np.array(pred_binary_l)
