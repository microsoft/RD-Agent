import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

def ens_and_decision(test_pred: np.ndarray, val_pred: np.ndarray, val_label: np.ndarray) -> np.ndarray:
    """
    Handle following:
    1) Ensemble predictions using a simple average.
    2) Make final decsion after ensemble.

    Parameters
    ----------
    test_pred : np.ndarray
        Predictions on the test data.
    val_pred : np.ndarray
        Predictions on the validation data.
    val_label : np.ndarray
        True labels of the validation data.
    Returns
    -------
    np.ndarray
        Predictions on the test data.
    """
    # Check if there is only one model's predictions
    if len(test_pred.shape) == 1:
        test_pred = test_pred.reshape(-1, 1)
        val_pred = val_pred.reshape(-1, 1)

    # Calculate AUROC scores for each model
    auroc_scores = [roc_auc_score(val_label, val_pred[:, i]) for i in range(val_pred.shape[1])]
    
    # Normalize AUROC scores to sum to 1
    auroc_scores = np.array(auroc_scores)
    weights = auroc_scores / np.sum(auroc_scores)
    
    weighted_val_pred = np.dot(val_pred, weights)
    weighted_auroc = roc_auc_score(val_label, weighted_val_pred)
    auroc_scores = np.append(auroc_scores, weighted_auroc)
    
    # Save auroc_scores andto a CSV file
    pd.Series(auroc_scores, name='auroc').to_csv('scores.csv', index=False)
    
    # Weighted average of predictions
    weighted_test_pred = np.dot(test_pred, weights)
    
    
    pred_binary = [0 if value<0.50 else 1 for value in weighted_test_pred]
    return pred_binary
    # return (test_pred + val_pred) / 2
