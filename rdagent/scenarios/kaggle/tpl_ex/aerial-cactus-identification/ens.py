import numpy as np


def ens_and_decision(test_pred: np.ndarray, val_pred: np.ndarray, val_label: np.ndarray) -> np.ndarray:
    """
    Handle following:
    1) Ensemble predictions using a simple average.
    2) Make final decsion after ensemble (convert the predictions to final).

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

    pred_binary = [0 if value<0.50 else 1 for value in test_pred]
    return pred_binary
    # return (test_pred + val_pred) / 2
