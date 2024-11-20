import numpy as np


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

    pred_binary_l = [0 if value < 0.50 else 1 for value in test_pred_l[0]]
    return np.array(pred_binary_l)
