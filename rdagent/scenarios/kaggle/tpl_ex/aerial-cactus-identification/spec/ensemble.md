## Ensemble and Decision Making

- Implement a function for ensemble and decision making with the following signature:

```python
def ensemble_workflow(test_pred_l: list[np.ndarray], val_pred_l: list[np.ndarray], val_label: np.ndarray) -> np.ndarray:
    """
    Handle the following:
    1) Ensemble predictions using a simple average.
    2) Make final decision after ensemble (convert the predictions to final form).

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
        Predictions on the test data.
    """
```

- The function should combine predictions and convert them to a proper format.
