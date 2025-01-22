```python
from typing import Dict
import pandas as pd

# Function Definition

def ens_and_decision(test_preds_dict: Dict[str, pd.DataFrame], val_preds_dict: Dict[str, pd.DataFrame], val_label: pd.Series):
    """
    Perform ensemble of multiple model predictions and make final decision for the Forest Cover Type Prediction competition.

    This function aggregates predictions from multiple models using an ensemble strategy and generates the final prediction for the test data. It also evaluates each model and the ensemble strategy on validation data.

    Parameters:
        test_preds_dict (Dict[str, pd.DataFrame]): A dictionary of test predictions from different models. The key is the model file name, and the value is a DataFrame with shape (n_test_samples,).
        val_preds_dict (Dict[str, pd.DataFrame]): A dictionary of validation predictions from different models. The key is the model file name, and the value is a DataFrame with shape (n_val_samples,).
        val_label (pd.Series): Validation label with shape (n_val_samples,).

    Returns:
        final_pred (pd.Series): Ensemble prediction for the test data with shape (n_test_samples,).

    Notes:
        - Ensure all predictions in `test_preds_dict` and `val_preds_dict` have consistent shapes and dimensions.
        - Verify that `val_label` is provided and matches the length of `val_preds_dict` predictions.
        - Handle empty or invalid inputs gracefully with appropriate error messages.
        - Calculate the metric for each model and ensemble strategy, and save the results in a CSV file.
        - Clearly define how the ensemble predictions are aggregated (e.g., majority voting, weighted average).
        - Avoid introducing biases or overfitting during decision-making.
        - Avoid using progress bars (e.g., `tqdm`) in the implementation.
    """
    # Code implementation goes here
    pass
```