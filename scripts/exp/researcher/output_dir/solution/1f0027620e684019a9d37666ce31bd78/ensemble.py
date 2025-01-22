import numpy as np
import pandas as pd
from typing import Dict

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
    # Check if validation labels are provided
    if val_label is None or len(val_label) == 0:
        raise ValueError("Validation labels are required")

    # Check if prediction dictionaries are not empty
    if not test_preds_dict or not val_preds_dict:
        raise ValueError("Prediction dictionaries cannot be empty")

    # Ensure all predictions have consistent shapes
    n_val_samples = len(val_label)
    n_test_samples = None
    for model_name, val_pred in val_preds_dict.items():
        if len(val_pred) != n_val_samples:
            raise ValueError(f"Validation predictions from model {model_name} do not match the length of validation labels")
        if n_test_samples is None:
            n_test_samples = len(test_preds_dict[model_name])
        elif len(test_preds_dict[model_name]) != n_test_samples:
            raise ValueError(f"Test predictions from model {model_name} do not match the length of other test predictions")

    # Calculate validation accuracy for each model
    val_accuracies = {}
    for model_name, val_pred in val_preds_dict.items():
        accuracy = np.mean(val_pred == val_label)
        val_accuracies[model_name] = accuracy

    # Determine weights based on validation accuracy
    total_accuracy = sum(val_accuracies.values())
    weights = {model_name: accuracy / total_accuracy for model_name, accuracy in val_accuracies.items()}

    # Get unique classes from validation labels
    unique_classes = np.unique(val_label)
    num_classes = len(unique_classes)

    # Weighted average ensemble for validation set
    val_ensemble_pred = np.zeros((n_val_samples, num_classes), dtype=float)
    for model_name, weight in weights.items():
        val_pred_one_hot = pd.get_dummies(val_preds_dict[model_name], columns=unique_classes).reindex(columns=unique_classes, fill_value=0).to_numpy()
        val_ensemble_pred += weight * val_pred_one_hot.astype(float)
    val_ensemble_pred = np.argmax(val_ensemble_pred, axis=1)

    # Calculate ensemble accuracy on validation set
    ensemble_accuracy = np.mean(val_ensemble_pred == val_label)

    # Weighted average ensemble for test set
    test_ensemble_pred = np.zeros((n_test_samples, num_classes), dtype=float)
    for model_name, weight in weights.items():
        test_pred_one_hot = pd.get_dummies(test_preds_dict[model_name], columns=unique_classes).reindex(columns=unique_classes, fill_value=0).to_numpy()
        test_ensemble_pred += weight * test_pred_one_hot.astype(float)
    final_pred = np.argmax(test_ensemble_pred, axis=1)

    # Save validation accuracies and ensemble accuracy to CSV
    results_df = pd.DataFrame(list(val_accuracies.items()), columns=['Model', 'Validation Accuracy'])
    results_df.loc[len(results_df)] = ['Ensemble', ensemble_accuracy]
    results_df.to_csv('scores.csv', index=False)

    return pd.Series(final_pred)