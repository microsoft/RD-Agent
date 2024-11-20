# Specification for Implementing a Kaggle Competition Project

This document outlines the structure and interface protocols for implementing a machine learning project, similar to a Kaggle competition. Follow these guidelines to ensure consistency and maintainability across projects.

## Project Structure

The project should be organized into the following components:

1. **Data Loading** (`load_data.py`): A module responsible for loading and preprocessing raw data.
2. **Feature Engineering**(`feat*.py`): A module for transforming raw data into features suitable for model training.
3. **Model Workflow**(`model*.py`): A module that manages the training, validation, and testing of machine learning models.
4. **Ensemble and Decision Making**(`ens.py`): A module for combining predictions from multiple models and making final decisions.
5. **Workflow**(`main.py`): A script to put the above component together to get the final submission(`submission.csv`)

## Data Loading

- Implement a function to load data from raw files.
- The function should return training images, training labels, test images, and test IDs.

## Feature Engineering

- Implement a function for feature engineering with the following signature:

```python
def feature_eng(X: np.ndarray, y: np.ndarray | None = None, X_fit: np.ndarray | None = None, y_fit: np.ndarray | None = None, param: object | None = None) -> tuple[np.ndarray, np.ndarray | None, object]:
    """
    Perform feature engineering on the input data.

    Parameters:
    - X: np.ndarray
        The input data to be transformed.
    - y: np.ndarray | None
        The target data.
    - X_fit: np.ndarray | None
        Data for fitting the transformation parameters.
    - y_fit: np.ndarray | None
        Target data for fitting.
    - param: object | None
        Pre-fitted parameters for transformation.

    Returns:
    - transformed_data: np.ndarray
        Transformed data.
    - transformed_target: np.ndarray | None
        Transformed target data.
    - fitted_param: object
        Fitted parameters.
    """
```

- Ensure that the feature engineering process is consistent and can be applied to both training and test data.

## Model Workflow

- Implement a function to manage the model workflow with the following signature:

```python
def model_workflow(X: np.ndarray, y: np.ndarray, val_X: np.ndarray = None, val_y: np.ndarray = None, test_X: np.ndarray = None, **hyper_params) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Manages the workflow of a machine learning model, including training, validation, and testing.

    Parameters
    ----------
    X : np.ndarray
        Training data features.
    y : np.ndarray
        Training data labels.
    val_X : np.ndarray, optional
        Validation data features.
    val_y : np.ndarray, optional
        Validation data labels.
    test_X : np.ndarray, optional
        Test data features.
    **hyper_params
        Additional hyperparameters for the model.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        Predictions on the validation data, predictions on the test data
    """
```

- The function should handle data augmentation, model creation, training, and prediction.

## Ensemble and Decision Making

- Implement a function for ensemble and decision making with the following signature:

```python
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
```

- The function should combine predictions and convert them to a binary format.

## Submission

- Implement a script to generate the submission file.
- The script should write predictions to a CSV file in the format required by the competition.

## General Guidelines

- Ensure that all modules and functions are well-documented.
- Follow consistent naming conventions and code style.
- Use type annotations for function signatures to improve code readability and maintainability.
