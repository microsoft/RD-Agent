## Model Workflow

- Implement a function to manage the model workflow with the following signature:

```python
def model_workflow(X: np.ndarray, y: np.ndarray, val_X: np.ndarray = None, val_y: np.ndarray = None, test_X: np.ndarray = None, **hyper_params) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """
    Manages the workflow of a machine learning model, including training, validation
    The testing&validation's inference is included, as well

    - If test/valid exist, output inference on them
    - Follow the hyperparameter if exists
        - Hyperparameters at least has <early stop round>. The code must check if it is given and use it.
        - the returned hyperparameter should align with the input(except the newly generated early stop)
    - Return hyperparameters for retrain if not exists. Hyperparameters should have <early stop round>
    - If valid exist, add <early stop round> to update the hyperparameter


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
    tuple[np.ndarray | None, np.ndarray | None, dict]
        Predictions on the validation data, predictions on the test data
    """
```
- In this task, the shape of input(X of train, valid and test) should be (num_samples, height, width, channels).

- In this task, the shape of output should be (num_samples, num_class), as num_class = 1 here.

- The function should handle data augmentation, model creation, training, and prediction.
