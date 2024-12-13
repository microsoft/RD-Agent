
## Feature Engineering

- Implement a function for feature engineering with the following signature:

```python
def feat_eng(X: np.ndarray, y: np.ndarray | None = None, X_fit: np.ndarray | None = None, y_fit: np.ndarray | None = None, param: object | None = None) -> tuple[np.ndarray, np.ndarray | None, object]:
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
