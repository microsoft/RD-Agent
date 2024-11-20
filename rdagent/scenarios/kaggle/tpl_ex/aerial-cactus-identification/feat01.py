import numpy as np


def feat_eng(
    X: np.ndarray,
    y: np.ndarray | None = None,
    X_fit: np.ndarray | None = None,
    y_fit: np.ndarray | None = None,
    param: object | None = None,
) -> tuple[np.ndarray, np.ndarray | None, object]:
    """
    Perform feature engineering on the input data.

    Parameters:
    - X: np.ndarray
        The input data to be transformed. A concrete example could be:
        array([[[[207, 194, 203],
                ...,
                [191, 183, 164],
                [176, 168, 149],
                [181, 173, 152]]]], dtype=uint8)
    - y: np.ndarray | None
        The target data. A concrete example could be:
        array([1, 0, 1, 0, 1, 1, ..., ])
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

    Notes:
    - Some preprocessing (e.g., data selection) is based on y.

    Typical usage:
    .. code-block:: python

        X_transformed, y_transformed, fitted_param = feat_eng(X, y, X, y)
        X_test_transformed, _, _ = feat_eng(X_test, fitted_param)
    """
    # This is an example of identity feature transformation.
    # We'll not change the content of the data, but we'll demonstrate the typical workflow of feature engineering.
    if param is None:
        # Get parameters from the X_fit and y_fit
        pass
    # Use the fitted parameters to transform the data X, y
    return X, y, param
