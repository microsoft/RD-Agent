```python
import pandas as pd

# Function Definition

def feat_eng(X, y, X_test):
    """
    Perform feature engineering on the Forest Cover Type Prediction dataset.

    This function applies necessary transformations to the training and test datasets to prepare them for model training and evaluation. It handles missing values, scales features, and applies any competition-specific feature engineering steps.

    Parameters:
        X (pd.DataFrame): Train data to be transformed with shape (n_samples, n_features).
        y (pd.Series): Train label data with shape (n_samples,).
        X_test (pd.DataFrame): Test data with shape (n_test_samples, n_features).

    Returns:
        X_transformed (pd.DataFrame): Transformed train data with shape (n_samples, -1).
        y_transformed (pd.Series): Transformed train label data with shape (n_samples,).
        X_test_transformed (pd.DataFrame): Transformed test data with shape (n_test_samples, -1).

    Notes:
        - Ensure the sample size of the train data and the test data remains consistent.
        - The input shape and output shape should generally be the same, though some columns may be added or removed.
        - Avoid data leakage by only using features derived from training data.
        - Handle missing values and outliers appropriately.
        - Ensure consistency between feature data types and transformations.
        - Apply competition-specific feature engineering steps as needed.
        - Utilize GPU support or multi-processing to accelerate the feature engineering process if necessary.
    """
    # Code implementation goes here
    pass
```