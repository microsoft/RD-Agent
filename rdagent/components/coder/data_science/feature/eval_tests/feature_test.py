"""
A qualified data loader should support following features
- successfully run
- len(test) == len(test_ids) == submission length
- len(train) == len(y)

Please make sure the stdout is rich enough to support informative feedback
"""

import logging
import pickle

import numpy as np
import pandas as pd
from feat01 import feat_eng

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load data
from load_data import load_from_raw_data
from sklearn.model_selection import train_test_split

X, y, X_test, test_ids = load_from_raw_data()

X, y, X_param = feat_eng(X, y)
X_test, _, _ = feat_eng(X_test, param=X_param)


# Validate the conditions mentioned in the docstring
assert len(X_test) == len(test_ids), "Mismatch in length of test images and test IDs"
assert len(X) == len(y), "Mismatch in length of training images and labels"
# Check for missing values
if isinstance(X, pd.DataFrame):
    assert not X.isnull().values.any(), "Missing values found in training data"
    assert not X_test.isnull().values.any(), "Missing values found in test data"
    assert not y.isnull().values.any(), "Missing values found in labels"
elif isinstance(X, np.ndarray):
    assert not np.isnan(X).any(), "Missing values found in training data"
    assert not np.isnan(X_test).any(), "Missing values found in test data"
    assert not np.isnan(y).any(), "Missing values found in labels"
else:
    raise TypeError("Unsupported data type for X and y")

logging.info("Data loader test passed successfully. Length of test images matches length of test IDs.")

with open("data.pkl", "wb") as f:
    pickle.dump((X, y, X_test, test_ids), f)
