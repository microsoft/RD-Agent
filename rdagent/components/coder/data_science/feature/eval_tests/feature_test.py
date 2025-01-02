"""
A qualified data loader should support following features
- successfully run
- len(test) == len(test_ids) == submission length
- len(train) == len(y)

Please make sure the stdout is rich enough to support informative feedback
"""

import pickle

import numpy as np
import pandas as pd
from feature import feat_eng
from load_data import load_data

X, y, X_test, test_ids = load_data()
X, y, X_test = feat_eng(X, y, X_test)


# Validate the conditions mentioned in the docstring
assert len(X_test) == len(test_ids), "Mismatch in length of test images and test IDs"
assert len(X) == len(y), "Mismatch in length of training images and labels"

print("Feature Engineering test passed successfully. Length of test images matches length of test IDs.")
