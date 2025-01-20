"""
A qualified data loader should support following features
- successfully run
- len(test) == len(test_ids) == submission length
- len(train) == len(y)

Please make sure the stdout is rich enough to support informative feedback
"""

import pickle
from load_data import load_data

X, y, X_test, test_ids = load_data()

def get_length(data):
    return len(data) if isinstance(data, list) else data.shape[0]

assert get_length(X_test) == get_length(test_ids), "Mismatch in length of test images and test IDs"
assert get_length(X) == get_length(y), "Mismatch in length of training images and labels"

print("Data loader test passed successfully. Length of test images matches length of test IDs.")
