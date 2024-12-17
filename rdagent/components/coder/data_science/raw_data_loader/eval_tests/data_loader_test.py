"""
A qualified data loader should support following features
- successfully run
- len(test) == len(test_ids) == submission length
- len(train) == len(y)

Please make sure the stdout is rich enough to support informative feedback
"""

import logging
import pickle

from load_data import load_data

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

X, y, X_test, test_ids = load_data()

# Validate the conditions mentioned in the docstring
assert len(X_test) == len(test_ids), "Mismatch in length of test images and test IDs"
assert len(X) == len(y), "Mismatch in length of training images and labels"

logging.info("Data loader test passed successfully. Length of test images matches length of test IDs.")

with open("data.pkl", "wb") as f:
    pickle.dump((X, y, X_test, test_ids), f)
