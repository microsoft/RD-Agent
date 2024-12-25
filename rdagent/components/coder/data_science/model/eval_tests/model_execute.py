"""
adapt for cv models
"""

import os
import pickle
import traceback

import numpy as np
from load_data import load_data
from model01 import model_workflow
from sklearn.model_selection import train_test_split

X, y, test_X, test_ids = load_data()
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)


"""train_X = np.random.rand(8, 64, 64, 3)
train_y = np.random.rand(8, 1)
val_X = np.random.rand(8, 64, 64, 3)
val_y = np.random.rand(8, 1)
test_X = np.random.rand(8, 64, 64, 3)"""

print("The first execution begins.\n")
# Call model_workflow
val_pred, test_pred, hypers = model_workflow(
    X=train_X,
    y=train_y,
    val_X=val_X,
    val_y=val_y,
    test_X=None,
)
# val_pred = np.random.rand(8, 1)
# test_pred = np.random.rand(8, 1)

execution_feedback_str = "The first Execution successful.\n"
if val_pred is not None:
    execution_feedback_str += f"Validation predictions shape: {val_pred.shape}\n"
else:
    execution_feedback_str += "Validation predictions are None.\n"
if test_pred is not None:
    execution_feedback_str += f"Test predictions shape: {test_pred.shape}\n"
else:
    execution_feedback_str += "Test predictions are None.\n"
if hypers is not None:
    execution_feedback_str += f"Hyperparameters:{hypers}\n"
else:
    execution_feedback_str += "Hyperparameters are None.\n"
print(execution_feedback_str)

print("The second execution begins.\n")
val_pred, test_pred, finalhypers = model_workflow(X=train_X, y=train_y, val_X=None, val_y=None, test_X=test_X, **hypers)
execution_feedback_str = "The second Execution successful.\n"
if val_pred is not None:
    execution_feedback_str += f"Validation predictions shape: {val_pred.shape}\n"
else:
    execution_feedback_str += "Validation predictions are None.\n"
if test_pred is not None:
    execution_feedback_str += f"Test predictions shape: {test_pred.shape}\n"
else:
    execution_feedback_str += "Test predictions are None.\n"
if hypers is not None:
    execution_feedback_str += f"Hyperparameters:{finalhypers}\n"
else:
    execution_feedback_str += "Hyperparameters are None.\n"
print(execution_feedback_str)
