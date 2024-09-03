import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fea_share_preprocess import preprocess_script
from sklearn.metrics import accuracy_score, matthews_corrcoef


# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DIRNAME = Path(__file__).absolute().resolve().parent


# support various method for metrics calculation
def compute_metrics_for_classification(y_true, y_pred):
    """Compute accuracy metric for classification."""
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def compute_metrics_for_classification(y_true, y_pred):
    """Compute MCC for classification."""
    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc

# 1) Preprocess the data
#TODO 如果已经做过数据预处理了，不需要再做了
X_train, X_valid, y_train, y_valid, X_test, passenger_ids = preprocess_script()

# 2) Auto feature engineering
X_train_l, X_valid_l = [], []
X_test_l = []
for f in DIRNAME.glob("feat*.py"):
    m = __import__(f.name.strip(".py"))
    X_train = m.feat_eng(X_train)
    X_valid = m.feat_eng(X_valid)
    X_test = m.feat_eng(X_test)

    X_train_l.append(X_train)
    X_valid_l.append(X_valid)
    X_test_l.append(X_test)

X_train = pd.concat(X_train_l, axis=1)
X_valid = pd.concat(X_valid_l, axis=1)
X_test = pd.concat(X_test_l, axis=1)


def align_features(train_df, valid_df):
    # Align the features of validation data to the training data
    valid_df = valid_df.reindex(columns=train_df.columns, fill_value=0)
    return valid_df


X_valid = align_features(X_train, X_valid)
X_test = align_features(X_train, X_test)

# 3) Train the model
model_l = []  # list[tuple[model, predict_func,]]
for f in DIRNAME.glob("model*.py"):
    # TODO put select() in model.py: fit(X_train, y_train, X_valid, y_valid)
    m = __import__(f.name.strip(".py"))
    model_l.append((m.fit(X_train, y_train, X_valid, y_valid), m.predict))

# Evaluate the model on the validation set
y_valid_pred_l = []
for model, predict_func in model_l:
    y_valid_pred_l.append(predict_func(model, X_valid))

# Ensemble
# TODO: ensemble method in a script
# Average the predictions and apply a threshold to determine class labels
y_valid_pred = np.mean(y_valid_pred_l, axis=0)
y_valid_pred = (y_valid_pred > 0.5).astype(int)

mcc = compute_metrics_for_classification(y_valid, y_valid_pred)
print("Final on validation set: ", mcc)

# Save the validation accuracy
pd.Series(data=[mcc], index=["MCC"]).to_csv(
    "submission_score.csv"
)

# Make predictions on the test set and save them
y_test_pred_bool_l = []
for m, m_pred in model_l:
    y_test_pred_bool_l.append(
        m_pred(m, X_test).astype(int)
    )  # TODO Make this an ensemble. Currently it uses the last prediction

y_test_pred = np.mean(y_test_pred_bool_l, axis=0)
y_test_pred = (y_test_pred > 0.5).astype(int)  # TODO Make it a module. Ensemble prediction

y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)  # 将整数转换回 'e' 或 'p'
submission_result = pd.DataFrame({"id": passenger_ids, "class": y_test_pred_labels})

# submit predictions for the test set
submission_result.to_csv("submission.csv", index=False)