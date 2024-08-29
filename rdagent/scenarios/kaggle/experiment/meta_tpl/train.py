import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fea_share_preprocess import preprocess
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from rdagent.scenarios.kaggle.experiment.meta_tpl.fea_share_preprocess import preprocess

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DIRNAME = Path(__file__).absolute().resolve().parent
#DIRNAME = Path("path/to/RD-Agent/rdagent/scenarios/kaggle/experiment/meta_tpl")

# support various method for metrics calculation
def compute_metrics_for_classification(y_true, y_pred):
    """Compute accuracy metric for classification."""
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def compute_metrics_for_classification(y_true, y_pred):
    """Compute MCC for classification."""
    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc


# Load and preprocess the data
data_df = pd.read_csv("path/to/playground-series-s4e8/train.csv")
data_df = data_df.drop(["id"], axis=1)

X = data_df.drop(["class"], axis=1)
y = data_df[["class"]]

label_encoder = LabelEncoder()
# 将 y 转换为 1D 数组
y = label_encoder.fit_transform(y)  # 将 y 转换为一维
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

# 1) Preprocess the data
X_train = preprocess(X_train)
X_valid = preprocess(X_valid)

submission_df = pd.read_csv("path/to/playground-series-s4e8/test.csv")
passenger_ids = submission_df["id"]
submission_df = submission_df.drop(["id"], axis=1)
X_test = preprocess(submission_df)
print("Step 1 done")

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

print("Step 2 done")


# 3) Train the model
model_l = []  # list[tuple[model, predict_func,]]
for f in DIRNAME.glob("model*.py"):
    # TODO put select() in model.py: fit(X_train, y_train, X_valid, y_valid)
    print(f"Training model from file: {f.name}")

    m = __import__(f.name.strip(".py"))
    model_l.append((m.fit(X_train, y_train, X_valid, y_valid), m.predict, f.name))

print("model append done")

# Evaluate the model on the validation set
y_valid_pred_l = []
"""for model, predict_func in model_l:
    y_valid_pred_l.append(predict_func(model, X_valid))"""

for i, (model, predict_func, model_name) in enumerate(model_l):
    print(f"Evaluating model from file: {model_name}")

    y_valid_pred = predict_func(model, X_valid)
    y_valid_pred_l.append(y_valid_pred)
    mcc = compute_metrics_for_classification(y_valid, y_valid_pred)
    mcc_result = pd.DataFrame({
        "Model": [model_name],  
        "Metric": ["MCC"], 
        "Value": [mcc]
    })
    # Save individual model predictions
    mcc_result.to_csv(f"path/to/playground-series-s4e8/submission_score_{model_name.strip('.py')}.csv", index=False)
    print(f"{model_name} final on validation set: {mcc}")

# Make predictions on the test set and save each model's predictions
y_test_pred_l = []

for i, (model, predict_func, model_name) in enumerate(model_l):
    # Print the current model file name being processed
    print(f"Making predictions with model from file: {model_name}")
    
    y_test_pred = predict_func(model, X_test)
    y_test_pred_l.append(y_test_pred)
    
    # Convert probability predictions to classification labels (0 or 1)
    y_test_pred_labels = (y_test_pred > 0.5).astype(int)
    
    # Inverse transform to convert 0 and 1 back to original classes 'e' or 'p'
    y_test_pred_original = label_encoder.inverse_transform(y_test_pred_labels)
    
    # Save each model's test set predictions
    test_result = pd.DataFrame({
        "id": passenger_ids, 
        "class": y_test_pred_original  # Use original labels
    })
    # Use model name as part of the filename
    test_result.to_csv(f"path/to/playground-series-s4e8/test_predictions_{model_name.strip('.py')}.csv", index=False)
    print(f"Test predictions for {model_name} saved.")


# Ensemble
# TODO: ensemble method in a script