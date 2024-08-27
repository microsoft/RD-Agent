import random
import os

import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from rdagent.scenarios.kaggle.experiment.meta_tpl.fea_share_preprocess import preprocess
from fea_share_preprocess import preprocess


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

# Load and preprocess the data
# 1) Overall preprocess: appear only once in a competition
data_df = pd.read_csv("/home/v-xisenwang/git_ignore_folder/data/playground-series-s4e8/train.csv")
data_df = data_df.drop(["id"], axis=1)
 
X = data_df.drop(["class"], axis=1)
y = data_df["class"].to_numpy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # 将类别标签转换为数值
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

# Preprocess the data
X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
y_train = preprocess(y_train)
y_valid = preprocess(y_valid)

# 2) auto preprocess
X_train_l, X_valid_l = [], []
y_train_l, y_valid_l = [], []
for f in DIRNAME.glob("feat*.py"):
    m = __import__(f.name.strip(".py"))
    X_train = m.feat_eng(X_train)
    X_valid = m.feat_eng(X_valid)
    y_train = m.feat_eng(y_train)
    y_valid = m.feat_eng(y_valid)

    X_train_l.append(X_train)
    X_valid_l.append(X_valid)
    y_train_l.append(y_train)
    y_valid_l.append(y_valid)

X_train = pd.concat(X_train_l, axis=1)
X_valid = pd.concat(X_valid_l, axis=1)
y_train = pd.concat(y_train_l, axis=1)
y_valid = pd.concat(y_valid_l, axis=1)

# Train the model
model_l = []  # list[tuple[model, predict_func,]]
for f in DIRNAME.glob("model*.py"):
    # TODO put select() in model.py: fit(X_train, y_train, X_valid, y_valid)
    m = __import__(f.name.strip(".py"))
    model_l.append((m.fit(X_train, y_train, X_valid, y_valid), m.predict))

# Evaluate the model on the validation set
y_valid_pred_l = []
for model, predict_func in model_l:
    y_valid_pred_l.append(predict_func(model, X_valid))

# y_valid_pred = predict_func(model, X_valid)

# Ensemble
# TODO: ensemble method in a script
# Average the predictions and apply a threshold to determine class labels
y_valid_pred = np.mean(y_valid_pred_l, axis=0)
y_valid_pred = (y_valid_pred > 0.5).astype(int)

mcc = compute_metrics_for_classification(y_valid, y_valid_pred)
print("Final on validation set: ", mcc)

# Save the validation accuracy
pd.Series(data=[mcc], index=["MCC"]).to_csv("/home/v-xisenwang/RD-Agent/rdagent/scenarios/kaggle/experiment/meta_tpl/submission_score.csv")

# Load and preprocess the test set
submission_df = pd.read_csv("/home/v-xisenwang/git_ignore_folder/data/playground-series-s4e8/test.csv")
passenger_ids = submission_df["id"]
submission_df = submission_df.drop(["id"], axis=1)
X_test = preprocessor.transform(submission_df)


# Make predictions on the test set and save them
y_test_pred_bool = predict(model, X_test)
y_test_pred_int = y_test_pred_bool.astype(int)  # 转换布尔值为 0 或 1
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred_int)  # 将整数转换回 'e' 或 'p'
submission_result = pd.DataFrame({"id": ids, "class": y_test_pred_labels})

# submit predictions for the test set
submission_result.to_csv("./submission.csv", index=False)
