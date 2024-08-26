import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_classification(y_true, y_pred):
    """Compute accuracy metric for classification."""
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent

# Load and preprocess the data
# 1) Overall preprocess: appear only once in a competition
data_df = pd.read_csv("/root/.data/train.csv")
data_df = data_df.drop(["PassengerId", "Name"], axis=1)

X = data_df.drop(["Transported"], axis=1)
y = data_df.Transported.to_numpy()

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

# 2) auto preprocess
X_train_l, X_valid_l = [], []
for f in DIRNAME.glob("fea*.py"):
    m = __import__(f.name.strip(".py"))
    X_train, X_valid = m.reprocesss(X)
    X_train_l.append(X_train)
    X_valid_l.append(X_valid)
X_train = pd.concat(X_train_l, axis=1)
X_valid = pd.concat(X_valid_l, axis=1)

# TODO: the processing y;

# Train the model
model_l = []  # list[tuple[model, predict_func,]]
for f in DIRNAME.glob("model*.py"):
    m = __import__(f.name.strip(".py"))
    model_l.append((m.fit(X_train, y_train, X_valid, y_valid), m.predict))

# Evaluate the model on the validation set
y_valid_pred_l = []
for model, predict_func in model_l:
    y_valid_pred_l.append(predict_func(model, X_valid))

# y_valid_pred = predict_func(model, X_valid)

# Ensemble
# TODO: ensemble method in a script
y_valid_pred = np.sum(y_valid_pred_l) / len(y_valid_pred_l)

accuracy = compute_metrics_for_classification(y_valid, y_valid_pred)
print("Final Accuracy on validation set: ", accuracy)

# Save the validation accuracy
pd.Series(data=[accuracy], index=["ACC"]).to_csv("./submission_score.csv")

# Load and preprocess the test set
submission_df = pd.read_csv("/root/.data/test.csv")
passenger_ids = submission_df["PassengerId"]
submission_df = submission_df.drop(["PassengerId", "Name"], axis=1)
X_test = preprocessor.transform(submission_df)

# Make predictions on the test set and save them
y_test_pred = predict(model, X_test)
submission_result = pd.DataFrame({"PassengerId": passenger_ids, "Transported": y_test_pred})
# submit predictions for the test set
submission_result.to_csv("./submission.csv", index=False)
