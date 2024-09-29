import importlib.util
import random
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
from fea_share_preprocess import preprocess_script
from sklearn.metrics import matthews_corrcoef

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DIRNAME = Path(__file__).absolute().resolve().parent


# support various method for metrics calculation
def compute_metrics_for_classification(y_true, y_pred):
    """Compute MCC for classification."""
    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc


def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 1) Preprocess the data
X_train, X_valid, y_train, y_valid, X_test, ids = preprocess_script()

# Replace the existing train-validation split with StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# 3) Train the model
model_l = []
for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    for f in DIRNAME.glob("model/model*.py"):
        select_python_path = f.with_name(f.stem.replace("model", "select") + f.suffix)
        select_m = import_module_from_path(select_python_path.stem, select_python_path)
        X_train_selected = select_m.select(X_train_fold.copy())
        X_val_selected = select_m.select(X_val_fold.copy())

        m = import_module_from_path(f.stem, f)
        model_l.append((m.fit(X_train_selected, y_train_fold, X_val_selected, y_val_fold), m.predict, select_m))

# 4) Evaluate the model on the validation set
metrics_all = []
for model, predict_func, select_m in model_l:
    X_valid_selected = select_m.select(X_valid.copy())
    y_valid_pred = predict_func(model, X_valid_selected)
    y_valid_pred = np.argmax(y_valid_pred, axis=1)
    metrics = accuracy_score(y_valid, y_valid_pred)
    print("Accuracy on validation set: ", metrics)
    metrics_all.append(metrics)

# 5) Save the validation accuracy
max_index = np.argmax(metrics_all)
pd.Series(data=[metrics_all[max_index]], index=["Accuracy"]).to_csv("submission_score.csv")

# 6) Make predictions on the test set and save them
X_test_selected = model_l[max_index][2].select(X_test.copy())
y_test_pred = model_l[max_index][1](model_l[max_index][0], X_test_selected)
y_test_pred = np.argmax(y_test_pred, axis=1)

# Convert numeric predictions back to original labels
label_encoder = LabelEncoder()
label_encoder.fit(["Graduate", "Dropout", "Enrolled"])
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

# 7) Submit predictions for the test set
submission_result = pd.DataFrame({"id": ids, "Target": y_test_pred_labels})
submission_result.to_csv("submission.csv", index=False)
