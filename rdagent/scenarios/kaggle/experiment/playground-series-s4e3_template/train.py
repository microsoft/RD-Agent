import importlib.util
import random
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
from fea_share_preprocess import preprocess_script

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DIRNAME = Path(__file__).absolute().resolve().parent

def compute_metrics_for_classification(y_true, y_pred):
    """Compute average ROC AUC for multi-label classification."""
    auc_scores = []
    for i in range(y_true.shape[1]):
        # Convert y_pred to numpy array if it's a list
        if isinstance(y_pred, list):
            y_pred_i = np.array(y_pred[i])[:, 1]  # Get probabilities for positive class
        else:
            y_pred_i = y_pred[:, i]
        auc_scores.append(roc_auc_score(y_true.iloc[:, i], y_pred_i))
    return np.mean(auc_scores)

def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_positive_proba(y_pred):
    """Extract positive class probabilities from prediction."""
    if isinstance(y_pred, list):
        return np.column_stack([proba[:, 1] for proba in y_pred])
    return y_pred

# 1) Preprocess the data
X_train, X_valid, y_train, y_valid, X_test, ids = preprocess_script()

# Print X_train head and shape
print("X_train shape:", X_train.shape)
print("\nX_train head:")
print(X_train.head())

# Print y_train head and shape
print("\ny_train shape:", y_train.shape)
print("\ny_train head:")
print(y_train.head())

# Replace the existing train-validation split with StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# 3) Train the model
model_l = []
for train_index, val_index in skf.split(X_train, y_train.iloc[:, 0]):  # Using the first target for stratification
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
    
    # Debug print
    print("y_valid_pred type:", type(y_valid_pred))
    if isinstance(y_valid_pred, list):
        print("y_valid_pred length:", len(y_valid_pred))
        print("First element shape:", y_valid_pred[0].shape)
    else:
        print("y_valid_pred shape:", y_valid_pred.shape)
    
    y_valid_pred_positive = get_positive_proba(y_valid_pred)
    metrics = compute_metrics_for_classification(y_valid, y_valid_pred_positive)
    print("Average ROC AUC on validation set: ", metrics)
    metrics_all.append(metrics)

# 5) Save the validation ROC AUC
max_index = np.argmax(metrics_all)
pd.Series(data=[metrics_all[max_index]], index=["ROC AUC"]).to_csv("submission_score.csv")

# 6) Make predictions on the test set and save them
X_test_selected = model_l[max_index][2].select(X_test.copy())
y_test_pred = model_l[max_index][1](model_l[max_index][0], X_test_selected)
y_test_pred_positive = get_positive_proba(y_test_pred)

# 7) Submit predictions for the test set
submission_result = pd.DataFrame({"id": ids})
target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
for i, col in enumerate(target_columns):
    submission_result[col] = y_test_pred_positive[:, i]

submission_result.to_csv("submission.csv", index=False)
