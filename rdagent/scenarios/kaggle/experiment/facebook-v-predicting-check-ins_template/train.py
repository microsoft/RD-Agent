import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
from fea_share_preprocess import preprocess_script
from sklearn.metrics import average_precision_score, log_loss

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DIRNAME = Path(__file__).absolute().resolve().parent


def compute_map3(y_true, y_pred):
    """Compute Mean Average Precision @ 3 for multi-class classification."""
    return average_precision_score(y_true, y_pred, average="micro")


def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 1) Preprocess the data
X_train, X_valid, y_train, y_valid, X_test, place_id_encoder, test_row_ids, n_classes = preprocess_script()

# 2) Auto feature engineering
X_train_l, X_valid_l = [], []
X_test_l = []

for f in DIRNAME.glob("feature/feat*.py"):
    cls = import_module_from_path(f.stem, f).feature_engineering_cls()
    cls.fit(X_train)
    X_train_f = cls.transform(X_train)
    X_valid_f = cls.transform(X_valid)
    X_test_f = cls.transform(X_test)

    if X_train_f.shape[-1] == X_valid_f.shape[-1] and X_train_f.shape[-1] == X_test_f.shape[-1]:
        X_train_l.append(X_train_f)
        X_valid_l.append(X_valid_f)
        X_test_l.append(X_test_f)

X_train = pd.concat(X_train_l, axis=1)
X_valid = pd.concat(X_valid_l, axis=1)
X_test = pd.concat(X_test_l, axis=1)

# 3) Train the model
model_l = []  # list[tuple[model, predict_func, validation_score]]
for f in DIRNAME.glob("model/model*.py"):
    select_python_path = f.with_name(f.stem.replace("model", "select") + f.suffix)
    select_m = import_module_from_path(select_python_path.stem, select_python_path)
    X_train_selected = select_m.select(X_train.copy())
    X_valid_selected = select_m.select(X_valid.copy())

    m = import_module_from_path(f.stem, f)
    # Check if the fit function accepts n_classes
    if "n_classes" in m.fit.__code__.co_varnames:
        model = m.fit(X_train, y_train, X_valid, y_valid, n_classes)
    else:
        model = m.fit(X_train, y_train, X_valid, y_valid)

    # Evaluate the model on the validation set
    y_valid_pred = m.predict(model, X_valid)
    validation_score = log_loss(y_valid, y_valid_pred)

    model_l.append((model, m.predict, validation_score))

# Sort models by validation score (lower is better for log loss)
model_l.sort(key=lambda x: x[2])

# 4) Use the best model for predictions
best_model, best_predict_func, _ = model_l[0]

# 5) Make predictions on the validation set using the best model
y_valid_pred = best_predict_func(best_model, X_valid)

# Compute metrics
map3 = compute_map3(y_valid, y_valid_pred)
print(f"MAP@3 on validation set: {map3}")

# 6) Save the validation metrics
pd.Series(data=[map3], index=["MAP@3"]).to_csv("submission_score.csv")

# 7) Make predictions on the test set using the best model
y_test_pred = best_predict_func(best_model, X_test)

# Get top 3 predictions for each test sample
top_3_indices = np.argsort(-y_test_pred, axis=1)[:, :3]
top_3_place_ids = place_id_encoder.inverse_transform(top_3_indices)

# Create submission DataFrame
submission_result = pd.DataFrame(
    {"row_id": test_row_ids, "place_id": [" ".join(map(str, ids)) for ids in top_3_place_ids]}
)

submission_result.to_csv("submission.csv", index=False)
