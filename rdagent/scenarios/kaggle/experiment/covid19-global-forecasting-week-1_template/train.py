import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
from fea_share_preprocess import preprocess_script
from sklearn.metrics import mean_squared_log_error

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DIRNAME = Path(__file__).absolute().resolve().parent

def compute_rmsle(y_true, y_pred):
    """Compute Root Mean Squared Logarithmic Error for regression."""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 1) Preprocess the data
X_train, X_valid, y_train, y_valid, X_test, forecast_ids = preprocess_script()

# 2) Train the model
model_l = []  # list[tuple[model, predict_func,]]
for f in DIRNAME.glob("model/model*.py"):
    m = import_module_from_path(f.stem, f)
    model_l.append((m.fit(X_train, y_train, X_valid, y_valid), m.predict))

print("X_valid", X_valid.head())

# 3) Evaluate the model on the validation set
metrics_all = []
for model, predict_func in model_l:
    y_valid_pred = predict_func(model, X_valid)
    
    # Debug: Check for negative values in actual and predicted ConfirmedCases
    print("Minimum value in y_valid['ConfirmedCases']:", y_valid['ConfirmedCases'].min())
    print("Minimum value in y_valid_pred['ConfirmedCases']:", y_valid_pred['ConfirmedCases'].min())
    
    # Debug: Print rows where actual or predicted values are negative
    negative_actual = y_valid[y_valid['ConfirmedCases'] < 0]
    negative_pred = y_valid_pred[y_valid_pred['ConfirmedCases'] < 0]
    
    if not negative_actual.empty:
        print("Rows with negative actual ConfirmedCases:")
        print(negative_actual)
    
    if not negative_pred.empty:
        print("Rows with negative predicted ConfirmedCases:")
        print(negative_pred)
    
    # Add a small positive value to avoid negative or zero values
    epsilon = 1e-8
    y_valid_cases = np.maximum(y_valid['ConfirmedCases'], epsilon)
    y_pred_cases = np.maximum(y_valid_pred['ConfirmedCases'], epsilon)
    
    rmsle_cases = compute_rmsle(y_valid_cases, y_pred_cases)
    rmsle_fatalities = compute_rmsle(np.maximum(y_valid['Fatalities'], epsilon), np.maximum(y_valid_pred['Fatalities'], epsilon))
    rmsle_avg = (rmsle_cases + rmsle_fatalities) / 2
    print(f"Average RMSLE on valid set: {rmsle_avg}")
    metrics_all.append(rmsle_avg)

# 4) Save the validation accuracy
min_index = np.argmin(metrics_all)
pd.Series(data=[metrics_all[min_index]], index=["RMSLE"]).to_csv("submission_score.csv")

# 5) Make predictions on the test set and save them
y_test_pred = model_l[min_index][1](model_l[min_index][0], X_test)

# 6) Submit predictions for the test set
submission_result = pd.DataFrame({
    "ForecastId": forecast_ids,
    "ConfirmedCases": y_test_pred['ConfirmedCases'],
    "Fatalities": y_test_pred['Fatalities']
})
submission_result.to_csv("submission.csv", index=False)
