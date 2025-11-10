"""Example usage of Context7 MCP integration."""

import asyncio

from context7 import query_context7


async def main():
    """Main function for testing context7 functionality."""
    error_msg = """### TRACEBACK: Traceback (most recent call last):
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
value_or_values = func(trial)
^^^^^^^^^^^
File "/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/3ee7771734bb4b04af50f76e8c0e5ed7/main.py", line 226, in <lambda>
study.optimize(lambda trial: lgb_optuna_objective(trial, X_sub, y_sub, num_class, debug),
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/3ee7771734bb4b04af50f76e8c0e5ed7/main.py", line 201, in lgb_optuna_objective
gbm = lgb.train(
^^^^^^^^^^
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/lightgbm/engine.py", line 282, in train
booster = Booster(params=params, train_set=train_set)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/lightgbm/basic.py", line 3641, in __init__
_safe_call(
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/lightgbm/basic.py", line 296, in _safe_call
raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))
lightgbm.basic.LightGBMError: No OpenCL device found
### SUPPLEMENTARY_INFO: lgb.train called with device=gpu, gpu_platform_id=0, gpu_device_id=0."""
    full_code = """
    import os
import sys
import time
import numpy as np
import pandas as pd
import argparse

import lightgbm as lgb
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

def print_eda(train, test):
    eda = []
    eda.append("=== Start of EDA part ===")
    eda.append("# Initial Data Assessment & Sanitization")

    eda.append(f"Train shape: {train.shape}")
    eda.append(f"Test shape: {test.shape}")

    eda.append("\nFirst 5 rows of train:")
    eda.append(str(train.head().to_string()))

    eda.append("\nFirst 5 rows of test:")
    eda.append(str(test.head().to_string()))

    eda.append("\n# Data Types per column (train):")
    eda.append(str(train.dtypes.value_counts()))
    eda.append("\n# Data Types per column (test):")
    eda.append(str(test.dtypes.value_counts()))

    eda.append("\nMissing values per column (train):")
    eda.append(str(train.isnull().sum().sort_values(ascending=False).head(10)))
    eda.append("\nMissing values per column (test):")
    eda.append(str(test.isnull().sum().sort_values(ascending=False).head(10)))

    eda.append("\nUnique values per column (train):")
    uniques = train.nunique().sort_values(ascending=False)
    eda.append(str(uniques.head(15)))
    eda.append("\nUnique values per column (test):")
    uniques_test = test.nunique().sort_values(ascending=False)
    eda.append(str(uniques_test.head(15)))

    if "Cover_Type" in train.columns:
        eda.append("\nTarget variable distribution (Cover_Type):")
        eda.append(str(train['Cover_Type'].value_counts().sort_index()))

    num_cols = [col for col in train.columns if train[col].dtype in ['int32', 'int64', 'float32', 'float64']]
    if len(num_cols) > 20:
        num_cols = [c for c in num_cols if "Soil" not in c and "Wilder" not in c and c not in ["Id", "Cover_Type"]]
    eda.append('\n# Numerical column summary (central tendency, spread, potential outliers):')
    eda.append(str(train[num_cols].describe().transpose().head(10)))

    bin_cols = [col for col in train.columns if train[col].nunique(dropna=False) == 2 and col not in ["Cover_Type"]]
    eda.append("\n# Number of binary indicator features in train: {}".format(len(bin_cols)))
    eda.append("Sample binary indicator columns: {}".format(bin_cols[:10]))

    eda.append("=== End of EDA part ===")
    print('\n'.join(eda[:10000]))  # truncate if very long

def get_numeric_int32_cols(df, exclude=[]):
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            cols.append(col)
    return cols

def engineer_features(df):
    df = df.copy()
    pi = np.pi
    df['Aspect_rad'] = (df['Aspect'] * pi / 180).astype(np.float32)
    df['Aspect_sin'] = np.sin(df['Aspect_rad']).astype(np.float32)
    df['Aspect_cos'] = np.cos(df['Aspect_rad']).astype(np.float32)
    df['Elevation_minus_VertHydro'] = (df['Elevation'] - df['Vertical_Distance_To_Hydrology']).astype(np.float32)
    den = (df['Horizontal_Distance_To_Hydrology'] + 1).astype(np.float32)
    ratio = df['Elevation'] / den
    ratio = ratio.replace([np.inf, -np.inf], 0.0)
    ratio = ratio.fillna(0).astype(np.float32)
    df['Elevation_Hydro_Ratio'] = ratio
    df['Slope_Hillshade'] = (df['Slope'] * df['Hillshade_Noon']).astype(np.float32)
    return df

def stratified_subsample(X, y, frac, seed, n_class):
    n_total = len(y)
    n_sub = int(n_total * frac)
    value_counts = y.value_counts()
    min_per_class = value_counts.min()
    # Always guarantee at least 2 samples per class (for Sklearn stratify)
    if n_sub < n_class * 2:
        n_sub = max(n_class * 2, n_sub)
    if min_per_class < 2 or min_per_class*n_class < n_sub:
        print(f"[WARNING] Subsample size ({n_sub}) or class frequency is too small, using full data for Optuna in this run.")
        return X, y
    for c in range(n_class):
        if (y == c).sum() < 2:
            print(f"[WARNING] Class {c} has <2 samples, using full data for Optuna.")
            return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n_sub, stratify=y, random_state=seed, shuffle=True,
    )
    return X_sub, y_sub

def compute_class_scaling(val_pred_proba, val_labels):
    from scipy.optimize import minimize
    n_class = val_pred_proba.shape[1]
    y_true = val_labels
    def custom_acc(scaling):
        arr = val_pred_proba * scaling.reshape((1, -1))
        pred = np.argmax(arr, axis=1)
        return -accuracy_score(y_true, pred)
    scaling0 = np.ones(n_class, dtype=np.float32)
    bounds = [(0.8, 1.2) for _ in range(n_class)]
    result = minimize(custom_acc, scaling0, method='L-BFGS-B', bounds=bounds)
    best_scaling = result.x
    acc_scaled = -result.fun
    acc_base = accuracy_score(y_true, np.argmax(val_pred_proba, axis=1))
    if acc_scaled - acc_base >= 0.001:
        return best_scaling
    return None

def apply_scaling(pred_proba, scaling_factors):
    if scaling_factors is None:
        return pred_proba
    return pred_proba * scaling_factors.reshape((1, -1))

def build_lgb_dataset(X, y=None, categorical_feature='auto', free_raw_data=True):
    if y is not None:
        return lgb.Dataset(
            data=X,
            label=y,
            free_raw_data=free_raw_data,
            categorical_feature=categorical_feature
        )
    else:
        return lgb.Dataset(
            data=X,
            free_raw_data=free_raw_data,
            categorical_feature=categorical_feature
        )

def get_dtype_map_train(train, exclude=[]):
    dt = {}
    for col in train.columns:
        if col in exclude:
            continue
        if pd.api.types.is_integer_dtype(train[col]) or pd.api.types.is_float_dtype(train[col]):
            dt[col] = np.int32
    return dt

def get_dtype_map_test(test, exclude=[]):
    return get_dtype_map_train(test, exclude=exclude)

def free_memory(objs):
    import gc
    for x in objs:
        del x
    gc.collect()

def lgb_optuna_objective(trial, X_sub, y_sub, num_class, debug):
    param = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'random_state': 42,
    }
    param['learning_rate'] = trial.suggest_float('learning_rate', 0.03, 0.21, log=True)
    max_depth = trial.suggest_int('max_depth', 5, 14)
    param['max_depth'] = max_depth
    param['num_leaves'] = trial.suggest_int("num_leaves", 2 ** (max_depth - 1), 2 ** max_depth - 1)
    param['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 120)
    param['min_child_weight'] = trial.suggest_float('min_child_weight', 1e-3, 100.0, log=True)
    param['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True)
    param['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True)
    param['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    param['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    param['subsample_freq'] = 1
    param['n_jobs'] = 4
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []
    # Setup LightGBM callbacks for log-eval and early-stopping
    if debug:
        num_boost_round = 10
        early_stopping_rounds = 5
    else:
        num_boost_round = 200
        early_stopping_rounds = 100
    callbacks = [lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(50)]
    for train_idx, val_idx in skf.split(X_sub, y_sub):
        X_train, X_val = X_sub.iloc[train_idx, :], X_sub.iloc[val_idx, :]
        y_train, y_val = y_sub.iloc[train_idx], y_sub.iloc[val_idx]
        lgb_train = build_lgb_dataset(X_train, y_train)
        lgb_val = build_lgb_dataset(X_val, y_val)
        gbm = lgb.train(
            param,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            num_boost_round=num_boost_round,
            callbacks=callbacks
        )
        val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        val_pred_labels = np.argmax(val_pred, axis=1)
        acc = accuracy_score(y_val, val_pred_labels)
        val_scores.append(acc)
    mean_acc = np.mean(val_scores)
    trial.set_user_attr('mean_accuracy', mean_acc)
    print(f"[Optuna trial] Params: {param} --> mean valid accuracy: {mean_acc:.5f}")
    return mean_acc

def run_optuna_search(X_sub, y_sub, num_class, debug, timeout_min=60, n_trials=100):
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if debug:
        timeout = 120
        n_trials = 10
    else:
        timeout = int(timeout_min * 60)
    study.optimize(lambda trial: lgb_optuna_objective(trial, X_sub, y_sub, num_class, debug),
                   timeout=timeout, n_trials=n_trials, show_progress_bar=False)
    trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)
    unique_param_sets = []
    seen_strs = set()
    for t in trials:
        param_str = str({k: t.params[k] for k in sorted(t.params.keys())})
        if param_str not in seen_strs:
            unique_param_sets.append((t.value, t.params))
            seen_strs.add(param_str)
        if len(unique_param_sets) >= 3:
            break
    print('Top 3 hyperparameter sets:')
    for i, (score, params) in enumerate(unique_param_sets, 1):
        print(f"  Rank {i}: Accuracy={score:.5f} | Params={params}")
    return [p for _, p in unique_param_sets]

def train_full_and_select(train_features, train_labels, num_class, best_param_list, debug):
    results = []
    for i, params in enumerate(best_param_list):
        X, y = train_features, train_labels
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        lgb_train = build_lgb_dataset(X_tr, y_tr)
        lgb_val = build_lgb_dataset(X_val, y_val)
        full_params = dict(params)
        full_params.update({
            'objective': 'multiclass',
            'num_class': num_class,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'random_state': 42,
        })
        if debug:
            num_boost_round = 10
            early_stopping_rounds = 5
        else:
            num_boost_round = 1000
            early_stopping_rounds = 200
        callbacks = [lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(50)]
        print(f"[Full Training] Model {i+1}, start, params: {full_params}")
        bst = lgb.train(
            full_params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            num_boost_round=num_boost_round,
            callbacks=callbacks
        )
        val_pred_proba = bst.predict(X_val, num_iteration=bst.best_iteration)
        val_pred_labels = np.argmax(val_pred_proba, axis=1)
        acc = accuracy_score(y_val, val_pred_labels)
        scaling_factors = compute_class_scaling(val_pred_proba, y_val)
        if scaling_factors is not None:
            scaled_proba = apply_scaling(val_pred_proba, scaling_factors)
            acc_scaled = accuracy_score(y_val, np.argmax(scaled_proba, axis=1))
            print(f"[Full {i+1}] val_acc_raw={acc:.5f} val_acc_scaled={acc_scaled:.5f}")
        else:
            acc_scaled = None
            print(f"[Full {i+1}] val_acc_raw={acc:.5f} no scaling improvement")
        results.append({
            "model": bst,
            "val_acc_raw": acc,
            "val_acc_scaled": acc_scaled,
            "scaling_factors": scaling_factors,
            "val_pred_proba": val_pred_proba,
            "val_labels": y_val,
            "params": full_params,
        })
    best_idx = 0
    best_acc = -1
    for i, row in enumerate(results):
        acc_this = row["val_acc_scaled"] if row["val_acc_scaled"] is not None else row["val_acc_raw"]
        if acc_this > best_acc:
            best_acc = acc_this
            best_idx = i
    print("Selecting Model", best_idx+1, "as final_model")
    final_row = results[best_idx]
    return final_row, results

def main():
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    DEBUG = args.debug

    dpath = './workspace_input/'
    print("[INFO] Loading data...")

    train_sample = pd.read_csv(os.path.join(dpath, 'train.csv'), nrows=500)
    test_sample = pd.read_csv(os.path.join(dpath, 'test.csv'), nrows=500)
    numeric_cols_train = get_numeric_int32_cols(train_sample, exclude=[])
    dtype_map_train = {c: np.int32 for c in numeric_cols_train if c not in ["Id", "Cover_Type"] + []}
    numeric_cols_test = get_numeric_int32_cols(test_sample, exclude=[])
    dtype_map_test = {c: np.int32 for c in numeric_cols_test if c not in ["Id"] + []}

    try:
        train = pd.read_csv(
            os.path.join(dpath, 'train.csv'),
            dtype=dtype_map_train,
            low_memory=False
        )
        test = pd.read_csv(
            os.path.join(dpath, 'test.csv'),
            dtype=dtype_map_test,
            low_memory=False
        )
    except Exception as e:
        print("Error loading files:", e)
        sys.exit(1)

    print_eda(train, test)

    print("[INFO] Engineering features ...")
    train = engineer_features(train)
    test = engineer_features(test)

    y = train['Cover_Type'] - 1
    X = train.drop(columns=['Cover_Type'])
    X_test = test.copy()

    n_class = 7
    if DEBUG:
        optuna_frac = 0.1
    else:
        optuna_frac = 0.20
    print(f"[INFO] Creating stratified Optuna sample: fraction={optuna_frac}")
    X_sub, y_sub = stratified_subsample(X, y, frac=optuna_frac, seed=42, n_class=n_class)
    print(f"[INFO] Optuna subsample size: {len(X_sub)}")

    print("[INFO] Running Optuna search for LightGBM ...")
    optuna_best_params_list = run_optuna_search(X_sub, y_sub, n_class, DEBUG,
                                               timeout_min=2 if DEBUG else 60,
                                               n_trials=10 if DEBUG else 100)

    print("[INFO] Full training ...")
    start_time = time.time()
    best_row, all_model_results = train_full_and_select(X, y, n_class, optuna_best_params_list, DEBUG)
    end_time = time.time()
    debug_time = end_time - start_time

    if DEBUG:
        if len(X_sub) == len(X):
            scale = (1000 / 10)
        else:
            scale = (1/optuna_frac) * (1000 / 10)
        est = scale * debug_time
        print("=== Start of Debug Information ===")
        print(f"debug_time: {debug_time:.2f}")
        print(f"estimated_time: {est:.2f}")
        print("=== End of Debug Information ===")

    final_model = best_row["model"]
    scaling_factors = best_row["scaling_factors"]

    print("[INFO] Inference and Submission ...")
    test_pred_proba = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    if scaling_factors is not None:
        test_pred_proba = apply_scaling(test_pred_proba, scaling_factors)
    test_pred_label = np.argmax(test_pred_proba, axis=1) + 1
    submission = pd.DataFrame({'Id': X_test['Id'], 'Cover_Type': test_pred_label.astype(np.int32)})
    submission = submission[['Id', 'Cover_Type']]
    submission.to_csv('submission.csv', index=False)
    print("[INFO] Saved submission.csv")

    ind = ["model_1", "ensemble"]
    accs = []
    val_acc = best_row["val_acc_scaled"] if best_row["val_acc_scaled"] is not None else best_row["val_acc_raw"]
    accs.append(val_acc)
    accs.append(val_acc)
    df_scores = pd.DataFrame({'Accuracy': accs}, index=ind)
    df_scores.index.name = "Model"
    df_scores.to_csv("scores.csv")
    print("[INFO] Saved scores.csv")

    print("[COMPLETE]")

if __name__ == '__main__':
    main()
    """
    # Normal usage (verbose=False by default)
    result = await query_context7(error_message=error_msg, full_code=full_code, verbose=True)
    print("Result:", result)

    # Debug usage with verbose output
    # result = await query_context7(error_msg, verbose=True)
    # print("Debug Result:", result)


if __name__ == "__main__":
    asyncio.run(main())
