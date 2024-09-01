from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from rdagent.scenarios.kaggle.experiment.meta_tpl.fea_share_preprocess import preprocess


def compute_metrics_for_classification(y_true, y_pred):
    """Compute MCC for classification."""
    from sklearn.metrics import matthews_corrcoef

    return matthews_corrcoef(y_true, y_pred)


def perform_kfold_cross_validation(X, y, n_splits=2, random_seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_metrics = []

    DIRNAME = Path(__file__).absolute().resolve().parent

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]

        # TODO: Preprocess and Feature Engineering before K-Fold CV

        # Preprocess the data
        X_train_fold = preprocess(X_train_fold)
        X_valid_fold = preprocess(X_valid_fold)

        # Feature Engineering
        X_train_l_fold, X_valid_l_fold = [], []
        for f in DIRNAME.glob("feat*.py"):
            m = __import__(f.name.strip(".py"))
            X_train_fold = m.feat_eng(X_train_fold)
            X_valid_fold = m.feat_eng(X_valid_fold)

            X_train_l_fold.append(X_train_fold)
            X_valid_l_fold.append(X_valid_fold)

        X_train_fold = pd.concat(X_train_l_fold, axis=1)
        X_valid_fold = pd.concat(X_valid_l_fold, axis=1)

        # Align features
        X_valid_fold = X_valid_fold.reindex(columns=X_train_fold.columns, fill_value=0)

        # Train and evaluate models
        mcc_scores = []
        model_l = []  # Reinitialize model list
        for f in DIRNAME.glob("model*.py"):
            m = __import__(f.name.strip(".py"))
            model = m.fit(X_train_fold, y_train_fold, X_valid_fold, y_valid_fold)
            y_valid_pred = m.predict(model, X_valid_fold)
            mcc = compute_metrics_for_classification(y_valid_fold, y_valid_pred)
            mcc_scores.append(mcc)
            print(f"Fold {fold+1}, Model {f.name}: MCC = {mcc}")

        # Store the average MCC score for this fold
        avg_mcc = np.mean(mcc_scores)
        fold_metrics.append(avg_mcc)
        print(f"Fold {fold+1} average MCC: {avg_mcc}")

    # Calculate the overall average MCC
    overall_avg_mcc = np.mean(fold_metrics)
    result_df = pd.DataFrame({"Overall Average MCC": [overall_avg_mcc]})
    result_df.to_csv(f"path/to/playground-series-s4e8/cv_score_{f.name.strip('.py')}.csv", index=False)

    print(f"Overall Average MCC across all folds: {overall_avg_mcc}")
    return overall_avg_mcc


# This allows the script to be run directly
if __name__ == "__main__":
    # Load and preprocess the data
    data_df = pd.read_csv("path/to/playground-series-s4e8/train.csv")
    data_df = data_df.drop(["id"], axis=1)

    X = data_df.drop(["class"], axis=1)
    y = data_df[["class"]]

    label_encoder = LabelEncoder()
    # transfrom y to 1D
    y = label_encoder.fit_transform(y)
    result = perform_kfold_cross_validation(X, y)
