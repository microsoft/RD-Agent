from typing import Optional, Dict
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import logging

def model_workflow(X, y, val_X: Optional[pd.DataFrame] = None, val_y: Optional[pd.Series] = None, test_X: Optional[pd.DataFrame] = None, hyper_params: Dict = None):
    """
    Train and evaluate a machine learning model for the Forest Cover Type Prediction competition.
    This function trains a model using the provided training data and hyperparameters. It can also evaluate the model on validation data if provided and generate predictions for test data.
    Parameters:
        X (pd.DataFrame): Training feature data with shape (n_samples, n_features).
        y (pd.Series): Training label data with shape (n_samples,).
        val_X (Optional[pd.DataFrame]): Validation feature data with shape (n_val_samples, n_features). Default is None.
        val_y (Optional[pd.Series]): Validation label data with shape (n_val_samples,). Default is None.
        test_X (Optional[pd.DataFrame]): Test feature data with shape (n_test_samples, n_features). Default is None.
        hyper_params (dict): Dictionary of hyperparameters for model configuration. Default is None.
    Returns:
        pred_val (Optional[pd.Series]): Predictions on validation data with shape (n_val_samples,). Returned if validation data is provided.
        pred_test (Optional[pd.Series]): Predictions on test data with shape (n_test_samples,). Returned if test data is provided.
        hyper_params (dict): Updated dictionary of hyperparameters after training.
    Notes:
        - Ensure input arrays (`X`, `y`, `val_X`, `val_y`, `test_X`) have consistent dimensions and shapes.
        - Use default values for hyperparameters if `hyper_params` is not provided.
        - Train the model on `X` and `y`.
        - Evaluate the model using `val_X` and `val_y` if validation data is available.
        - If `test_X` is provided, generate predictions for it.
        - Avoid using progress bars (e.g., `tqdm`) in the implementation.
        - Utilize GPU support for training if necessary to accelerate the process.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set default hyperparameters if not provided
    if hyper_params is None:
        hyper_params = {'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'min_child_weight': 1, 'reg_alpha': 0, 'reg_lambda': 1}

    # Encode labels to ensure they are in the range [0, num_classes-1]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    if val_y is not None:
        val_y_encoded = label_encoder.transform(val_y)

    # Initialize the XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=hyper_params['n_estimators'],
        learning_rate=hyper_params['learning_rate'],
        max_depth=hyper_params['max_depth'],
        subsample=hyper_params['subsample'],
        colsample_bytree=hyper_params['colsample_bytree'],
        gamma=hyper_params['gamma'],
        min_child_weight=hyper_params['min_child_weight'],
        reg_alpha=hyper_params['reg_alpha'],
        reg_lambda=hyper_params['reg_lambda'],
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='gpu_hist'  # Use GPU support for faster training
    )

    # Train the model with early stopping
    eval_set = [(X, y_encoded)]
    if val_X is not None and val_y is not None:
        eval_set.append((val_X, val_y_encoded))

    logger.info('Starting model training...')
    model.fit(X, y_encoded, eval_set=eval_set, early_stopping_rounds=50, verbose=False)
    logger.info('Model training completed.')

    pred_val = None
    pred_test = None

    # Evaluate on validation data if provided
    if val_X is not None and val_y is not None:
        pred_val = model.predict(val_X)
        accuracy = accuracy_score(val_y_encoded, pred_val)
        logger.info(f'Validation Accuracy: {accuracy:.4f}')

    # Generate predictions for test data if provided
    if test_X is not None:
        pred_test = model.predict(test_X)

    return pred_val, pred_test, hyper_params