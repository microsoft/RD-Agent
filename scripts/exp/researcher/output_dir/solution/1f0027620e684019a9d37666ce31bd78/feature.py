import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function Definition
def feat_eng(X, y, X_test):
    """
    Perform feature engineering on the Forest Cover Type Prediction dataset.
    This function applies necessary transformations to the training and test datasets to prepare them for model training and evaluation. It handles missing values, scales features, and applies any competition-specific feature engineering steps.
    Parameters:
        X (pd.DataFrame): Train data to be transformed with shape (n_samples, n_features).
        y (pd.Series): Train label data with shape (n_samples,).
        X_test (pd.DataFrame): Test data with shape (n_test_samples, n_features).
    Returns:
        X_transformed (pd.DataFrame): Transformed train data with shape (n_samples, -1).
        y_transformed (pd.Series): Transformed train label data with shape (n_samples,).
        X_test_transformed (pd.DataFrame): Transformed test data with shape (n_test_samples, -1).
    Notes:
        - Ensure the sample size of the train data and the test data remains consistent.
        - The input shape and output shape should generally be the same, though some columns may be added or removed.
        - Avoid data leakage by only using features derived from training data.
        - Handle missing values and outliers appropriately.
        - Ensure consistency between feature data types and transformations.
        - Apply competition-specific feature engineering steps as needed.
        - Utilize GPU support or multi-processing to accelerate the feature engineering process if necessary.
    """
    # Ensure consistent columns between train and test datasets
    common_cols = X.columns.intersection(X_test.columns)
    X = X[common_cols]
    X_test = X_test[common_cols]

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    # Detect and handle outliers by capping them at the 1st and 99th percentiles
    for col in X_imputed.select_dtypes(include=[np.number]).columns:
        lower_bound = X_imputed[col].quantile(0.01)
        upper_bound = X_imputed[col].quantile(0.99)
        X_imputed[col] = np.clip(X_imputed[col], lower_bound, upper_bound)
        X_test_imputed[col] = np.clip(X_test_imputed[col], lower_bound, upper_bound)

    # Create new features based on interaction between existing features
    interaction_features = ['Elevation', 'Aspect', 'Slope']
    for i in range(len(interaction_features)):
        for j in range(i+1, len(interaction_features)):
            feat1 = interaction_features[i]
            feat2 = interaction_features[j]
            X_imputed[f'{feat1}_{feat2}_Product'] = X_imputed[feat1] * X_imputed[feat2]
            X_imputed[f'{feat1}_{feat2}_Ratio'] = X_imputed[feat1] / (X_imputed[feat2] + 1e-5)
            X_imputed[f'{feat1}_{feat2}_Difference'] = X_imputed[feat1] - X_imputed[feat2]
            X_test_imputed[f'{feat1}_{feat2}_Product'] = X_test_imputed[feat1] * X_test_imputed[feat2]
            X_test_imputed[f'{feat1}_{feat2}_Ratio'] = X_test_imputed[feat1] / (X_test_imputed[feat2] + 1e-5)
            X_test_imputed[f'{feat1}_{feat2}_Difference'] = X_test_imputed[feat1] - X_test_imputed[feat2]

    # Create polynomial features up to the second degree for numerical columns
    poly = PolynomialFeatures(degree=2, include_bias=False)
    num_cols = X_imputed.select_dtypes(include=[np.number]).columns
    poly_features = poly.fit_transform(X_imputed[num_cols])
    poly_features_test = poly.transform(X_test_imputed[num_cols])
    poly_feature_names = poly.get_feature_names_out(num_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    poly_df_test = pd.DataFrame(poly_features_test, columns=poly_feature_names)

    # Drop original numerical columns and add polynomial features
    X_transformed = pd.concat([X_imputed.drop(columns=num_cols), poly_df], axis=1)
    X_test_transformed = pd.concat([X_test_imputed.drop(columns=num_cols), poly_df_test], axis=1)

    # Normalize continuous features using Min-Max scaling
    scaler = MinMaxScaler()
    cont_cols = X_transformed.select_dtypes(include=[np.number]).columns
    X_transformed[cont_cols] = scaler.fit_transform(X_transformed[cont_cols])
    X_test_transformed[cont_cols] = scaler.transform(X_test_transformed[cont_cols])

    # Encode categorical features using one-hot encoding
    cat_cols = [col for col in X_transformed.columns if 'Soil_Type' in col or 'Wilderness_Area' in col]
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    cat_features = one_hot_encoder.fit_transform(X_transformed[cat_cols])
    cat_features_test = one_hot_encoder.transform(X_test_transformed[cat_cols])
    cat_feature_names = one_hot_encoder.get_feature_names_out(cat_cols)
    cat_df = pd.DataFrame(cat_features, columns=cat_feature_names)
    cat_df_test = pd.DataFrame(cat_features_test, columns=cat_feature_names)

    # Drop original categorical columns and add one-hot encoded features
    X_transformed = pd.concat([X_transformed.drop(columns=cat_cols), cat_df], axis=1)
    X_test_transformed = pd.concat([X_test_transformed.drop(columns=cat_cols), cat_df_test], axis=1)

    return X_transformed, y, X_test_transformed