import pandas as pd

# Function Definition
def load_data():
    """
    Load and preprocess the Forest Cover Type Prediction dataset.
    This function reads the training and test datasets from the specified location, processes the data to ensure it is correctly formatted and ready for machine learning model training and evaluation. It handles missing values, converts data types, and prepares the feature matrices and target vectors.
    Data Source Location:
        - /kaggle/input/
    Returns:
        X (pd.DataFrame): Feature matrix for training data with shape (n_samples, n_features).
        y (pd.Series): Target vector for training data with shape (n_samples,).
        X_test (pd.DataFrame): Feature matrix for test data with shape (n_test_samples, n_features).
        test_ids (pd.Series): Identifiers for the test data with shape (n_test_samples,).
    Notes:
        - Ensure proper file encoding (UTF-8) and delimiters (CSV comma-separated).
        - Convert data types correctly (e.g., numeric for features, categorical for target).
        - Handle missing values appropriately (e.g., impute or drop rows/columns).
        - Optimize memory usage for large datasets using techniques like downcasting or reading data in chunks if necessary.
    """
    # Load the training data
    train_df = pd.read_csv('/kaggle/input/train.csv')
    
    # Load the test data
    test_df = pd.read_csv('/kaggle/input/test.csv')

    # Extract features and target from training data
    X = train_df.drop(columns=['Cover_Type'])
    y = train_df['Cover_Type']
    
    # Extract features and IDs from test data
    X_test = test_df.drop(columns=['Id'])
    test_ids = test_df['Id']

    return X, y, X_test, test_ids