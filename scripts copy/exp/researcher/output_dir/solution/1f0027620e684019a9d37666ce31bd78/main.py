import pandas as pd
from sklearn.model_selection import train_test_split
# Importing the necessary modules
from load_data import load_data
from feature import feat_eng
from ensemble import ens_and_decision
# List of model files to be used in the ensemble
model_names = ['model_xgboost_classifier']
# Main workflow function
def main():
    """
    Main workflow for the Forest Cover Type Prediction competition.
    This function integrates data loading, feature engineering, model training, and ensemble decision-making into a cohesive workflow. It saves the final predictions in the required submission format.
    """
    # Load data
    X, y, X_test, test_ids = load_data()
    # Feature engineering
    X_transformed, y_transformed, X_test_transformed = feat_eng(X, y, X_test)
    # Split the dataset into training and validation sets (80% train, 20% validation)
    train_X, val_X, train_y, val_y = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)
    # Initialize dictionaries to store predictions
    val_preds_dict = {}
    test_preds_dict = {}
    # Train models and collect predictions
    for mn in model_names:
        model_module = __import__(mn)
        val_preds_dict[mn], test_preds_dict[mn], _ = model_module.model_workflow(
            X=train_X,
            y=train_y,
            val_X=val_X,
            val_y=val_y,
            test_X=X_test_transformed
        )
    # Perform ensemble and make final decision
    final_pred = ens_and_decision(test_preds_dict, val_preds_dict, val_y)
    # Create submission file
    submission = pd.DataFrame({'Id': test_ids, 'Cover_Type': final_pred})
    submission.to_csv('submission.csv', index=False)
    print(f'Submission file shape: {submission.shape}')
    print(submission.head())
if __name__ == '__main__':
    main()