import time
from sklearn.model_selection import train_test_split
from load_data import load_data
from feature import feat_eng
from model01 import model_workflow


def log_execution_results(start_time, val_pred, test_pred, hypers, execution_label):
    """Log the results of a single model execution."""
    feedback_str = f"{execution_label} successful.\n"
    feedback_str += f"Validation predictions shape: {val_pred.shape if val_pred is not None else 'None'}\n"
    feedback_str += f"Test predictions shape: {test_pred.shape if test_pred is not None else 'None'}\n"
    feedback_str += f"Hyperparameters: {hypers if hypers is not None else 'None'}\n"
    feedback_str += f"Execution time: {time.time() - start_time:.2f} seconds.\n"
    print(feedback_str)


# Load and preprocess data
X, y, test_X, test_ids = load_data()
X, y, test_X = feat_eng(X, y, test_X)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.8, random_state=42)

# First execution
print("The first execution begins.\n")
start_time = time.time()
val_pred, test_pred, hypers = model_workflow(
    X=train_X,
    y=train_y,
    val_X=val_X,
    val_y=val_y,
    test_X=None,
)
log_execution_results(start_time, val_pred, test_pred, hypers, "The first execution")

# Second execution
print("The second execution begins.\n")
start_time = time.time()
val_pred, test_pred, final_hypers = model_workflow(
    X=train_X,
    y=train_y,
    val_X=None,
    val_y=None,
    test_X=test_X,
    hyper_params=hypers,
)
log_execution_results(start_time, val_pred, test_pred, final_hypers, "The second execution")

print("Model code test passed successfully.")
