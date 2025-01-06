"""
A qualified ensemble implementation should:
- Successfully run
- Return predictions
- Have correct shapes for inputs and outputs
- Use validation data appropriately
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from load_data import load_data
from feature import feat_eng
from ensemble import ens_and_decision

X, y, test_X, test_ids = load_data()
X, y, test_X = feat_eng(X, y, test_X)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

test_preds_dict = {}
val_preds_dict = {}
{% for mn in model_names %}
from {{mn}} import model_workflow as {{mn}}_workflow
test_preds_dict["{{mn}}"], val_preds_dict["{{mn}}"], _ = {{mn}}_workflow(
    X=train_X,
    y=train_y,
    val_X=val_X,
    val_y=val_y,
    test_X=test_X
)
{% endfor %}

# Run ensemble
try:
    final_pred = ens_and_decision(test_preds_dict, val_preds_dict, val_y)

    # Check shape
    assert final_pred.shape == val_y.shape, "Wrong output shape"

    # check if scores.csv is generated
    if not Path("scores.csv").exists():
        raise Exception("scores.csv is not generated")
    
    print("Ensemble test passed successfully.")
    print(f"Output shape: {final_pred.shape}")
    print(f"Unique values in predictions: {np.unique(final_pred)}")

except Exception as e:
    print(f"Test failed: {str(e)}")
