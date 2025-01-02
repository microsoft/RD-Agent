"""
A qualified ensemble implementation should:
- Successfully run
- Return predictions
- Have correct shapes for inputs and outputs
- Use validation data appropriately
"""

import numpy as np
from ensemble import ens_and_decision
from pathlib import Path

# Create test data
n_models = 3
n_samples = 100

# Create synthetic predictions
test_pred_l = [np.random.rand(n_samples, 1) for _ in range(n_models)]
val_pred_l = [np.random.rand(n_samples, 1) for _ in range(n_models)]
val_label = np.random.randint(0, 2, (n_samples, 1))

# Run ensemble
try:
    final_predictions = ens_and_decision(test_pred_l, val_pred_l, val_label)

    # Check shape
    assert final_predictions.shape == (n_samples, 1), "Wrong output shape"

    # check if scores.csv is generated
    if not Path("scores.csv").exists():
        raise Exception("scores.csv is not generated")
    
    print("Ensemble test passed successfully.")
    print(f"Output shape: {final_predictions.shape}")
    print(f"Unique values in predictions: {np.unique(final_predictions)}")

except Exception as e:
    print(f"Test failed: {str(e)}")
