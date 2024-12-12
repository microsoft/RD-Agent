"""
Helper functions for testing the ensemble coder(CoSTEER-based) component.
"""

from rdagent.components.coder.data_science.ensemble import EnsembleCoSTEER
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.kaggle.tpl_ex.aerial_cactus_identification.load_data import load_from_raw_data
from rdagent.scenarios.kaggle.tpl_ex.aerial_cactus_identification.feat01 import feat_eng
from rdagent.scenarios.kaggle.tpl_ex.aerial_cactus_identification.model01 import model_workflow
from sklearn.model_selection import train_test_split


def develop_ensemble():
    # Initialize scenario and coder
    scen = DataScienceScen(competition="aerial-cactus-identification")
    ensemble_coder = EnsembleCoSTEER(scen)

    # Load competition data and process it
    train_images, train_labels, test_images, test_ids = load_from_raw_data()
    
    # Feature engineering
    train_images, train_labels, train_param = feat_eng(train_images, train_labels)
    test_images, _, _ = feat_eng(test_images, param=train_param)
    
    # Split for validation
    train_images, validation_images, train_labels, validation_labels = train_test_split(
        train_images, train_labels, test_size=0.1, random_state=42
    )
    
    # Get model predictions (simulating multiple models)
    val_pred, test_pred = model_workflow(
        train_images, train_labels, validation_images, validation_labels, test_images
    )

    # Create the ensemble task with actual data context
    task = EnsembleTask(
        name="EnsembleTask",
        description="""Implement ensemble and decision making for model predictions.
        Input shapes:
        - test_pred: {test_shape}
        - val_pred: {val_shape}
        - val_label: {label_shape}""".format(
            test_shape=test_pred.shape,
            val_shape=val_pred.shape,
            label_shape=validation_labels.shape
        )
    )

    # Develop the experiment
    exp = ensemble_coder.develop(task)
    return exp


if __name__ == "__main__":
    develop_ensemble() 
