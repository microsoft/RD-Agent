"""
Helper functions for testing the ensemble coder(CoSTEER-based) component.
"""

from rdagent.components.coder.data_science.ensemble import EnsembleCoSTEER
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.scenarios.data_science.scen import DataScienceScen


def develop_ensemble():
    scen = DataScienceScen(competition="aerial-cactus-identification")
    ensemble_coder = EnsembleCoSTEER(scen)

    # Create the task
    task = EnsembleTask(
        name="EnsembleTask",
        description="Implement ensemble and decision making for model predictions"
    )

    # Develop the experiment
    exp = ensemble_coder.develop(task)
    return exp


if __name__ == "__main__":
    develop_ensemble() 
