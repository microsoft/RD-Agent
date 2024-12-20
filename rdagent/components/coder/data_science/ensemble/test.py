"""
Helper functions for testing the ensemble coder(CoSTEER-based) component.
"""
import sys
from pathlib import Path

from rdagent.components.coder.data_science.ensemble import EnsembleCoSTEER
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment

# Add the competition folder to path
COMPETITION_PATH = Path(__file__).parent.parent.parent.parent.parent / "scenarios" / "kaggle" / "tpl_ex" / "aerial-cactus-identification"
sys.path.append(str(COMPETITION_PATH))

EnsembleExperiment = DSExperiment 

def load_ensemble_spec():
    spec_path = COMPETITION_PATH / "spec" / "ensemble.md"
    with open(spec_path, 'r') as f:
        return f.read()


def develop_ensemble():
    # Initialize scenario and coder
    scen = DataScienceScen(competition="aerial-cactus-identification")
    ensemble_coder = EnsembleCoSTEER(scen)
    # Load ensemble specification
    ensemble_spec = load_ensemble_spec()

    # Create the ensemble task with actual data context and specification
    task = EnsembleTask(
        name="EnsembleTask",
        description=
        """
        Implement ensemble and decision making for model predictions.
        """
    )

    exp = EnsembleExperiment(
        sub_tasks=[task]
    )

    # Injecting the corresponding specification
    exp.experiment_workspace.inject_code(**{"spec/ensemble.md": ensemble_spec})

    # Develop the experiment
    exp = ensemble_coder.develop(exp)
    return exp


if __name__ == "__main__":
    develop_ensemble() 
