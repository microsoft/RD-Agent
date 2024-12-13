"""
Helper functions for testing the feature coder(CoSTEER-based) component.
- Does the developer loop work correctly

It is NOT:
- it is not interface unittest(i.e. workspace evaluator in the CoSTEER Loop)
"""

import pickle

from rdagent.components.coder.data_science.feature import FeatureCoSTEER
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.scenarios.data_science.experiment.experiment import FeatureExperiment
from rdagent.scenarios.data_science.scen import DataScienceScen

# from rdagent.components.coder.data_science.feature.es import ModelMultiProcessEvolvingStrategy


def develop_one_competition(competition: str):  # -> experiment
    scen = DataScienceScen(competition=competition)
    feature_coder = FeatureCoSTEER(scen)

    with open(
        "/home/v-yuanteli/RD-Agent/rdagent/scenarios/kaggle/tpl_ex/aerial-cactus-identification/spec/feature.md", "r"
    ) as file:
        feat_spec = file.read()

    # Create the experiment
    ft = FeatureTask(name="FeatureTask", description=scen.competition_descriptions, spec=feat_spec)
    exp = FeatureExperiment(
        sub_tasks=[ft],
    )

    with open(
        "/home/v-yuanteli/RD-Agent/rdagent/scenarios/kaggle/tpl_ex/aerial-cactus-identification/load_data.py", "r"
    ) as file:
        load_data_code = file.read()
    exp.experiment_workspace.inject_code(**{"load_data.py": load_data_code})

    # Develop the experiment
    exp = feature_coder.develop(exp)


if __name__ == "__main__":
    develop_one_competition("aerial-cactus-identification")