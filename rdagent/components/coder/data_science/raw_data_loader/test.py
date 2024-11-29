"""
Helper functions for testing the raw_data_loader coder(CoSTEER-based) component.
- Does the developer loop work correctly

It is NOT:
- it is not interface unittest(i.e. workspace evaluator in the CoSTEER Loop)
"""

from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.raw_data_loader import DataLoaderCoSTEER
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.data_science.experiment.experiment import DataLoaderExperiment

def develop_one_competition(competition: str): # -> experiment
    scen = DataScienceScen(competition=competition)
    data_loader_coder = DataLoaderCoSTEER(scen)

    # Create the experiment
    dlt = DataLoaderTask(name="DataLoaderTask", description="")
    exp = DataLoaderExperiment(
        sub_tasks=[dlt],
    )

    # Develop the experiment
    exp = data_loader_coder.develop(exp)


if __name__ == "__main__":
    develop_one_competition("aerial-cactus-identification")
