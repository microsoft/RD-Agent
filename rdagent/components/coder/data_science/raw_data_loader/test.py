"""
Helper functions for testing the raw_data_loader coder(CoSTEER-based) component.
- Does the developer loop work correctly

It is NOT:
- it is not interface unittest(i.e. workspace evaluator in the CoSTEER Loop)
"""

from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask, DataLoaderExperiment

def build_dummpy_exp(): # -> experiment
    dlt = DataLoaderTask(name="DataLoaderTask", description="")
    exp = DataLoaderExperiment(
        sub_tasks=[dlt],
    )


def get_developer():
    ...
