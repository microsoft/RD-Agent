"""
Generate dataset to test the workflow output
"""

from pathlib import Path

from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.data_science.workflow import WorkflowCoSTEER
from rdagent.components.coder.data_science.workflow.eval import (
    WorkflowGeneralCaseSpecEvaluator,
)
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.experiment import FBWorkspace
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.scen import KaggleScen


def develop_one_competition(competition: str):
    scen = KaggleScen(competition=competition)
    workflow_coder = WorkflowCoSTEER(scen)

    wt = WorkflowTask(
        name="WorkflowTask",
        description="Integrate the existing processes of load_data, feature, model, and ensemble into a complete workflow.",
        base_code="",
    )

    tpl_ex_path = Path(__file__).resolve() / Path("rdagent/scenarios/kaggle/tpl_ex").resolve() / competition
    injected_file_names = ["spec/workflow.md", "load_data.py", "feature.py", "model01.py", "ensemble.py", "main.py"]

    workflowexp = FBWorkspace()
    for file_name in injected_file_names:
        file_path = tpl_ex_path / file_name
        workflowexp.inject_files(**{file_name: file_path.read_text()})

    wt.base_code += workflowexp.file_dict["main.py"]
    exp = DSExperiment(
        sub_tasks=[wt],
    )

    """es = WorkflowMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)
    new_code = es.implement_one_task(target_task=wt, queried_knowledge=None, workspace = workflowexp)
    print(new_code)"""

    """eva = WorkflowGeneralCaseSpecEvaluator(scen=scen)
    exp.feedback = eva.evaluate(target_task=wt, queried_knowledge=None, implementation=workflowexp, gt_implementation=None)
    print(exp.feedback)"""

    # Run the experiment
    for file_name in injected_file_names:
        file_path = tpl_ex_path / file_name
        exp.experiment_workspace.inject_files(**{file_name: file_path.read_text()})

    exp = workflow_coder.develop(exp)


if __name__ == "__main__":
    develop_one_competition("aerial-cactus-identification")
    # dotenv run -- python rdagent/components/coder/data_science/workflow/test.py
