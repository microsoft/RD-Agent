"""
Generate dataset to test the workflow output
"""

from pathlib import Path

from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.data_science.workflow import WorkflowCoSTEER
from rdagent.components.coder.data_science.workflow.es import (
    WorkflowMultiProcessEvolvingStrategy,
)
from rdagent.components.coder.data_science.workflow.eval import (
    WorkflowGeneralCaseSpecEvaluator,
)
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.experiment import FBWorkspace
from rdagent.scenarios.data_science.experiment.experiment import WorkflowExperiment
from rdagent.scenarios.data_science.scen import DataScienceScen

def develop_one_competition(competition: str):
    scen = DataScienceScen(competition=competition)
    workflow_coder = WorkflowCoSTEER(scen)

    wt = WorkflowTask(
        name="WorkflowTask",
        description="Integrate the existing processes of load_data, feature, model, and ensemble into a complete workflow.",
        spec="",
        base_code={
            
        }
    )

    tpl_ex_path = Path(__file__).resolve() / Path("rdagent/scenarios/kaggle/tpl_ex").resolve() / competition
    injected_file_names = ["spec/workflow.md", "load_data.py", "feat01.py", "model01.py", "ens.py", "main.py"]

    workflowexp = FBWorkspace()
    for file_name in injected_file_names:
        file_path = tpl_ex_path / file_name
        workflowexp.inject_code(**{file_name: file_path.read_text()})

    wt.spec += workflowexp.code_dict["spec/model.md"]
    wt.base_code += workflowexp.code_dict["model01.py"]
    exp = WorkflowExperiment(
        sub_tasks=[wt],
    )

    es = WorkflowMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)
    new_code = es.implement_one_task(target_task=wt, queried_knowledge=None)
    print(new_code)


if __name__ == "__main__":
    develop_one_competition("aerial-cactus-identification")
    # dotenv run -- python rdagent/components/coder/data_science/workflow/test.py
