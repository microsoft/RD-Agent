"""
Generate dataset to test the model workflow output
"""

from pathlib import Path

from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.data_science.model import ModelCoSTEER
from rdagent.components.coder.data_science.model.eval import (
    ModelGeneralCaseSpecEvaluator,
)
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.core.experiment import FBWorkspace
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.scen import KaggleScen


# Take tasks, spec.md and feat as input, generate a feedback as output
def develop_one_competition(competition: str):
    scen = KaggleScen(competition=competition)
    model_coder = ModelCoSTEER(scen)

    # Create the task
    mt = ModelTask(
        name="ModelTask",
        description="A CNN Model",
        model_type="CNN",
        architecture="\hat{y}_u = CNN(X_u)",
        # variables="variables: {'\\hat{y}_u': 'The predicted output for node u', 'X_u': 'The input features for node u'}",
        hyperparameters="...",
        base_code="",
    )

    tpl_ex_path = Path(__file__).resolve() / Path("rdagent/scenarios/kaggle/tpl_ex").resolve() / competition
    injected_file_names = ["spec/model.md", "load_data.py", "feature.py", "model01.py"]

    modelexp = FBWorkspace()
    for file_name in injected_file_names:
        file_path = tpl_ex_path / file_name
        modelexp.inject_files(**{file_name: file_path.read_text()})

    mt.base_code += modelexp.file_dict["model01.py"]
    exp = DSExperiment(
        sub_tasks=[mt],
    )

    # Test the evaluator:
    """eva = ModelGeneralCaseSpecEvaluator(scen=scen)
    exp.feedback = eva.evaluate(target_task=mt, queried_knowledge=None, implementation=modelexp, gt_implementation=None)
    print(exp.feedback)"""

    # Test the evolving strategy:
    """es = ModelMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)
    new_code = es.implement_one_task(target_task=mt, queried_knowledge=None, workspace=modelexp)
    print(new_code)"""

    # Run the experiment
    for file_name in injected_file_names:
        file_path = tpl_ex_path / file_name
        exp.experiment_workspace.inject_files(**{file_name: file_path.read_text()})

    exp = model_coder.develop(exp)


if __name__ == "__main__":
    develop_one_competition("aerial-cactus-identification")
    # dotenv run -- python rdagent/components/coder/data_science/model/test.py
