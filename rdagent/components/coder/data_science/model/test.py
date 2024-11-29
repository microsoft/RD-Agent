"""
Generate dataset to test the model workflow output
"""
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.components.coder.data_science.model import ModelCoSTEER
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.scenarios.data_science.experiment.experiment import ModelExperiment
from pathlib import Path

# Take tasks, spec.md and feat as input, generate a feedback as output
def develop_one_competition(competition: str):
    scen = DataScienceScen(competition=competition)
    model_coder = ModelCoSTEER(scen)
    
    # Create the experiment
    mt = ModelTask(name="ModelTask", description="", base_code="import pandas...")
    exp = ModelExperiment(
        sub_tasks=[mt],
    )
    
    tpl_ex_path = Path(__file__).resolve() / Path("rdagent/scenarios/kaggle/tpl_ex").resolve() / competition
    injected_file_names = ["spec.md", "load_data.py", "feat01.py"]
    for file_name in injected_file_names:
        file_path = tpl_ex_path / file_name
        exp.experiment_workspace.inject_code(**{file_name: file_path.read_text()})
    
    # Run the experiment
    exp = model_coder.develop(exp)


if __name__ == "__main__":
    develop_one_competition("aerial-cactus-identification")
    # dotenv run -- python rdagent/components/coder/data_science/model/test.py