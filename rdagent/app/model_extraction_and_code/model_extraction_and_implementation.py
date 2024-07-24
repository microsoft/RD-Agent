# %%
from dotenv import load_dotenv

from rdagent.components.coder.model_coder.task_loader import (
    ModelExperimentLoaderFromPDFfiles,
)
from rdagent.scenarios.qlib.developer.model_coder import QlibModelCoSTEER
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario


def extract_models_and_implement(report_file_path: str = ".../test") -> None:
    scenario = QlibModelScenario()
    exp = ModelExperimentLoaderFromPDFfiles().load(report_file_path)
    exp = QlibModelCoSTEER(scenario).develop(exp)
    return exp


import fire

if __name__ == "__main__":
    fire.Fire(extract_models_and_implement)
