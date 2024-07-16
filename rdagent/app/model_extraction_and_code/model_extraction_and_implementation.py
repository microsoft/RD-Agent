# %%
from dotenv import load_dotenv

from rdagent.components.coder.model_coder.task_loader import (
    ModelExperimentLoaderFromPDFfiles,
)
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
from rdagent.scenarios.qlib.model_coder import QlibModelCoSTEER


def extract_models_and_implement(report_file_path: str = "../test_doc") -> None:
    scenario = QlibModelScenario()
    exp = ModelExperimentLoaderFromPDFfiles().load(report_file_path)
    exp = QlibModelCoSTEER(scenario).develop(exp)
    return exp


import fire

if __name__ == "__main__":
    fire.Fire(extract_models_and_implement)
