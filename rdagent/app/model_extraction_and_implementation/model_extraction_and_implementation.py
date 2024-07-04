# %%
from dotenv import load_dotenv

from rdagent.components.coder.model_coder.one_shot import ModelCodeWriter
from rdagent.components.coder.model_coder.task_loader import (
    ModelExperimentLoaderFromPDFfiles,
)


def extract_models_and_implement(report_file_path: str = "../test_doc") -> None:
    factor_tasks = ModelExperimentLoaderFromPDFfiles().load(report_file_path)
    implementation_result = ModelCodeWriter().generate(factor_tasks)
    return implementation_result


import fire

if __name__ == "__main__":
    fire.Fire(extract_models_and_implement)
