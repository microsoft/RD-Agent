# %%
from dotenv import load_dotenv

from rdagent.components.task_implementation.model_implementation.one_shot import (
    ModelCodeWriter,
)
from rdagent.components.task_implementation.model_implementation.task_loader import (
    ModelImplementationExperimentLoaderFromPDFfiles,
)


def extract_models_and_implement(report_file_path: str = "../test_doc") -> None:
    factor_tasks = ModelImplementationExperimentLoaderFromPDFfiles().load(report_file_path)
    implementation_result = ModelCodeWriter().generate(factor_tasks)
    return implementation_result


import fire

if __name__ == "__main__":
    fire.Fire(extract_models_and_implement)
