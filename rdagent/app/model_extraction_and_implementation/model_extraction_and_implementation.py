# %%
from dotenv import load_dotenv
from rdagent.model_implementation.one_shot import ModelTaskGen
from rdagent.model_implementation.task_extraction import ModelImplementationTaskLoaderFromPDFfiles

assert load_dotenv()


def extract_models_and_implement(report_file_path: str) -> None:
    factor_tasks = ModelImplementationTaskLoaderFromPDFfiles().load(report_file_path)
    implementation_result = ModelTaskGen().generate(factor_tasks)
    return implementation_result


if __name__ == "__main__":
    extract_models_and_implement("../test_doc")