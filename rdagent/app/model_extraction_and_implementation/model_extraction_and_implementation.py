# %%
from dotenv import load_dotenv
from rdagent.components.task_implementation.model_implementation.one_shot import ModelTaskGen
from rdagent.components.task_implementation.model_implementation.task_extraction import ModelImplementationTaskLoaderFromPDFfiles



def extract_models_and_implement(report_file_path: str="../test_doc") -> None:
    factor_tasks = ModelImplementationTaskLoaderFromPDFfiles().load(report_file_path)
    implementation_result = ModelTaskGen().generate(factor_tasks)
    return implementation_result


import fire
if __name__ == "__main__":
    fire.Fire(extract_models_and_implement)
