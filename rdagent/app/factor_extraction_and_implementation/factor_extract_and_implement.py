# %%
from dotenv import load_dotenv

from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorExperimentLoaderFromPDFfiles,
)
from rdagent.scenarios.qlib.factor_task_implementation import (
    COSTEERFG_QUANT_FACTOR_IMPLEMENTATION,
)

assert load_dotenv()


def extract_factors_and_implement(report_file_path: str) -> None:
    factor_tasks = FactorExperimentLoaderFromPDFfiles().load(report_file_path)
    implementation_result = COSTEERFG_QUANT_FACTOR_IMPLEMENTATION().generate(factor_tasks)
    # Qlib to run the implementation
    return implementation_result


if __name__ == "__main__":
    extract_factors_and_implement("/home/xuyang1/workspace/report.pdf")
