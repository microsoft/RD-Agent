# %%
from dotenv import load_dotenv

from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorImplementationExperimentLoaderFromPDFfiles,
)
from rdagent.scenarios.qlib.factor_task_implementation import (
    COSTEERFG_QUANT_FACTOR_IMPLEMENTATION,
)
from rdagent.log import rdagent_logger as logger

assert load_dotenv()


def extract_factors_and_implement(report_file_path: str) -> None:
    with logger.tag('factor'):
        with logger.tag('load'):
            factor_tasks = FactorImplementationExperimentLoaderFromPDFfiles().load(report_file_path)
        with logger.tag('implementation'):
            implementation_result = COSTEERFG_QUANT_FACTOR_IMPLEMENTATION().generate(factor_tasks)
            logger.log_object(implementation_result, tag="results")
    # Qlib to run the implementation
    return implementation_result


if __name__ == "__main__":
    extract_factors_and_implement("/home/xuyang1/workspace/report.pdf")

# %%
