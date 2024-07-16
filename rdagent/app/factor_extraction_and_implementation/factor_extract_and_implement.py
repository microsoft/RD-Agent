# %%
from dotenv import load_dotenv

from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import FactorExperimentLoaderFromPDFfiles
from rdagent.scenarios.qlib.factor_task_implementation import QlibFactorCoSTEER
from rdagent.log import rdagent_logger as logger

assert load_dotenv()


def extract_factors_and_implement(report_file_path: str) -> None:
    with logger.tag('extract_factors_and_implement'):
        with logger.tag('load_factor_tasks'):
            factor_tasks = FactorExperimentLoaderFromPDFfiles().load(report_file_path)
        with logger.tag('implement_factors'):
            implementation_result = QlibFactorCoSTEER().generate(factor_tasks)
    # Qlib to run the implementation
    return implementation_result


if __name__ == "__main__":
    extract_factors_and_implement("/home/xuyang1/workspace/report.pdf")

# %%
