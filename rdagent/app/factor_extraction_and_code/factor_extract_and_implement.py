# %%
from dotenv import load_dotenv

from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.scenarios.qlib.developer.factor_coder import QlibFactorCoSTEER
from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorExperimentLoaderFromPDFfiles,
)

assert load_dotenv()


def extract_factors_and_implement(report_file_path: str) -> None:
    scenario = QlibFactorScenario()
    exp = FactorExperimentLoaderFromPDFfiles().load(report_file_path)
    exp = QlibFactorCoSTEER(scenario).develop(exp)
    # Qlib to run the implementation in rd loop
    return exp


if __name__ == "__main__":
    extract_factors_and_implement("/home/xuyang1/workspace/report.pdf")
