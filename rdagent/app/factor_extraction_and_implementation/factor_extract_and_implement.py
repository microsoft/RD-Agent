# %%
from dotenv import load_dotenv
from rdagent.factor_implementation.CoSTEER import CoSTEERFG
from rdagent.factor_implementation.task_loader.pdf_loader import FactorImplementationTaskLoaderFromPDFfiles

assert load_dotenv()


def extract_factors_and_implement(report_file_path: str) -> None:
    factor_tasks = FactorImplementationTaskLoaderFromPDFfiles().load(report_file_path)
    implementation_result = CoSTEERFG().generate(factor_tasks)
    return implementation_result


if __name__ == "__main__":
    extract_factors_and_implement("/home/xuyang1/workspace/report.pdf")
