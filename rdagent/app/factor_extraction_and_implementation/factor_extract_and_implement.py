# %%
from pathlib import Path

from dotenv import load_dotenv
from rdagent.document_process.document_analysis import (
    check_factor_viability,
    classify_report_from_dict,
    deduplicate_factors_by_llm,
    extract_factors_from_report_dict,
    merge_file_to_factor_dict_to_factor_dict,
)
from rdagent.document_process.document_reader import load_and_process_pdfs_by_langchain


def extract_factors_and_implement(report_file_path: str) -> None:
    assert load_dotenv()
    docs_dict = load_and_process_pdfs_by_langchain(Path(report_file_path))

    selected_report_dict = classify_report_from_dict(report_dict=docs_dict, vote_time=1)
    file_to_factor_result = extract_factors_from_report_dict(docs_dict, selected_report_dict)
    factor_dict = merge_file_to_factor_dict_to_factor_dict(file_to_factor_result)

    factor_viability = check_factor_viability(factor_dict)

    factor_dict, duplication_names_list = deduplicate_factors_by_llm(factor_dict, factor_viability)


if __name__ == "__main__":
    extract_factors_and_implement("/home/xuyang1/workspace/report.pdf")
