# %%
import json
from pathlib import Path

from dotenv import load_dotenv

from document_process.document_analysis import (
    check_factor_dict_viability,
    deduplicate_factors_several_times,
    extract_factors_from_report_dict_and_classify_result,
)
from document_process.document_reader import classify_report_from_dict, load_and_process_pdfs_by_langchain
from oai.llm_utils import APIBackend


def extract_factors_and_implement(report_file_path: str):
    assert load_dotenv()
    api = APIBackend()
    docs_dict_select = load_and_process_pdfs_by_langchain(Path(report_file_path))

    selected_report_dict = classify_report_from_dict(report_dict=docs_dict_select, api=api, vote_time=1)
    file_to_factor_result = extract_factors_from_report_dict_and_classify_result(docs_dict_select, selected_report_dict)

    factor_dict = {}
    for file_name in file_to_factor_result:
        for factor_name in file_to_factor_result[file_name]:
            factor_dict.setdefault(factor_name, [])
            factor_dict[factor_name].append(file_to_factor_result[file_name][factor_name])

    factor_dict_simple_deduplication = {}
    for factor_name in factor_dict:
        if len(factor_dict[factor_name]) > 1:
            factor_dict_simple_deduplication[factor_name] = max(
                factor_dict[factor_name], key=lambda x: len(x["formulation"]),
            )
        else:
            factor_dict_simple_deduplication[factor_name] = factor_dict[factor_name][0]
    # %%

    factor_viability = check_factor_dict_viability(factor_dict_simple_deduplication)
    # json.dump(
    #     factor_viability,
    #     open(
    #         "factor_viability_all_reports.json",
    #         "w",
    #     ),
    #     indent=4,
    # )

    # factor_viability = json.load(
    #     open(
    #         "factor_viability_all_reports.json"
    #     )
    # )

    # %%

    duplication_names_list = deduplicate_factors_several_times(factor_dict_simple_deduplication)
    duplication_names_list = sorted(duplication_names_list, key=lambda x: len(x), reverse=True)
    json.dump(duplication_names_list, open("duplication_names_list.json", "w"), indent=4)

    # %%
    factor_dict_viable = {
        factor_name: factor_dict_simple_deduplication[factor_name]
        for factor_name in factor_dict_simple_deduplication
        if factor_viability[factor_name]["viability"]
    }

    to_replace_dict = {}
    for duplication_names in duplication_names_list:
        for duplication_factor_name in duplication_names[1:]:
            to_replace_dict[duplication_factor_name] = duplication_names[0]

    added_lower_name_set = set()
    factor_dict_deduplication_with_llm = dict()
    for factor_name in factor_dict_simple_deduplication:
        if factor_name not in to_replace_dict and factor_name.lower() not in added_lower_name_set:
            added_lower_name_set.add(factor_name.lower())
            factor_dict_deduplication_with_llm[factor_name] = factor_dict_simple_deduplication[factor_name]

    to_replace_viable_dict = {}
    for duplication_names in duplication_names_list:
        viability_list = [factor_viability[name]["viability"] for name in duplication_names]
        if True not in viability_list:
            continue
        target_factor_name = duplication_names[viability_list.index(True)]
        for duplication_factor_name in duplication_names:
            if duplication_factor_name == target_factor_name:
                continue
            to_replace_viable_dict[duplication_factor_name] = target_factor_name

    added_lower_name_set = set()
    factor_dict_deduplication_with_llm_and_viable = dict()
    for factor_name in factor_dict_viable:
        if factor_name not in to_replace_viable_dict and factor_name.lower() not in added_lower_name_set:
            added_lower_name_set.add(factor_name.lower())
            factor_dict_deduplication_with_llm_and_viable[factor_name] = factor_dict_simple_deduplication[factor_name]

    # %%

    dump_md_list = [
        [factor_dict_simple_deduplication, "final_factor_book"],
        [factor_dict_viable, "final_viable_factor_book"],
        [factor_dict_deduplication_with_llm, "final_deduplicated_factor_book"],
        [factor_dict_deduplication_with_llm_and_viable, "final_deduplicated_viable_factor_book"],
    ]

    for dump_md in dump_md_list:
        factor_name_set = set()
        current_index = 1
        target_dict = dump_md[0]
        json.dump(target_dict, open(f"{dump_md[1]}.json", "w"), indent=4)
        with open(
            rf"{dump_md[1]}.md",
            "w",
        ) as fw:
            for factor_name in target_dict:
                formulation = target_dict[factor_name]["formulation"]
                if factor_name in formulation:
                    target_factor_name = factor_name.replace("_", r"\_")
                    formulation = formulation.replace(factor_name, target_factor_name)
                for variable in target_dict[factor_name]["variables"]:
                    if variable in formulation:
                        target_variable = variable.replace("_", r"\_")
                        formulation = formulation.replace(variable, target_variable)

                fw.write(f"## {current_index}. 因子名称：{factor_name}\n")
                fw.write(f"### Viability: {target_dict[factor_name]['viability']}\n")
                fw.write(f"### Viability Reason: {target_dict[factor_name]['viability_reason']}\n")
                fw.write(f"### description: {target_dict[factor_name]['description']}\n")
                fw.write(f"### formulation: $$ {formulation} $$\n")
                fw.write(f"### formulation string: {formulation}\n")
                # write a table of variable and its description

                fw.write("### variable tables: \n")
                fw.write("| variable | description |\n")
                fw.write("| -------- | ----------- |\n")
                for variable in target_dict[factor_name]["variables"]:
                    fw.write(f"| {variable} | {target_dict[factor_name]['variables'][variable]} |\n")

                current_index += 1
