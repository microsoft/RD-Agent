from __future__ import annotations

import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from jinja2 import Environment, StrictUndefined
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from rdagent.components.document_reader.document_reader import (
    load_and_process_pdfs_by_langchain,
)
from rdagent.components.loader.experiment_loader import FactorExperimentLoader
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.prompts import Prompts
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.factor_experiment_loader.json_loader import (
    FactorExperimentLoaderFromDict,
)

document_process_prompts = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


def classify_report_from_dict(
    report_dict: Mapping[str, str],
    vote_time: int = 1,
    substrings: tuple[str] = (),
) -> dict[str, dict[str, str]]:
    """
    Parameters:
    - report_dict (Dict[str, str]):
      A dictionary where the key is the path of the report (ending with .pdf),
      and the value is either the report content as a string.
    - input_max_token (int): Specifying the maximum number of input tokens.
    - vote_time (int): An integer specifying how many times to vote.
    - substrings (list(str)): List of hardcode substrings.

    Returns:
    - Dict[str, Dict[str, str]]: A dictionary where each key is the path of the report,
      with a single key 'class' and its value being the classification result (0 or 1).

    """
    # if len(substrings) == 0:
    #     substrings = (
    #         "金融工程",
    #         "金工",
    #         "回测",
    #         "因子",
    #         "机器学习",
    #         "深度学习",
    #         "量化",
    #     )

    res_dict = {}
    classify_prompt = document_process_prompts["classify_system"]

    for key, value in tqdm(report_dict.items()):
        if not key.endswith(".pdf"):
            continue
        file_name = key

        if isinstance(value, str):
            content = value
        else:
            logger.warning(f"Input format does not meet the requirements: {file_name}")
            res_dict[file_name] = {"class": 0}
            continue

        # pre-filter document with key words is not necessary, skip this check for now
        # if (
        #     not any(substring in content for substring in substrings) and False
        # ):
        #     res_dict[file_name] = {"class": 0}
        # else:
        while (
            APIBackend().build_messages_and_calculate_token(
                user_prompt=content,
                system_prompt=classify_prompt,
            )
            > LLM_SETTINGS.chat_token_limit
        ):
            content = content[: -(LLM_SETTINGS.chat_token_limit // 100)]

        vote_list = []
        for _ in range(vote_time):
            user_prompt = content
            system_prompt = classify_prompt
            res = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
            try:
                res = json.loads(res)
                vote_list.append(int(res["class"]))
            except json.JSONDecodeError:
                logger.warning(f"Return value could not be parsed: {file_name}")
                res_dict[file_name] = {"class": 0}
            count_0 = vote_list.count(0)
            count_1 = vote_list.count(1)
            if max(count_0, count_1) > int(vote_time / 2):
                break

        result = 1 if count_1 > count_0 else 0
        res_dict[file_name] = {"class": result}

    return res_dict


def __extract_factors_name_and_desc_from_content(
    content: str,
) -> dict[str, dict[str, str]]:
    session = APIBackend().build_chat_session(
        session_system_prompt=document_process_prompts["extract_factors_system"],
    )

    extracted_factor_dict = {}
    current_user_prompt = content

    for _ in range(10):
        extract_result_resp = session.build_chat_completion(
            user_prompt=current_user_prompt,
            json_mode=True,
        )
        ret_dict = json.loads(extract_result_resp)
        factors = ret_dict["factors"]
        if len(factors) == 0:
            break
        for factor_name, factor_description in factors.items():
            extracted_factor_dict[factor_name] = factor_description
        current_user_prompt = document_process_prompts["extract_factors_follow_user"]

    return extracted_factor_dict


def __extract_factors_formulation_from_content(
    content: str,
    factor_dict: dict[str, str],
) -> dict[str, dict[str, str]]:
    factor_dict_df = pd.DataFrame(
        factor_dict.items(),
        columns=["factor_name", "factor_description"],
    )

    system_prompt = document_process_prompts["extract_factor_formulation_system"]
    current_user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(
            document_process_prompts["extract_factor_formulation_user"],
        )
        .render(report_content=content, factor_dict=factor_dict_df.to_string())
    )

    session = APIBackend().build_chat_session(session_system_prompt=system_prompt)
    factor_to_formulation = {}

    for _ in range(10):
        extract_result_resp = session.build_chat_completion(
            user_prompt=current_user_prompt,
            json_mode=True,
        )
        ret_dict = json.loads(extract_result_resp)
        for name, formulation_and_description in ret_dict.items():
            if name in factor_dict:
                factor_to_formulation[name] = formulation_and_description
        if len(factor_to_formulation) != len(factor_dict):
            remain_df = factor_dict_df[~factor_dict_df["factor_name"].isin(factor_to_formulation)]
            current_user_prompt = (
                "Some factors are missing. Please check the following"
                " factors and their descriptions and continue extraction.\n"
                "==========================Remaining factors"
                "==========================\n" + remain_df.to_string()
            )
        else:
            break

    return factor_to_formulation


def __extract_factor_and_formulation_from_one_report(
    content: str,
) -> dict[str, dict[str, str]]:
    final_factor_dict_to_one_report = {}
    factor_dict = __extract_factors_name_and_desc_from_content(content)
    if len(factor_dict) != 0:
        factor_to_formulation = __extract_factors_formulation_from_content(
            content,
            factor_dict,
        )
    for factor_name in factor_dict:
        if (
            factor_name not in factor_to_formulation
            or "formulation" not in factor_to_formulation[factor_name]
            or "variables" not in factor_to_formulation[factor_name]
        ):
            continue

        final_factor_dict_to_one_report.setdefault(factor_name, {})
        final_factor_dict_to_one_report[factor_name]["description"] = factor_dict[factor_name]

        # use code to correct _ in formulation
        formulation = factor_to_formulation[factor_name]["formulation"]
        if factor_name in formulation:
            target_factor_name = factor_name.replace("_", r"\_")
            formulation = formulation.replace(factor_name, target_factor_name)
        for variable in factor_to_formulation[factor_name]["variables"]:
            if variable in formulation:
                target_variable = variable.replace("_", r"\_")
                formulation = formulation.replace(variable, target_variable)

        final_factor_dict_to_one_report[factor_name]["formulation"] = formulation
        final_factor_dict_to_one_report[factor_name]["variables"] = factor_to_formulation[factor_name]["variables"]

    return final_factor_dict_to_one_report


def extract_factors_from_report_dict(
    report_dict: dict[str, str],
    useful_no_dict: dict[str, dict[str, str]],
    n_proc: int = 11,
) -> dict[str, dict[str, dict[str, str]]]:
    useful_report_dict = {}
    for key, value in useful_no_dict.items():
        if isinstance(value, dict):
            if int(value.get("class")) == 1:
                useful_report_dict[key] = report_dict[key]
        else:
            logger.warning(f"Invalid input format: {key}")

    file_name_list = list(useful_report_dict.keys())

    final_report_factor_dict = {}
    factor_dict_list = multiprocessing_wrapper(
        [
            (__extract_factor_and_formulation_from_one_report, (useful_report_dict[file_name],))
            for file_name in file_name_list
        ],
        n=RD_AGENT_SETTINGS.multi_proc_n,
    )
    for index, file_name in enumerate(file_name_list):
        final_report_factor_dict[file_name] = factor_dict_list[index]
    logger.info(f"Factor extraction completed for {len(final_report_factor_dict)} reports")

    return final_report_factor_dict


def merge_file_to_factor_dict_to_factor_dict(
    file_to_factor_dict: dict[str, dict],
) -> dict:
    factor_dict = {}
    for file_name in file_to_factor_dict:
        for factor_name in file_to_factor_dict[file_name]:
            factor_dict.setdefault(factor_name, [])
            factor_dict[factor_name].append(file_to_factor_dict[file_name][factor_name])

    factor_dict_simple_deduplication = {}
    for factor_name in factor_dict:
        if len(factor_dict[factor_name]) > 1:
            factor_dict_simple_deduplication[factor_name] = max(
                factor_dict[factor_name],
                key=lambda x: len(x["formulation"]),
            )
        else:
            factor_dict_simple_deduplication[factor_name] = factor_dict[factor_name][0]
    return factor_dict_simple_deduplication


def __check_factor_dict_relevance(
    factor_df_string: str,
) -> dict[str, dict[str, str]]:
    extract_result_resp = APIBackend().build_messages_and_create_chat_completion(
        system_prompt=document_process_prompts["factor_relevance_system"],
        user_prompt=factor_df_string,
        json_mode=True,
    )
    return json.loads(extract_result_resp)


def check_factor_relevance(
    factor_dict: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    factor_relevance_dict = {}

    factor_df = pd.DataFrame(factor_dict).T
    factor_df.index.names = ["factor_name"]

    while factor_df.shape[0] > 0:
        result_list = multiprocessing_wrapper(
            [
                (__check_factor_dict_relevance, (factor_df.iloc[i : i + 50, :].to_string(),))
                for i in range(0, factor_df.shape[0], 50)
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )

        for result in result_list:
            for factor_name, relevance in result.items():
                factor_relevance_dict[factor_name] = relevance

        factor_df = factor_df[~factor_df.index.isin(factor_relevance_dict)]

    filtered_factor_dict = {
        factor_name: factor_dict[factor_name]
        for factor_name in factor_dict
        if factor_relevance_dict[factor_name]["relevance"]
    }

    return factor_relevance_dict, filtered_factor_dict


def __check_factor_dict_viability_simulate_json_mode(
    factor_df_string: str,
) -> dict[str, dict[str, str]]:
    extract_result_resp = APIBackend().build_messages_and_create_chat_completion(
        system_prompt=document_process_prompts["factor_viability_system"],
        user_prompt=factor_df_string,
        json_mode=True,
    )
    return json.loads(extract_result_resp)


def check_factor_viability(
    factor_dict: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    factor_viability_dict = {}

    factor_df = pd.DataFrame(factor_dict).T
    factor_df.index.names = ["factor_name"]

    while factor_df.shape[0] > 0:
        result_list = multiprocessing_wrapper(
            [
                (__check_factor_dict_viability_simulate_json_mode, (factor_df.iloc[i : i + 50, :].to_string(),))
                for i in range(0, factor_df.shape[0], 50)
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )

        for result in result_list:
            for factor_name, viability in result.items():
                factor_viability_dict[factor_name] = viability

        factor_df = factor_df[~factor_df.index.isin(factor_viability_dict)]

    filtered_factor_dict = {
        factor_name: factor_dict[factor_name]
        for factor_name in factor_dict
        if factor_viability_dict[factor_name]["viability"]
    }

    return factor_viability_dict, filtered_factor_dict


def __check_factor_duplication_simulate_json_mode(
    factor_df: pd.DataFrame,
) -> list[list[str]]:
    current_user_prompt = factor_df.to_string()

    working_list = [factor_df]
    final_list = []

    while len(working_list) > 0:
        current_df = working_list.pop(0)
        if (
            APIBackend().build_messages_and_calculate_token(
                user_prompt=current_df.to_string(), system_prompt=document_process_prompts["factor_duplicate_system"]
            )
            > LLM_SETTINGS.chat_token_limit
        ):
            working_list.append(current_df.iloc[: current_df.shape[0] // 2, :])
            working_list.append(current_df.iloc[current_df.shape[0] // 2 :, :])
        else:
            final_list.append(current_df)

    generated_duplicated_groups = []
    for current_df in final_list:
        current_factor_to_string = current_df.to_string()
        session = APIBackend().build_chat_session(
            session_system_prompt=document_process_prompts["factor_duplicate_system"],
        )
        for _ in range(10):
            extract_result_resp = session.build_chat_completion(
                user_prompt=current_factor_to_string,
                json_mode=True,
            )
            ret_dict = json.loads(extract_result_resp)
            if len(ret_dict) == 0:
                return generated_duplicated_groups
            else:
                generated_duplicated_groups.extend(ret_dict)
                current_factor_to_string = """Continue to extract duplicated groups. If no more duplicated group found please respond empty dict."""
    return generated_duplicated_groups


def __kmeans_embeddings(embeddings: np.ndarray, k: int = 20) -> list[list[str]]:
    x_normalized = normalize(embeddings)

    np.random.seed(42)

    kmeans = KMeans(
        n_clusters=k,
        init="random",
        max_iter=100,
        n_init=10,
        random_state=42,
    )

    # KMeans algorithm uses Euclidean distance, and we need to customize a function to find the most similar cluster center
    def find_closest_cluster_cosine_similarity(
        data: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        similarity = cosine_similarity(data, centroids)
        return np.argmax(similarity, axis=1)

    # Initializes the cluster center
    rng = np.random.default_rng(seed=42)
    centroids = rng.choice(x_normalized, size=k, replace=False)

    # Iterate until convergence or the maximum number of iterations is reached
    for _ in range(kmeans.max_iter):
        # Assign the sample to the nearest cluster center
        closest_clusters = find_closest_cluster_cosine_similarity(
            x_normalized,
            centroids,
        )

        # update the cluster center
        new_centroids = np.array(
            [x_normalized[closest_clusters == i].mean(axis=0) for i in range(k)],
        )
        new_centroids = normalize(new_centroids)  # 归一化新的簇中心

        # Check whether the cluster center has changed
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    clusters = find_closest_cluster_cosine_similarity(x_normalized, centroids)
    cluster_to_index = {}
    for index, cluster in enumerate(clusters):
        cluster_to_index.setdefault(cluster, []).append(index)
    return sorted(
        cluster_to_index.values(),
        key=lambda x: len(x),
        reverse=True,
    )


def __deduplicate_factor_dict(factor_dict: dict[str, dict[str, str]]) -> list[list[str]]:
    if len(factor_dict) == 0:
        return []
    factor_df = pd.DataFrame(factor_dict).T
    factor_df.index.names = ["factor_name"]

    factor_names = sorted(factor_dict)

    factor_name_to_full_str = {}
    for factor_name in factor_dict:
        description = factor_dict[factor_name]["description"]
        formulation = factor_dict[factor_name]["formulation"]
        variables = factor_dict[factor_name]["variables"]
        factor_name_to_full_str[
            factor_name
        ] = f"""Factor name: {factor_name}
Factor description: {description}
Factor formulation: {formulation}
Factor variables: {variables}
"""

    full_str_list = [factor_name_to_full_str[factor_name] for factor_name in factor_names]
    embeddings = APIBackend.create_embedding(full_str_list)

    target_k = None
    if len(full_str_list) < RD_AGENT_SETTINGS.max_input_duplicate_factor_group:
        kmeans_index_group = [list(range(len(full_str_list)))]
        target_k = 1
    else:
        for k in range(
            len(full_str_list) // RD_AGENT_SETTINGS.max_input_duplicate_factor_group,
            RD_AGENT_SETTINGS.max_kmeans_group_number,
        ):
            kmeans_index_group = __kmeans_embeddings(embeddings=embeddings, k=k)
            if len(kmeans_index_group[0]) < RD_AGENT_SETTINGS.max_input_duplicate_factor_group:
                target_k = k
                logger.info(f"K-means group number: {k}")
                break
    factor_name_groups = [[factor_names[index] for index in index_group] for index_group in kmeans_index_group]

    duplication_names_list = []

    result_list = multiprocessing_wrapper(
        [
            (__check_factor_duplication_simulate_json_mode, (factor_df.loc[factor_name_group, :],))
            for factor_name_group in factor_name_groups
        ],
        n=RD_AGENT_SETTINGS.multi_proc_n,
    )

    duplication_names_list = []

    for deduplication_factor_names_list in result_list:
        filter_factor_names = [
            factor_name for factor_name in set(deduplication_factor_names_list) if factor_name in factor_dict
        ]
        if len(filter_factor_names) > 1:
            duplication_names_list.append(filter_factor_names)

    return duplication_names_list


def deduplicate_factors_by_llm(  # noqa: C901, PLR0912
    factor_dict: dict[str, dict[str, str]],
    factor_viability_dict: dict[str, dict[str, str]] | None = None,
) -> list[list[str]]:
    final_duplication_names_list = []
    current_round_factor_dict = factor_dict

    # handle multi-round deduplication
    for _ in range(10):
        duplication_names_list = __deduplicate_factor_dict(current_round_factor_dict)

        new_round_names = []
        for duplication_names in duplication_names_list:
            if len(duplication_names) < RD_AGENT_SETTINGS.max_output_duplicate_factor_group:
                final_duplication_names_list.append(duplication_names)
            else:
                new_round_names.extend(duplication_names)

        if len(new_round_names) != 0:
            current_round_factor_dict = {factor_name: factor_dict[factor_name] for factor_name in new_round_names}
        else:
            break

    # sort the final list of duplicates by their length, largest first
    final_duplication_names_list = sorted(final_duplication_names_list, key=lambda x: len(x), reverse=True)

    to_replace_dict = {}  # to map duplicates to the target factor names
    for duplication_names in duplication_names_list:
        if factor_viability_dict is not None:
            # check viability of each factor in the duplicates group
            viability_list = [factor_viability_dict[name]["viability"] for name in duplication_names]
            if True not in viability_list:
                continue
            target_factor_name = duplication_names[viability_list.index(True)]
        else:
            target_factor_name = duplication_names[0]
        for duplication_factor_name in duplication_names:
            if duplication_factor_name == target_factor_name:
                continue
            to_replace_dict[duplication_factor_name] = target_factor_name

    llm_deduplicated_factor_dict = {}
    added_lower_name_set = set()
    for factor_name in factor_dict:
        # only add factors that haven't been replaced and are not duplicates
        if factor_name not in to_replace_dict and factor_name.lower() not in added_lower_name_set:
            if factor_viability_dict is not None and not factor_viability_dict[factor_name]["viability"]:
                continue
            added_lower_name_set.add(factor_name.lower())
            llm_deduplicated_factor_dict[factor_name] = factor_dict[factor_name]

    return llm_deduplicated_factor_dict, final_duplication_names_list


class FactorExperimentLoaderFromPDFfiles(FactorExperimentLoader):
    def load(self, file_or_folder_path: str) -> dict:
        with logger.tag("docs"):
            docs_dict = load_and_process_pdfs_by_langchain(file_or_folder_path)
            logger.log_object(docs_dict)

        selected_report_dict = classify_report_from_dict(report_dict=docs_dict, vote_time=1)

        with logger.tag("file_to_factor_result"):
            file_to_factor_result = extract_factors_from_report_dict(docs_dict, selected_report_dict)
            logger.log_object(file_to_factor_result)

        with logger.tag("factor_dict"):
            factor_dict = merge_file_to_factor_dict_to_factor_dict(file_to_factor_result)
            logger.log_object(factor_dict)

        with logger.tag("filtered_factor_dict"):
            factor_viability, filtered_factor_dict = check_factor_viability(factor_dict)
            logger.log_object(filtered_factor_dict)

        # factor_dict, duplication_names_list = deduplicate_factors_by_llm(factor_dict, factor_viability)

        return FactorExperimentLoaderFromDict().load(filtered_factor_dict)
