from __future__ import annotations

import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd
import tiktoken
import yaml
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from jinja2 import Template
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from core.conf import FincoSettings as Config
from oai.llm_utils import APIBackend, create_embedding_with_multiprocessing
from core.log import FinCoLog

if TYPE_CHECKING:
    from langchain_core.documents import Document

from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

with (Path(__file__).parent / "util_prompt.yaml").open(encoding="utf8") as f:
    UTIL_PROMPT = yaml.safe_load(
        f,
    )


def load_documents_by_langchain(path: Path) -> list:
    """Load documents from the specified path.

    Args:
        path (str): The path to the directory or file containing the documents.

    Returns:
        list: A list of loaded documents.
    """
    loader = PyPDFDirectoryLoader(str(path), silent_errors=True) if path.is_dir() else PyPDFLoader(str(path))
    return loader.load()


def process_documents_by_langchain(docs: list[Document]) -> dict[str, str]:
    """Process a list of documents and group them by document name.

    Args:
        docs (list): A list of documents.

    Returns:
        dict: A dictionary where the keys are document names and the values are
        the concatenated content of the documents.
    """
    content_dict = {}

    for doc in docs:
        doc_name = str(Path(doc.metadata["source"]).resolve())
        doc_content = doc.page_content

        if doc_name not in content_dict:
            content_dict[str(doc_name)] = doc_content
        else:
            content_dict[str(doc_name)] += doc_content

    return content_dict


def load_and_process_pdfs_by_langchain(path: Path) -> dict[str, str]:
    return process_documents_by_langchain(load_documents_by_langchain(path))


def load_and_process_one_pdf_by_azure_document_intelligence(
    path: Path,
    key: str,
    endpoint: str,
) -> str:
    pages = len(PyPDFLoader(str(path)).load())
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    with path.open("rb") as file:
        result = document_analysis_client.begin_analyze_document(
            "prebuilt-document",
            file,
            pages=f"1-{pages}",
        ).result()
    return result.content


def load_and_process_pdfs_by_azure_document_intelligence(path: Path) -> dict[str, str]:
    config = Config()

    assert config.azure_document_intelligence_key is not None
    assert config.azure_document_intelligence_endpoint is not None

    content_dict = {}
    ab_path = path.resolve()
    if ab_path.is_file():
        assert ".pdf" in ab_path.suffixes, "The file must be a PDF file."
        proc = load_and_process_one_pdf_by_azure_document_intelligence
        content_dict[str(ab_path)] = proc(
            ab_path,
            config.azure_document_intelligence_key,
            config.azure_document_intelligence_endpoint,
        )
    else:
        for file_path in ab_path.rglob("*"):
            if file_path.is_file() and ".pdf" in file_path.suffixes:
                content_dict[str(file_path)] = load_and_process_one_pdf_by_azure_document_intelligence(
                    file_path,
                    config.azure_document_intelligence_key,
                    config.azure_document_intelligence_endpoint,
                )
    return content_dict


def classify_report_from_dict(
    report_dict: Mapping[str, str],
    api: APIBackend,
    input_max_token: int = 128000,
    vote_time: int = 1,
    substrings: tuple[str] = (),
) -> dict[str, dict[str, str]]:
    """
    Parameters:
    - report_dict (Dict[str, str]):
      A dictionary where the key is the path of the report (ending with .pdf),
      and the value is either the report content as a string.
    - api (APIBackend): An instance of the APIBackend class.
    - input_max_token (int): Specifying the maximum number of input tokens.
    - vote_time (int): An integer specifying how many times to vote.
    - substrings (list(str)): List of hardcode substrings.

    Returns:
    - Dict[str, Dict[str, str]]: A dictionary where each key is the path of the report,
      with a single key 'class' and its value being the classification result (0 or 1).

    """
    if len(substrings) == 0:
        substrings = (
            "FinCo",
            "金融工程",
            "金工",
            "回测",
            "因子",
            "机器学习",
            "深度学习",
            "量化",
        )

    res_dict = {}
    classify_prompt = UTIL_PROMPT["classify_system"]
    enc = tiktoken.encoding_for_model("gpt-4-turbo")

    for key, value in report_dict.items():
        if not key.endswith(".pdf"):
            continue
        file_name = key

        if isinstance(value, str):
            content = value
        else:
            FinCoLog().warning(f"输入格式不符合要求: {file_name}")
            res_dict[file_name] = {"class": 0}
            continue

        if not any(substring in content for substring in substrings):
            res_dict[file_name] = {"class": 0}
        else:
            gpt_4_max_token = 128000
            if input_max_token < gpt_4_max_token:
                content = enc.encode(content)
                max_token_1 = max(0, min(len(content), input_max_token) - 1)
                content = enc.decode(content[:max_token_1])

            vote_list = []
            for _ in range(vote_time):
                user_prompt = content
                system_prompt = classify_prompt
                res = api.build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                )
                try:
                    res = json.loads(res)
                    vote_list.append(int(res["class"]))
                except json.JSONDecodeError:
                    FinCoLog().warning(f"返回值无法解析: {file_name}")
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
        session_system_prompt=UTIL_PROMPT["extract_factors_system"],
    )

    extracted_factor_dict = {}
    current_user_prompt = content

    for _ in range(10):
        extract_result_resp = session.build_chat_completion(
            user_prompt=current_user_prompt,
            json_mode=False,
        )
        re_search_res = re.search(r"```json(.*)```", extract_result_resp, re.S)
        ret_json_str = re_search_res.group(1) if re_search_res is not None else ""
        try:
            ret_dict = json.loads(ret_json_str)
            parse_success = bool(isinstance(ret_dict, dict)) and "factors" in ret_dict
        except json.JSONDecodeError:
            parse_success = False
        if ret_json_str is None or not parse_success:
            current_user_prompt = (
                "Your response didn't follow the instruction" " might be wrong json format. Try again."
            )
        else:
            factors = ret_dict["factors"]
            if len(factors) == 0:
                break
            for factor_name, factor_description in factors.items():
                extracted_factor_dict[factor_name] = factor_description
            current_user_prompt = UTIL_PROMPT["extract_factors_follow_user"]

    return extracted_factor_dict


def __extract_factors_formulation_from_content(
    content: str,
    factor_dict: dict[str, str],
) -> dict[str, dict[str, str]]:
    factor_dict_df = pd.DataFrame(
        factor_dict.items(),
        columns=["factor_name", "factor_description"],
    )

    system_prompt = UTIL_PROMPT["extract_factor_formulation_system"]
    current_user_prompt = Template(
        UTIL_PROMPT["extract_factor_formulation_user"],
    ).render(report_content=content, factor_dict=factor_dict_df.to_string())

    session = APIBackend().build_chat_session(session_system_prompt=system_prompt)
    factor_to_formulation = {}

    for _ in range(10):
        extract_result_resp = session.build_chat_completion(
            user_prompt=current_user_prompt,
            json_mode=False,
        )
        re_search_res = re.search(r"```json(.*)```", extract_result_resp, re.S)
        ret_json_str = re_search_res.group(1) if re_search_res is not None else ""
        try:
            ret_dict = json.loads(ret_json_str)
            parse_success = bool(isinstance(ret_dict, dict))
        except json.JSONDecodeError:
            parse_success = False
        if ret_json_str is None or not parse_success:
            current_user_prompt = (
                "Your response didn't follow the instruction" " might be wrong json format. Try again."
            )
        else:
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


def extract_factor_and_formulation_from_one_report(
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


def extract_factors_from_report_dict_and_classify_result(
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
            FinCoLog().warning(f"输入格式不符合要求: {key}")

    final_report_factor_dict = {}
    # for file_name, content in useful_report_dict.items():
    #     final_report_factor_dict.setdefault(file_name, {})
    #     final_report_factor_dict[
    #         file_name
    #     ] = extract_factor_and_formulation_from_one_report(content)

    while len(final_report_factor_dict) != len(useful_report_dict):
        pool = mp.Pool(n_proc)
        pool_result_list = []
        file_names = []
        for file_name, content in useful_report_dict.items():
            if file_name in final_report_factor_dict:
                continue
            file_names.append(file_name)
            pool_result_list.append(
                pool.apply_async(
                    extract_factor_and_formulation_from_one_report,
                    (content,),
                ),
            )

        pool.close()
        pool.join()

        for index, result in enumerate(pool_result_list):
            if result.get is not None:
                file_name = file_names[index]
                final_report_factor_dict.setdefault(file_name, {})
                final_report_factor_dict[file_name] = result.get()
        FinCoLog().info(f"已经完成{len(final_report_factor_dict)}个报告的因子提取")

    return final_report_factor_dict


def check_factor_dict_viability_simulate_json_mode(
    factor_df_string: str,
) -> dict[str, dict[str, str]]:
    session = APIBackend().build_chat_session(
        session_system_prompt=UTIL_PROMPT["factor_viability_system"],
    )
    current_user_prompt = factor_df_string

    for _ in range(10):
        extract_result_resp = session.build_chat_completion(
            user_prompt=current_user_prompt,
            json_mode=False,
        )
        re_search_res = re.search(r"```json(.*)```", extract_result_resp, re.S)
        ret_json_str = re_search_res.group(1) if re_search_res is not None else ""
        try:
            ret_dict = json.loads(ret_json_str)
            parse_success = bool(isinstance(ret_dict, dict))
        except json.JSONDecodeError:
            parse_success = False
        if ret_json_str is None or not parse_success:
            current_user_prompt = (
                "Your response didn't follow the " "instruction might be wrong json format. Try again."
            )
        else:
            return ret_dict
    return {}


def check_factor_dict_viability(
    factor_dict: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    factor_viability_dict = {}

    factor_df = pd.DataFrame(factor_dict).T
    factor_df.index.names = ["factor_name"]

    while factor_df.shape[0] > 0:
        pool = mp.Pool(8)

        result_list = []
        for i in range(0, factor_df.shape[0], 50):
            target_factor_df_string = factor_df.iloc[i : i + 50, :].to_string()

            result_list.append(
                pool.apply_async(
                    check_factor_dict_viability_simulate_json_mode,
                    (target_factor_df_string,),
                ),
            )

        pool.close()
        pool.join()

        for result in result_list:
            respond = result.get()
            for factor_name, viability in respond.items():
                factor_viability_dict[factor_name] = viability

        factor_df = factor_df[~factor_df.index.isin(factor_viability_dict)]

    return factor_viability_dict


def check_factor_duplication_simulate_json_mode(
    factor_df: pd.DataFrame,
) -> list[list[str]]:
    session = APIBackend().build_chat_session(
        session_system_prompt=UTIL_PROMPT["factor_duplicate_system"],
    )
    current_user_prompt = factor_df.to_string()

    generated_duplicated_groups = []
    for _ in range(20):
        extract_result_resp = session.build_chat_completion(
            user_prompt=current_user_prompt,
            json_mode=False,
        )
        re_search_res = re.search(r"```json(.*)```", extract_result_resp, re.S)
        ret_json_str = re_search_res.group(1) if re_search_res is not None else ""
        try:
            ret_dict = json.loads(ret_json_str)
            parse_success = bool(isinstance(ret_dict, list))
        except json.JSONDecodeError:
            parse_success = False
        if ret_json_str is None or not parse_success:
            current_user_prompt = (
                "Your previous response didn't follow"
                " the instruction might be wrong json"
                " format. Try reducing the factors."
            )
        elif len(ret_dict) == 0:
            return generated_duplicated_groups
        else:
            generated_duplicated_groups.extend(ret_dict)
            current_user_prompt = (
                "Continue to extract duplicated"
                " groups. If no more duplicated group"
                " found please respond empty dict."
            )
    return generated_duplicated_groups


def kmeans_embeddings(embeddings: np.ndarray, k: int = 20) -> list[list[str]]:
    x_normalized = normalize(embeddings)

    kmeans = KMeans(
        n_clusters=k,
        init="random",
        max_iter=100,
        n_init=10,
        random_state=42,
    )

    # KMeans算法使用欧氏距离, 我们需要自定义一个函数来找到最相似的簇中心
    def find_closest_cluster_cosine_similarity(
        data: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        similarity = cosine_similarity(data, centroids)
        return np.argmax(similarity, axis=1)

    # 初始化簇中心
    rng = np.random.default_rng()
    centroids = rng.choice(x_normalized, size=k, replace=False)

    # 迭代直到收敛或达到最大迭代次数
    for _ in range(kmeans.max_iter):
        # 分配样本到最近的簇中心
        closest_clusters = find_closest_cluster_cosine_similarity(
            x_normalized,
            centroids,
        )

        # 更新簇中心
        new_centroids = np.array(
            [x_normalized[closest_clusters == i].mean(axis=0) for i in range(k)],
        )
        new_centroids = normalize(new_centroids)  # 归一化新的簇中心

        # 检查簇中心是否发生变化
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


def deduplicate_factor_dict(factor_dict: dict[str, dict[str, str]]) -> list[list[str]]:
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
    embeddings = create_embedding_with_multiprocessing(full_str_list)

    target_k = None
    if len(full_str_list) < Config().max_input_duplicate_factor_group:
        kmeans_index_group = [list(range(len(full_str_list)))]
        target_k = 1
    else:
        for k in range(
            len(full_str_list) // Config().max_input_duplicate_factor_group,
            30,
        ):
            kmeans_index_group = kmeans_embeddings(embeddings=embeddings, k=k)
            if len(kmeans_index_group[0]) < Config().max_input_duplicate_factor_group:
                target_k = k
                FinCoLog().info(f"K-means group number: {k}")
                break
    factor_name_groups = [[factor_names[index] for index in index_group] for index_group in kmeans_index_group]

    duplication_names_list = []

    pool = mp.Pool(target_k)
    result_list = []
    result_list = [
        pool.apply_async(
            check_factor_duplication_simulate_json_mode,
            (factor_df.loc[factor_name_group, :],),
        )
        for factor_name_group in factor_name_groups
    ]

    pool.close()
    pool.join()

    for result in result_list:
        deduplication_factor_names_list = result.get()
        for deduplication_factor_names in deduplication_factor_names_list:
            filter_factor_names = [
                factor_name for factor_name in set(deduplication_factor_names) if factor_name in factor_dict
            ]
            if len(filter_factor_names) > 1:
                duplication_names_list.append(filter_factor_names)

    return duplication_names_list


def deduplicate_factors_several_times(
    factor_dict: dict[str, dict[str, str]],
) -> list[list[str]]:
    final_duplication_names_list = []
    current_round_factor_dict = factor_dict
    for _ in range(10):
        duplication_names_list = deduplicate_factor_dict(current_round_factor_dict)

        new_round_names = []
        for duplication_names in duplication_names_list:
            if len(duplication_names) < Config().max_output_duplicate_factor_group:
                final_duplication_names_list.append(duplication_names)
            else:
                new_round_names.extend(duplication_names)

        if len(new_round_names) != 0:
            current_round_factor_dict = {factor_name: factor_dict[factor_name] for factor_name in new_round_names}
        else:
            return final_duplication_names_list
    return []
