from __future__ import annotations

import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import tiktoken
from jinja2 import Template
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.log import RDAgentLog
from rdagent.core.prompts import Prompts
from rdagent.core.task import TaskLoader
from rdagent.document_reader.document_reader import load_and_process_pdfs_by_langchain
from rdagent.model_implementation.task import ModelImplementationTaskLoaderFromDict
from rdagent.oai.llm_utils import APIBackend, create_embedding_with_multiprocessing


document_process_prompts = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

def extract_model_from_doc(doc_content: str) -> dict:
    """
    Extract model information from document content.

    Parameters
    ----------
    doc_content : str
        Document content.

    Returns
    -------
    dict
        {factor_name: dict{description, formulation, variables}}
    """
    session = APIBackend().build_chat_session(
        session_system_prompt=document_process_prompts["extract_model_formulation_system"],
    )
    current_user_prompt = doc_content

    # Extract model information from document content.
    model_dict = {}

    for _ in range(10):
        # try to extract model information from the document content, retry at most 10 times.
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
            current_user_prompt = "Your response didn't follow the instruction might be wrong json format. Try again."
        else:
            for name, formulation_and_description in ret_dict.items():
                if name not in model_dict:
                    model_dict[name] = formulation_and_description
            if len(model_dict) == 0:
                current_user_prompt = "No model extracted. Please try again."
            else:
                break


    RDAgentLog().info(f"已经完成{len(model_dict)}个模型的提取")

    return model_dict



def extract_model_from_docs(docs_dict):
    model_dict = {}
    for doc_name, doc_content in docs_dict.items():
        model_dict[doc_name] = extract_model_from_doc(doc_content)
    return model_dict


class ModelImplementationTaskLoaderFromPDFfiles(TaskLoader):
    def load(self, file_or_folder_path: Path) -> dict:
        docs_dict = load_and_process_pdfs_by_langchain(Path(file_or_folder_path)) # dict{file_path:content}
        model_dict = extract_model_from_docs(docs_dict)
        # take in dict {factor_name: dict{description, formulation, variables}}
        return ModelImplementationTaskLoaderFromDict().load(model_dict)

if __name__ == "__main__":
    doc_dict = load_and_process_pdfs_by_langchain(Path("../test_doc"))
    print(doc_dict.keys()) # if you run code like "python -u", the print content will be truncated