import json
import os
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from rdagent.core.prompts import Prompts
from rdagent.oai.llm_utils import APIBackend

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


def extract_knowledge_from_high_score_answers(content: str):
    sys_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["extract_kaggle_knowledge_prompts"]["system"])
        .render()
    )

    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["extract_kaggle_knowledge_prompts"]["user"])
        .render(file_content=content)
    )

    response_analysis = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=True,
    )

    try:
        response_json_analysis = json.loads(response_analysis)
    except json.JSONDecodeError:
        response_json_analysis = {"error": "Failed to parse LLM's response as JSON"}

    return response_json_analysis


def extract_knowledge_from_feedback(feedback_response: dict) -> dict:
    """
    Extracts knowledge from LLM-generated feedback and structures it.
    """
    sys_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["extract_kaggle_knowledge_from_feedback_prompts"]["system"])
        .render()
    )

    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["extract_kaggle_knowledge_from_feedback_prompts"]["user"])
        .render(experiment_strategy=feedback_response)
    )

    response_analysis = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=True,
    )

    try:
        response_json_analysis = json.loads(response_analysis)
    except json.JSONDecodeError:
        response_json_analysis = {"error": "Failed to parse LLM's response as JSON"}

    return response_json_analysis


def process_all_case_files(directory_path: str):
    output_file = Path(directory_path) / "kaggle_experience_results.json"
    json_output = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".case"):
            file_path = os.path.join(directory_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                knowladge = extract_knowledge_from_high_score_answers(content)
                json_output.append(knowladge)

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(json_output, json_file, ensure_ascii=False)


if __name__ == "__main__":
    process_all_case_files(directory_path="git_ignore_folder/experience/tabular_cases_all")
