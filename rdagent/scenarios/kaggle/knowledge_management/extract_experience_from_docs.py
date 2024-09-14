import json
import os
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from rdagent.core.prompts import Prompts
from rdagent.oai.llm_utils import APIBackend

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


def process_with_gpt(content: str):
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


def process_all_case_files(directory_path: str):
    output_file = Path(directory_path) / "kaggle_experience_results.json"
    json_output = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".case"):
            file_path = os.path.join(directory_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                gpt_response = process_with_gpt(content)
                json_output.append(gpt_response)

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(json_output, json_file, ensure_ascii=False)


if __name__ == "__main__":
    process_all_case_files(directory_path="git_ignore_folder/experience/tabular_cases_all")
