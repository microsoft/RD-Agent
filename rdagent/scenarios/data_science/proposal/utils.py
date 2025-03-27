import json
import re
from typing import Dict, List
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

def extract_JSON(text) -> List | List[Dict]:
    # Try to directly load the text as JSON
    try:
        extracted_text = json.loads(text)
        if isinstance(extracted_text, dict):
            return [extracted_text]
        elif isinstance(extracted_text, list):
            return extracted_text
        else:
            return []
    except json.JSONDecodeError:
        pass
    
    # Try to extract the outermost JSON array
    try:
        match = re.search(r'\[(?:[^\[\]]|\[.*\])*\]', text)
        if match:
            extracted_text = json.loads(match.group(0))
            return extracted_text
    except json.JSONDecodeError:
        pass
    
    # Try to extract the first JSON object
    try:
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            extracted_text = json.loads(match.group(0))
            return [extracted_text]
    except json.JSONDecodeError:
        pass

    return []


def identify_scenario_problem() -> List[Dict]:
    sys_prompt = T(".prompts2:scenario_problem.system").r()
    user_prompt = T(".prompts2:scenario_problem.user").r()

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )
    return extract_JSON(response)


def identify_feedback_problem() -> List[Dict]:
    sys_prompt = T(".prompts2:feedback_problem.system").r()
    user_prompt = T(".prompts2:feedback_problem.user").r()

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )
    return extract_JSON(response)


def solution_gen() -> Dict:
    sys_prompt = T(".prompts2:solution_gen.system").r()
    user_prompt = T(".prompts2:solution_gen.user").r()

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )
    return extract_JSON(response)[0]


def solution_rank(solution: List[Dict]) -> Dict:
    pass

def task_gen(solution):
    sys_prompt = T(".prompts2:task_gen.system").r()
    user_prompt = T(".prompts2:task_gen.user").r()

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )
    return extract_JSON(response)[0]