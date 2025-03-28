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


def identify_scenario_problem(component_desc, scenario_desc, competition_desc, sota_exp_desc) -> List[Dict]:
    sys_prompt = T(".prompts2:scenario_problem.system").r(
        component_desc=component_desc,
        problem_spec = T(".prompts2:specification.problem").r()
        problem_output_format = T(".prompts2:output_format.problem").r()
    )
    user_prompt = T(".prompts2:scenario_problem.user").r(
        scenario_desc=scenario_desc,
        competition_desc=competition_desc,
        sota_exp_desc=sota_exp_desc
    )
    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )
    return extract_JSON(response)


def identify_feedback_problem(component_desc, scenario_desc, trace_desc_df, sota_exp_desc) -> List[Dict]:
    sys_prompt = T(".prompts2:scenario_problem.system").r(
        component_desc=component_desc,
        problem_spec = T(".prompts2:specification.problem").r()
        problem_output_format = T(".prompts2:output_format.problem").r()
    )
    user_prompt = T(".prompts2:feedback_problem.user").r(
        scenario_desc=scenario_desc,
        trace_desc_df=trace_desc_df,
        sota_exp_desc=sota_exp_desc
    )
    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )
    return extract_JSON(response)


def hypothesis_gen(component_desc, scenario_desc, trace_desc_df, sota_exp_desc, problems) -> Dict:
    sys_prompt = T(".prompts2:hypothesis_gen.system").r(
        component_desc = component_desc,
        hypothesis_spec = T(".prompts2:specification.hypothesis").r()
        hypothesis_output_format = T(".prompts2:output_format.hypothesis").r()
    )
    user_prompt = T(".prompts2:hypothesis_gen.user").r(
        scenario_desc = scenario_desc,
        trace_desc_df = trace_desc_df,
        sota_exp_desc = sota_exp_desc,
        problems = problems
    )
    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )
    return extract_JSON(response)