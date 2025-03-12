from pathlib import Path
import pickle
class Saver:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def dump(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with path.open("rb") as f:
            return pickle.load(f)


import json
import re
def extract_JSON(text):
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


from jinja2 import Environment, StrictUndefined
from rdagent.core.prompts import Prompts
from rdagent.oai.llm_utils import APIBackend
def solution_to_data(competition_description, solution) -> str:
    prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")
    sys_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["solution_to_data"]["system"])
        .render()
    )

    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["solution_to_data"]["user"])
        .render(competition_description=competition_description, 
                solution=solution)
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    return response


def solution_to_problem(competition_description, solution, feedback) -> str:
    prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")
    sys_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["solution_to_problem"]["system"])
        .render()
    )

    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["solution_to_problem"]["user"])
        .render(competition_description=competition_description, 
                solution=solution, 
                feedback=feedback)
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    return response


def extract_features(raw_features, ftype):
    features = extract_JSON(raw_features)
    extracted_features = []
    if features is not None:
        for feat in features:
            # data features
            if ftype == "data":
                if feat['Assessment'].lower() == 'no':
                    feature = f"{feat['Characteristic']}"
                    extracted_features.append({"label": "DATA", "feature": feature})
            # problem features
            elif ftype == "problem":
                feature = f"{feat['Problem']}"
                extracted_features.append({"label": "PROBLEM", "feature": feature})
            elif ftype == "reason":
                pass
            else:
                raise NotImplementedError
    return extracted_features