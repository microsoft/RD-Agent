from pathlib import Path
from typing import Dict

import yaml

from rdagent.core.utils import SingletonBaseClass


class Prompts(SingletonBaseClass, Dict[str, str]):
    def __init__(self, file_path: Path) -> None:
        with file_path.open(encoding="utf8") as file:
            prompt_yaml_dict = yaml.safe_load(file)

        if prompt_yaml_dict is None:
            error_message = f"Failed to load prompts from {file_path}"
            raise ValueError(error_message)

        for key, value in prompt_yaml_dict.items():
            self[key] = value
