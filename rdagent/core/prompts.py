from pathlib import Path
from typing import Dict

import yaml
from rdagent.core.utils import SingletonBaseClass


class Prompts(Dict[str, str], SingletonBaseClass):
    def __init__(self, file_path: Path):
        prompt_yaml_dict = yaml.load(
            open(
                file_path,
                encoding="utf8",
            ),
            Loader=yaml.FullLoader,
        )

        if prompt_yaml_dict is None:
            raise ValueError(f"Failed to load prompts from {file_path}")

        for key, value in prompt_yaml_dict.items():
            self[key] = value
