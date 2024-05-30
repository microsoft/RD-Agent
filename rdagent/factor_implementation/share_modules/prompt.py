from pathlib import Path
from typing import Dict

import yaml
from finco.utils import SingletonBaseClass


class FactorImplementationPrompts(Dict, SingletonBaseClass):
    def __init__(self):
        super().__init__()
        prompt_yaml_path = Path(__file__).parent / "prompts.yaml"

        prompt_yaml_dict = yaml.load(
            open(
                prompt_yaml_path,
                encoding="utf8",
            ),
            Loader=yaml.FullLoader,
        )

        for key, value in prompt_yaml_dict.items():
            self[key] = value
