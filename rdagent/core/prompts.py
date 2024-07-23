from __future__ import annotations

from pathlib import Path
from typing import Iterator, Dict
from collections import UserDict

import yaml
from rdagent.core.utils import SingletonBaseClass


class Prompts(SingletonBaseClass):
    def __init__(self, file_path: Path) -> None:
        self._prompts: dict[str, str] = {}
        with file_path.open(encoding="utf8") as file:
            prompt_yaml_dict = yaml.safe_load(file)

        if prompt_yaml_dict is None:
            error_message = f"Failed to load prompts from {file_path}"
            raise ValueError(error_message)

        for key, value in prompt_yaml_dict.items():
            self._prompts[key] = value

    def __getitem__(self, key: str) -> str:
        return self._prompts[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._prompts[key] = value

    def __iter__(self) -> Iterator[str]:
        return iter(self._prompts)

    def __len__(self) -> int:
        return len(self._prompts)

    def items(self) -> Iterator[tuple[str, str]]:
        return iter(self._prompts.items())

    def keys(self) -> Iterator[str]:
        return iter(self._prompts.keys())

    def values(self) -> Iterator[str]:
        return iter(self._prompts.values())
