"""
Here are some infrastruture to build a agent

The motivation of tempalte and AgentOutput Design
"""

import inspect
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from rdagent.core.utils import SingletonBaseClass

DIRNAME = Path(__file__).absolute().resolve().parent
PROJ_PATH = DIRNAME.parent.parent


# class T(SingletonBaseClass): TODO: singleton does not support args now.
class RDAT:
    """
    RD-Agent's Template
    Use the simplest way to (C)reate a Template and (r)ender it!!
    """

    def __init__(self, uri: str):
        """
        here are some uri usages
            case 1) "a.b.c:x.y.z"
                It will load DIRNAME/a/b/c.yaml as `yaml` and load yaml[x][y][z]
            case 2) ".c:x.y.z"
                It will load c.yaml in caller's (who call `T(uri)`) directory as `yaml` and load yaml[x][y][z]

            the loaded content will be saved in `self.template`
        """
        # Inspect the calling stack to get the caller's directory
        stack = inspect.stack()
        caller_frame = stack[1]
        caller_module = inspect.getmodule(caller_frame[0])
        caller_dir = Path(caller_module.__file__).parent

        # Parse the URI
        path_part, yaml_path = uri.split(":")
        yaml_keys = yaml_path.split(".")

        if path_part.startswith("."):
            yaml_file_path = caller_dir / f"{path_part[1:].replace('.', '/')}.yaml"
        else:
            yaml_file_path = (PROJ_PATH / path_part.replace(".", "/")).with_suffix(".yaml")

        # Load the YAML file
        with open(yaml_file_path, "r") as file:
            yaml_content = yaml.safe_load(file)

        # Traverse the YAML content to get the desired template
        for key in yaml_keys:
            yaml_content = yaml_content[key]

        self.template = yaml_content

    def r(self, **context: Any):
        """
        Render the template with the given context.
        """
        return Environment(undefined=StrictUndefined).from_string(self.template).render(**context)


T = RDAT  # shortcuts
