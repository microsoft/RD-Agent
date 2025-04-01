"""
Here are some infrastructure to build a agent

The motivation of template and AgentOutput Design
"""

import inspect
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FunctionLoader, StrictUndefined

from rdagent.log import rdagent_logger as logger

DIRNAME = Path(__file__).absolute().resolve().parent
PROJ_PATH = DIRNAME.parent.parent


def get_caller_dir(upshift: int = 0) -> Path:
    # Inspect the calling stack to get the caller's directory
    stack = inspect.stack()
    caller_frame = stack[1 + upshift]
    caller_module = inspect.getmodule(caller_frame[0])
    if caller_module and caller_module.__file__:
        caller_dir = Path(caller_module.__file__).parent
    else:
        caller_dir = DIRNAME
    return caller_dir


def load_yaml_content(uri: str, caller_dir: Path | None = None) -> Any:
    """
    Please refer to RDAT.__init__ file
    """
    if caller_dir is None:
        caller_dir = get_caller_dir(upshift=1)
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

    return yaml_content


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

                Form example, if you want to load "rdagent/scenarios/kaggle/experiment/prompts.yaml"
                `a.b.c` should be "scenarios.kaggle.experiment.prompts" and "rdagent" should be exclude
            case 2) ".c:x.y.z"
                It will load c.yaml in caller's (who call `T(uri)`) directory as `yaml` and load yaml[x][y][z]

            the loaded content will be saved in `self.template`
        """
        self.uri = uri
        caller_dir = get_caller_dir(1)
        if uri.startswith("."):
            try:
                # modify the uri to a raltive path to the project for easier finding prompts.yaml
                self.uri = f"{str(caller_dir.resolve().relative_to(PROJ_PATH)).replace('/', '.')}{uri}"
            except ValueError:
                pass
        self.template = load_yaml_content(uri, caller_dir=caller_dir)

    def r(self, **context: Any) -> str:
        """
        Render the template with the given context.
        """
        # loader=FunctionLoader(load_yaml_content) is for supporting grammar like below.
        # `{% include "scenarios.data_science.share:component_spec.DataLoadSpec" %}`
        rendered = (
            Environment(undefined=StrictUndefined, loader=FunctionLoader(load_yaml_content))
            .from_string(self.template)
            .render(**context)
            .strip("\n")
        )
        while "\n\n\n" in rendered:
            rendered = rendered.replace("\n\n\n", "\n\n")
        rendered = "\n".join(line for line in rendered.splitlines() if line.strip())
        logger.log_object(
            obj={
                "uri": self.uri,
                "template": self.template,
                "context": context,
                "rendered": rendered,
            },
            tag="debug_tpl",
        )
        return rendered


T = RDAT  # shortcuts
