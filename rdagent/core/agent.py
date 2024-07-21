"""
Motivation
==========
The system has a workflow with several steps, each designed for a specific purpose.

The Agent system simplifies the following tasks:
- Implemented:
    - If you have an agent named `GrammarAgent` in the file `gagent.py`, it will automatically load the prompts from the `prompt.GrammarAgent.yaml` file.
    - While `StructPrompt` may impose some restrictions on the prompts, it offers guidance and reusable code for most cases.
- Planned:
    - [ ] Aligning the output to a python dataclass instead of puring string.


Usage
=====
Any module in your system that behave like a function and require intelligence can be an Agent.
You can equip it via multiple inheritance with the Agent class.
"""
from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Any
from jinja2 import Environment, StrictUndefined
from rdagent.core.prompts import Prompts
from rdagent.oai.llm_utils import APIBackend


@dataclass
class StructPrompt:
    """
    A predefined user prompt;
    """
    system_tpl: str
    user_tpl: str = "{{user_input}}"
    demo: list[tuple[str, str]] = field(default_factory=list)

    # if `add_demo_to_system_prompt` then the demo will be added to the system prompt based on `system_demo_tpl`
    system_demo_tpl: str = """

Here are some demonstrations

{% for user, content in demo %}
Demo{{loop.index}}
    User:
    ```
    {{ user }}
    ```
    Assisant:
    ```
    {{ content }}
    ```
{% endfor %}
"""

class Agent:
    prompt: StructPrompt
    add_demo_to_system_prompt: bool = True  # Normally, LLM will attent more on system prompt.

    def __init__(self):
        class_file_path = Path(inspect.getfile(self.__class__))
        self.class_file_path = class_file_path
        prompt_file_path = class_file_path.parent / f"prompt.{self.__class__.__name__}.yaml"
        print(f"loading {class_file_path}:{self.__class__.__name__} prompts from {prompt_file_path}")
        Prompt = Prompts(file_path=prompt_file_path)
        self.prompt = StructPrompt(**Prompt)

    def _call(self, user_input: str, context: dict[str, str] = {}, json_mode: bool = False) -> Any:
        user_prompt = Environment(undefined=StrictUndefined).from_string(self.prompt.user_tpl).render(
            user_input=user_input, **context)
        system_prompt = Environment(undefined=StrictUndefined).from_string(self.prompt.system_tpl).render(**context)

        # former messages as demo
        former_messages = []
        if not self.add_demo_to_system_prompt:
            for q, a in self.prompt.demo:
                former_messages.append({"role": "user", "content": q})
                former_messages.append({"role": "assistant", "content": a})
        else:
            # add demo to system prompt
            if self.prompt.demo:
                system_prompt += Environment(undefined=StrictUndefined).from_string(self.prompt.system_demo_tpl).render(demo=self.prompt.demo)

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, former_messages=former_messages, json_mode=json_mode)
        return resp

    def call(self, user_input: str, context: dict[str, str] = {}) -> str:
        return self._call(user_input, context, json_mode=False)

    def call_json(self, user_input: str, context: dict[str, str] = {}) -> Any:
        return self._call(user_input, context, json_mode=True)
