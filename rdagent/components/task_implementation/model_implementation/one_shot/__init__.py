import re
from pathlib import Path
from typing import Sequence

from jinja2 import Environment, StrictUndefined

from rdagent.components.task_implementation.model_implementation.task import (
    ModelImplTask,
    ModelTaskImpl,
)
from rdagent.core.task_generator import TaskGenerator
from rdagent.core.prompts import Prompts
from rdagent.oai.llm_utils import APIBackend

DIRNAME = Path(__file__).absolute().resolve().parent


class ModelTaskGen(TaskGenerator):
    def generate(self, task_l: Sequence[ModelImplTask]) -> Sequence[ModelTaskImpl]:
        mti_l = []
        for t in task_l:
            mti = ModelTaskImpl(t)
            mti.prepare()
            pr = Prompts(file_path=DIRNAME / "prompt.yaml")

            user_prompt_tpl = Environment(undefined=StrictUndefined).from_string(pr["code_implement_user"])
            sys_prompt_tpl = Environment(undefined=StrictUndefined).from_string(pr["code_implement_sys"])

            user_prompt = user_prompt_tpl.render(
                name=t.name,
                description=t.description,
                formulation=t.formulation,
                variables=t.variables,
                execute_desc=mti.execute_desc(),
            )
            system_prompt = sys_prompt_tpl.render()

            resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt)

            # Extract the code part from the response
            match = re.search(r".*```[Pp]ython\n(.*)\n```.*", resp, re.DOTALL)
            code = match.group(1)
            mti.inject_code(**{"model.py": code})
            mti_l.append(mti)
        return mti_l
