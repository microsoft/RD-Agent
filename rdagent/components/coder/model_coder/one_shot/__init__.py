import re
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelImplementation,
)
from rdagent.core.prompts import Prompts
from rdagent.core.task_generator import TaskGenerator
from rdagent.oai.llm_utils import APIBackend

DIRNAME = Path(__file__).absolute().resolve().parent


class ModelCodeWriter(TaskGenerator[ModelExperiment]):
    def generate(self, exp: ModelExperiment) -> ModelExperiment:
        mti_l = []
        for t in exp.sub_tasks:
            mti = ModelImplementation(t)
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
        exp.sub_implementations = mti_l
        return exp
