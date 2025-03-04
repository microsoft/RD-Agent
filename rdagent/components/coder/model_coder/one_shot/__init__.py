import re
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelFBWorkspace
from rdagent.core.developer import Developer
from rdagent.core.prompts import Prompts
from rdagent.oai.llm_utils import APIBackend
from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger

DIRNAME = Path(__file__).absolute().resolve().parent


class ModelCodeWriter(Developer[ModelExperiment]):
    def develop(self, exp: ModelExperiment) -> ModelExperiment:
        mti_l = []
        for t in exp.sub_tasks:
            mti = ModelFBWorkspace(t)
            mti.prepare()
            pr = Prompts(file_path=DIRNAME / "prompt.yaml")

            user_prompt_tpl = Environment(undefined=StrictUndefined).from_string(pr["code_implement_user"])
            sys_prompt_tpl = Environment(undefined=StrictUndefined).from_string(pr["code_implement_sys"])
            user_prompt = user_prompt_tpl.render(
                name=t.name,
                description=t.description,
                formulation=t.formulation,
                variables=t.variables,
                type=t.model_type,
            )
            system_prompt = sys_prompt_tpl.render()
            with open(DIRNAME / "prompt_generated.txt", "w") as f:
                f.write(system_prompt)
                f.write("\n")
                f.write(user_prompt)
                f.write("\n")
            resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt)
            from rdagent.utils.agent.ret import PythonAgentOut
            code = PythonAgentOut().extract_output(resp)        
            logger.info(f"{LogColors.BOLD}extracted code(next row):\n{code}{LogColors.END}", tag="debug_model_code")
            mti.inject_files(**{"model.py": code})
            mti_l.append(mti)
        exp.sub_workspace_list = mti_l
        return exp
