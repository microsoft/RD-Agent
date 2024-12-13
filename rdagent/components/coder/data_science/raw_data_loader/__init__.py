"""

Loop should not large change excclude
- Action Choice[current data loader & spec]
- other should share
    - Propose[choice] => Task[Choice] => CoSTEER => 
        - 

Extra feature:
- cache


File structure
- ___init__.py: the entrance/agent of coder
- evaluator.py
- conf.py
- exp.py: everything under the experiment, e.g.
    - Task
    - Experiment
    - Workspace
- test.py
    - Each coder could be tested.
"""

import json

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.components.coder.data_science.raw_data_loader.eval import (
    DataLoaderCoSTEEREvaluator,
)
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class DataLoaderMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: DataLoaderTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
    ) -> dict[str, str]:
        # return a workspace with "load_data.py", "spec/load_data.md" inside
        # assign the implemented code to the new workspace.
        competition_info = self.scen.competition_descriptions

        # 1. specifications
        system_prompt = T(".prompts:spec.system").r(competition_info=competition_info)
        data_loader_prompt = T(".prompts:spec.user.data_loader").r()
        feature_prompt = T(".prompts:spec.user.feature").r()
        model_prompt = T(".prompts:spec.user.model").r()
        ensemble_prompt = T(".prompts:spec.user.ensemble").r()
        workflow_prompt = T(".prompts:spec.user.workflow").r()

        spec_session = APIBackend().build_chat_session(session_system_prompt=system_prompt)

        data_loader_spec = json.loads(
            spec_session.build_chat_completion(user_prompt=data_loader_prompt, json_mode=True)
        )["spec"]
        feature_spec = json.loads(spec_session.build_chat_completion(user_prompt=feature_prompt, json_mode=True))[
            "spec"
        ]
        model_spec = json.loads(spec_session.build_chat_completion(user_prompt=model_prompt, json_mode=True))["spec"]
        ensemble_spec = json.loads(spec_session.build_chat_completion(user_prompt=ensemble_prompt, json_mode=True))[
            "spec"
        ]
        workflow_spec = json.loads(spec_session.build_chat_completion(user_prompt=workflow_prompt, json_mode=True))[
            "spec"
        ]

        # 2. code
        system_prompt = T(".prompts:data_loader_coder.system").r()
        user_prompt = T(".prompts:data_loader_coder.user").r(
            competition_info=competition_info, data_loader_spec=data_loader_spec
        )

        data_loader_code = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True
            )
        )["code"]

        return {
            "spec/data_loader.md": data_loader_spec,
            "spec/feature.md": feature_spec,
            "spec/model.md": model_spec,
            "spec/ensemble.md": ensemble_spec,
            "spec/workflow.md": workflow_spec,
            "load_data.py": data_loader_code,
        }

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """
        Assign the code list to the evolving item.

        The code list is aligned with the evolving item's sub-tasks.
        If a task is not implemented, put a None in the list.
        """
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                # evo.sub_workspace_list[index] = FBWorkspace(target_task=evo.sub_tasks[index])
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_code(**code_list[index])
        return evo


class DataLoaderCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        eva = CoSTEERMultiEvaluator(
            DataLoaderCoSTEEREvaluator(scen=scen), scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        es = DataLoaderMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)

        super().__init__(*args, settings=CoSTEER_SETTINGS, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)
