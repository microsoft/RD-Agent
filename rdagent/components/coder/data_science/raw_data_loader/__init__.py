"""

Loop should not large change exclude
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
import re
from pathlib import Path
from typing import Dict

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiEvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.components.coder.data_science.conf import (
    DSCoderCoSTEERSettings,
    get_ds_env,
)
from rdagent.components.coder.data_science.raw_data_loader.eval import (
    DataLoaderCoSTEEREvaluator,
)
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent


class DataLoaderMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: DataLoaderTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        # return a workspace with "load_data.py", "spec/load_data.md" inside
        # assign the implemented code to the new workspace.
        competition_info = self.scen.get_scenario_all_desc(eda_output=workspace.file_dict.get("EDA.md", None))
        data_folder_info = self.scen.processed_data_folder_description
        data_loader_task_info = target_task.get_task_information()

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[data_loader_task_info]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[data_loader_task_info]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            [
                knowledge
                for knowledge in queried_former_failed_knowledge[0]
                if knowledge.implementation.file_dict.get("load_data.py") != workspace.file_dict.get("load_data.py")
            ],
            queried_former_failed_knowledge[1],
        )

        # 1. specifications
        # TODO: We may move spec into a separated COSTEER task
        if DS_RD_SETTING.spec_enabled:
            if "spec/data_loader.md" not in workspace.file_dict:  # Only generate the spec once
                system_prompt = T(".prompts:spec.system").r(
                    runtime_environment=self.scen.get_runtime_environment(),
                    task_desc=data_loader_task_info,
                    competition_info=competition_info,
                    folder_spec=data_folder_info,
                )
                data_loader_prompt = T(".prompts:spec.user.data_loader").r(
                    latest_spec=workspace.file_dict.get("spec/data_loader.md")
                )
                feature_prompt = T(".prompts:spec.user.feature").r(
                    latest_spec=workspace.file_dict.get("spec/feature.md")
                )
                model_prompt = T(".prompts:spec.user.model").r(latest_spec=workspace.file_dict.get("spec/model.md"))
                ensemble_prompt = T(".prompts:spec.user.ensemble").r(
                    latest_spec=workspace.file_dict.get("spec/ensemble.md")
                )
                workflow_prompt = T(".prompts:spec.user.workflow").r(
                    latest_spec=workspace.file_dict.get("spec/workflow.md")
                )

                spec_session = APIBackend().build_chat_session(session_system_prompt=system_prompt)

                data_loader_spec = spec_session.build_chat_completion(user_prompt=data_loader_prompt)
                feature_spec = spec_session.build_chat_completion(user_prompt=feature_prompt)
                model_spec = spec_session.build_chat_completion(user_prompt=model_prompt)
                ensemble_spec = spec_session.build_chat_completion(user_prompt=ensemble_prompt)
                workflow_spec = spec_session.build_chat_completion(user_prompt=workflow_prompt)
            else:
                data_loader_spec = workspace.file_dict["spec/data_loader.md"]
                feature_spec = workspace.file_dict["spec/feature.md"]
                model_spec = workspace.file_dict["spec/model.md"]
                ensemble_spec = workspace.file_dict["spec/ensemble.md"]
                workflow_spec = workspace.file_dict["spec/workflow.md"]

        # 2. code
        system_prompt = T(".prompts:data_loader_coder.system").r(
            task_desc=data_loader_task_info,
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            out_spec=PythonAgentOut.get_spec(),
        )
        code_spec = (
            data_loader_spec
            if DS_RD_SETTING.spec_enabled
            else T("scenarios.data_science.share:component_spec.general").r(
                spec=T("scenarios.data_science.share:component_spec.DataLoadSpec").r(),
                test_code=(DIRNAME / "eval_tests" / "data_loader_test.txt").read_text(),
            )
        )
        user_prompt = T(".prompts:data_loader_coder.user").r(
            competition_info=competition_info,
            code_spec=code_spec,
            folder_spec=data_folder_info,
            latest_code=workspace.file_dict.get("load_data.py"),
            latest_code_feedback=prev_task_feedback,
        )

        for _ in range(5):
            data_loader_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            if data_loader_code != workspace.file_dict.get("load_data.py"):
                break
            else:
                user_prompt = user_prompt + "\nPlease avoid generating same code to former code!"
        else:
            raise CoderError("Failed to generate a new data loader code.")

        return (
            {
                "spec/data_loader.md": data_loader_spec,
                "spec/feature.md": feature_spec,
                "spec/model.md": model_spec,
                "spec/ensemble.md": ensemble_spec,
                "spec/workflow.md": workflow_spec,
                "load_data.py": data_loader_code,
            }
            if DS_RD_SETTING.spec_enabled
            else {
                "load_data.py": data_loader_code,
            }
        )

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
            evo.sub_workspace_list[index].inject_files(**code_list[index])
        return evo


class DataLoaderCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = DSCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(
            DataLoaderCoSTEEREvaluator(scen=scen), scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        es = DataLoaderMultiProcessEvolvingStrategy(scen=scen, settings=settings)

        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            max_loop=DS_RD_SETTING.coder_max_loop,
            **kwargs,
        )

    def develop(self, exp):
        new_exp = super().develop(exp)

        env = get_ds_env(
            extra_volumes={
                f"{DS_RD_SETTING.local_data_path}/{self.scen.competition}": T(
                    "scenarios.data_science.share:scen.input_path"
                ).r()
            },
            running_timeout_period=DS_RD_SETTING.full_timeout,
        )

        stdout = new_exp.experiment_workspace.execute(env=env, entry=f"python test/data_loader_test.py")
        match = re.search(r"(.*?)=== Start of EDA part ===(.*)=== End of EDA part ===", stdout, re.DOTALL)
        eda_output = match.groups()[1] if match else None
        if eda_output is not None:
            new_exp.experiment_workspace.inject_files(**{"EDA.md": eda_output})
        else:
            eda_output = "No EDA output."
            new_exp.experiment_workspace.inject_files(**{"EDA.md": eda_output})
        return new_exp
