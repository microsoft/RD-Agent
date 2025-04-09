import json
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
from rdagent.components.coder.data_science.conf import DSCoderCoSTEERSettings
from rdagent.components.coder.data_science.workflow.eval import (
    WorkflowGeneralCaseSpecEvaluator,
)
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T


class WorkflowMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: WorkflowTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        workflow_information_str = target_task.get_task_information()

        # 1. query
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[workflow_information_str]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[workflow_information_str]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            [
                knowledge
                for knowledge in queried_former_failed_knowledge[0]
                if knowledge.implementation.file_dict.get("main.py") != workspace.file_dict.get("main.py")
            ],
            queried_former_failed_knowledge[1],
        )

        # 2. code
        system_prompt = T(".prompts:workflow_coder.system").r(
            task_desc=workflow_information_str,
            competition_info=self.scen.get_scenario_all_desc(eda_output=workspace.file_dict.get("EDA.md", None)),
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            out_spec=PythonAgentOut.get_spec(),
        )
        user_prompt = T(".prompts:workflow_coder.user").r(
            load_data_code=workspace.file_dict["load_data.py"],
            feature_code=workspace.file_dict["feature.py"],
            model_codes=workspace.get_codes(r"^model_(?!test)\w+\.py$"),
            ensemble_code=workspace.file_dict["ensemble.py"],
            latest_code=workspace.file_dict.get("main.py"),
            code_spec=(
                workspace.file_dict["spec/workflow.md"]
                if DS_RD_SETTING.spec_enabled
                else T("scenarios.data_science.share:component_spec.Workflow").r()
            ),
            latest_code_feedback=prev_task_feedback,
        )

        for _ in range(5):
            workflow_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            if workflow_code != workspace.file_dict.get("main.py"):
                break
            else:
                user_prompt = user_prompt + "\nPlease avoid generating same code to former code!"
        else:
            raise CoderError("Failed to generate a new workflow code.")

        return {"main.py": workflow_code}

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


class WorkflowCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = DSCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(
            WorkflowGeneralCaseSpecEvaluator(scen=scen), scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        es = WorkflowMultiProcessEvolvingStrategy(scen=scen, settings=settings)
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
