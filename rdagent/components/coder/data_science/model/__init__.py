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
from rdagent.components.coder.data_science.conf import DSCoderCoSTEERSettings
from rdagent.components.coder.data_science.model.eval import (
    ModelGeneralCaseSpecEvaluator,
)
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonBatchEditOut
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent


class ModelMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: ModelTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        model_information_str = target_task.get_task_information()

        # 1. query
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[model_information_str]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[model_information_str]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            [
                knowledge
                for knowledge in queried_former_failed_knowledge[0]
                if knowledge.implementation.file_dict.get(f"{target_task.name}.py")
                != workspace.file_dict.get(f"{target_task.name}.py")
            ],
            queried_former_failed_knowledge[1],
        )

        # 2. code
        system_prompt = T(".prompts:model_coder.system").r(
            task_desc=model_information_str,
            competition_info=self.scen.get_scenario_all_desc(eda_output=workspace.file_dict.get("EDA.md", None)),
            data_loader_code=workspace.file_dict.get("load_data.py"),
            feature_code=workspace.file_dict["feature.py"],
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            out_spec=PythonBatchEditOut.get_spec(),
        )
        # user_prompt = T(".prompts:model_coder.user").r(
        #     model_spec=workspace.file_dict["spec/model.md"],
        #     feature_code=workspace.file_dict["feature.py"],
        #     latest_code=workspace.file_dict.get(f"{target_task.name}.py", None),
        # )
        # We want to use a simpler way to
        code_spec = (
            workspace.file_dict["spec/model.md"]
            if DS_RD_SETTING.spec_enabled
            else T("scenarios.data_science.share:component_spec.general").r(
                spec=T("scenarios.data_science.share:component_spec.Model").r(),
                test_code=(DIRNAME / "eval_tests" / "model_test.txt").read_text().replace("model01", target_task.name),
            )
        )
        user_prompt = T(".prompts:model_coder.user_general").r(
            code_spec=code_spec,
            latest_model_code=workspace.get_codes(
                r"^model_(?!test)\w+\.py$"
            ),  # TODO: If we have high failure rate here, we should clean this step with less information.
            latest_code_feedback=prev_task_feedback,
        )

        for _ in range(5):
            batch_edit = PythonBatchEditOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )

            if not all(i.startswith("model_") for i in batch_edit.keys()):
                user_prompt += "\nYou should only update model codes!"
                continue

            # 3. post process to align file name to the task name
            # we assumpt batch_edit only contains one model file update.
            batch_edit = {
                (f"{target_task.name}.py" if value != "__DEL__" and key != f"{target_task.name}.py" else key): value
                for key, value in batch_edit.items()
            }

            user_prompt = user_prompt + "\nPlease avoid generating same code to former code!"
            # TODO: besides same code problem, we should also consider other problems lead to retry.
            if f"{target_task.name}.py" not in batch_edit:
                continue

            if batch_edit and max(len(i.encode("utf-8")) for i in batch_edit.keys()) > 255:
                continue

            if batch_edit[f"{target_task.name}.py"] != "__DEL__" and batch_edit[
                f"{target_task.name}.py"
            ] != workspace.file_dict.get(f"{target_task.name}.py"):
                break

            # If the task involves model removal, assume it can only process one model at a time.
            if len(batch_edit) == 1 and batch_edit[f"{target_task.name}.py"] == "__DEL__":
                break
        else:
            raise CoderError("Failed to generate a new model code.")

        return batch_edit

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


class ModelCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = DSCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(
            ModelGeneralCaseSpecEvaluator(scen=scen), scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        # eva = ModelGeneralCaseSpecEvaluator(scen=scen)
        es = ModelMultiProcessEvolvingStrategy(scen=scen, settings=settings)

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
