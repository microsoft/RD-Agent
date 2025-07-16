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

from pathlib import Path

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
from rdagent.components.coder.data_science.pipeline.eval import PipelineCoSTEEREvaluator
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent


class PipelineMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: DataLoaderTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        competition_info = self.scen.get_scenario_all_desc(eda_output=workspace.file_dict.get("EDA.md", None))
        runtime_environment = self.scen.get_runtime_environment()
        data_folder_info = self.scen.processed_data_folder_description
        pipeline_task_info = target_task.get_task_information()

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[pipeline_task_info]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[pipeline_task_info] if queried_knowledge is not None else []
        )
        queried_former_failed_knowledge = (
            [
                knowledge
                for knowledge in queried_former_failed_knowledge[0]
                if knowledge.implementation.file_dict.get("main.py") != workspace.file_dict.get("main.py")
            ],
            queried_former_failed_knowledge[1],
        )

        system_prompt = T(".prompts:pipeline_coder.system").r(
            task_desc=pipeline_task_info,
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            out_spec=PythonAgentOut.get_spec(),
            runtime_environment=runtime_environment,
            enable_model_dump=DS_RD_SETTING.enable_model_dump,
            enable_debug_mode=DS_RD_SETTING.sample_data_by_LLM,
        )
        user_prompt = T(".prompts:pipeline_coder.user").r(
            competition_info=competition_info,
            folder_spec=data_folder_info,
            latest_code=workspace.file_dict.get("main.py"),
            latest_code_feedback=prev_task_feedback,
        )

        for _ in range(5):
            pipeline_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            if pipeline_code != workspace.file_dict.get("main.py"):
                break
            else:
                user_prompt = user_prompt + "\nPlease avoid generating same code to former code!"
        else:
            raise CoderError("Failed to generate a new pipeline code.")

        return {
            "main.py": pipeline_code,
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
            evo.sub_workspace_list[index].inject_files(**code_list[index])
        return evo


class PipelineCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = DSCoderCoSTEERSettings()
        eval_l = [PipelineCoSTEEREvaluator(scen=scen)]
        if DS_RD_SETTING.enable_model_dump:
            eval_l.append(ModelDumpEvaluator(scen=scen, data_type="sample"))

        eva = CoSTEERMultiEvaluator(
            single_evaluator=eval_l, scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        es = PipelineMultiProcessEvolvingStrategy(scen=scen, settings=settings)

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
