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
from rdagent.components.coder.data_science.pipeline.eval import PipelineCoSTEEREvaluator, PipelineCoSTEEREvaluatorV3
from rdagent.components.coder.data_science.pipeline.apply_patch import process_patch, DiffError
from rdagent.components.coder.data_science.raw_data_loader.eval import (
    DataLoaderCoSTEEREvaluator,
)
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
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
            spec=T("scenarios.data_science.share:component_spec.Pipeline").r(),
            enable_model_dump=DS_RD_SETTING.enable_model_dump,
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


class PipelineMultiProcessEvolvingStrategyV3(PipelineMultiProcessEvolvingStrategy):

    def apply_to_main_py(self, main_py: str, patch: str) -> str:
        """Apply patches to main.py"""
        def open_file(file_path: str) -> str:
            return main_py
        def write_file(file_path: str, content: str) -> None:
            nonlocal main_py
            main_py = content
        def remove_file(file_path: str) -> None:
            pass

        def bash_patch_to_patch(patch: str) -> str:
            start_patch_marker = "*** Begin Patch"
            start_index = patch.find(start_patch_marker)

            if start_index != -1:
                # Find the starting index of "EOF" after "*** Begin Patch"
                eof_marker = "EOF"
                # We search for EOF starting from the position *after* the "apply_patch <<\"EOF\"" line
                # to ensure we get the correct closing EOF.
                search_after_heredoc_eof = patch.find(eof_marker) + len(eof_marker)
                end_index = patch.find(eof_marker, search_after_heredoc_eof)

                if end_index != -1:
                    # Slice the patch including "*** Begin Patch" and excluding "EOF"
                    sliced_segment = patch[start_index:end_index]
                    return sliced_segment
                else:
                    logger.error("EOF marker not found after '*** Begin Patch'.")
                    return patch
            else:
                logger.error("'*** Begin Patch' marker not found.")
                return patch

        try:
            process_patch(
                bash_patch_to_patch(patch),
                open_file,
                write_file,
                remove_file,
            )
            return main_py
        except DiffError as exc:
            logger.error(f"Apply patch error: {exc}")
            return main_py

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

        main_py: str | None = workspace.file_dict.get("main.py")

        system_prompt_prefix = T(".prompts_v3:pipeline_coder.system_prefix").r(
            runtime_environment=runtime_environment,
            # out_spec=PythonAgentOut.get_spec(),
            # spec=T("scenarios.data_science.share:component_spec.Pipeline").r(),
            # enable_model_dump=DS_RD_SETTING.enable_model_dump,
        )
        user_shared = T(".prompts_v3:pipeline_coder.user_shared").r(
            competition_info=competition_info,
            folder_spec=data_folder_info,
            task=target_task,
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
        )
        if not main_py:
            system_prompt = T(".prompts_v3:pipeline_coder.system_newcode").r(
                prefix=system_prompt_prefix
            )
            user_prompt = T(".prompts_v3:pipeline_coder.user_newcode").r(
                shared=user_shared,
                latest_code=main_py,
                latest_code_feedback=prev_task_feedback,
            )

        else:
            system_prompt = T(".prompts_v3:pipeline_coder.system_patch").r(
                prefix=system_prompt_prefix
            )
            user_prompt = T(".prompts_v3:pipeline_coder.user_patch").r(
                shared=user_shared,
                latest_code=main_py,
                latest_code_feedback=prev_task_feedback,
            )

        APPLY_PATCH_TOOL = {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": T(".prompts_v3:pipeline_coder.apply_patch_tool_desc").r(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The apply_patch command that you wish to execute.",
                        }
                    },
                    "required": ["input"],
                },
            }
        }

        for _ in range(5):
            tool_kwargs = {}
            if main_py:
                tool_kwargs = {
                    "tools": [APPLY_PATCH_TOOL],
                    "tool_choice": "auto",
                }

            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                **tool_kwargs
            )
            if not isinstance(response, str) and response.tool_calls and main_py:
                tool_calls = response.tool_calls
                patch = json.loads(tool_calls[0].function.arguments)["input"]
                logger.info(f"Tool call: {tool_calls[0].function.name} with patch:\n{patch}")
                if main_py:
                    main_py = self.apply_to_main_py(main_py, patch)
            elif not isinstance(response, str):
                main_py = PythonAgentOut.extract_output(response.message)
            else:
                main_py = PythonAgentOut.extract_output(response)

            if main_py != workspace.file_dict.get("main.py"):
                break
            else:
                raise CoderError("The generated code is the same as the previous one.")
                user_prompt = user_prompt + "\nPlease avoid generating same code to former code!"
        else:
            raise CoderError("Failed to generate a new pipeline code.")

        return {
            "main.py": main_py,
        }


class PipelineCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = DSCoderCoSTEERSettings()
        if settings.prompt_version == "v1":
            eval_l = [PipelineCoSTEEREvaluator(scen=scen)]
        else:
            eval_l = [PipelineCoSTEEREvaluatorV3(scen=scen)]
        if DS_RD_SETTING.enable_model_dump:
            eval_l.append(ModelDumpEvaluator(scen=scen, data_type="sample"))

        eva = CoSTEERMultiEvaluator(
            single_evaluator=eval_l, scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        if settings.prompt_version == "v1":
            es = PipelineMultiProcessEvolvingStrategy(scen=scen, settings=settings)
        else:
            es = PipelineMultiProcessEvolvingStrategyV3(scen=scen, settings=settings)

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
