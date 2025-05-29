"""
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

from jinja2 import Environment, StrictUndefined

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
from rdagent.components.coder.data_science.ensemble.eval import EnsembleCoSTEEREvaluator
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent


class EnsembleMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: EnsembleTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        # Get task information for knowledge querying
        ensemble_information_str = target_task.get_task_information()

        # Query knowledge
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[ensemble_information_str]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[ensemble_information_str]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            [
                knowledge
                for knowledge in queried_former_failed_knowledge[0]
                if knowledge.implementation.file_dict.get("ensemble.py") != workspace.file_dict.get("ensemble.py")
            ],
            queried_former_failed_knowledge[1],
        )

        # Generate code with knowledge integration
        competition_info = self.scen.get_scenario_all_desc(eda_output=workspace.file_dict.get("EDA.md", None))
        system_prompt = T(".prompts:ensemble_coder.system").r(
            task_desc=ensemble_information_str,
            competition_info=competition_info,
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
            queried_former_failed_knowledge=(
                queried_former_failed_knowledge[0] if queried_former_failed_knowledge else None
            ),
            all_code=workspace.all_codes,
            out_spec=PythonAgentOut.get_spec(),
        )

        if DS_RD_SETTING.spec_enabled:
            code_spec = workspace.file_dict["spec/ensemble.md"]
        else:
            test_code = (
                Environment(undefined=StrictUndefined)
                .from_string((DIRNAME / "eval_tests" / "ensemble_test.txt").read_text())
                .render(
                    model_names=[
                        fn[:-3] for fn in workspace.file_dict.keys() if fn.startswith("model_") and "test" not in fn
                    ],
                    metric_name=self.scen.metric_name,
                )
            )
            code_spec = T("scenarios.data_science.share:component_spec.general").r(
                spec=T("scenarios.data_science.share:component_spec.Ensemble").r(), test_code=test_code
            )
        user_prompt = T(".prompts:ensemble_coder.user").r(
            code_spec=code_spec,
            latest_code=workspace.file_dict.get("ensemble.py"),
            latest_code_feedback=prev_task_feedback,
        )

        for _ in range(5):
            ensemble_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            if ensemble_code != workspace.file_dict.get("ensemble.py"):
                break
            else:
                user_prompt = user_prompt + "\nPlease avoid generating same code to former code!"
        else:
            raise CoderError("Failed to generate a new ensemble code.")

        return {
            "ensemble.py": ensemble_code,
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


class EnsembleCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = DSCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(EnsembleCoSTEEREvaluator(scen=scen), scen=scen)
        es = EnsembleMultiProcessEvolvingStrategy(scen=scen, settings=settings)

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
