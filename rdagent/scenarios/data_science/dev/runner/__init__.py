from typing import Literal

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiEvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolvable_subjects import FBWorkspace
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    CoSTEERQueriedKnowledge,
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator
from rdagent.core.exception import RunnerError
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.utils.agent.ret import PythonBatchEditOut, PythonBatchPatchOut
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry


class DSRunnerCoSTEERSettings(CoSTEERSettings):
    """Data Science CoSTEER settings"""

    class Config:
        env_prefix = "DS_Runner_CoSTEER_"

    max_seconds_multiplier: int = 1
    env_type: str = "docker"
    diff_mode: bool = False
    dump_stdout_type: Literal["full", "truncated"] = "truncated"
    # TODO: extract a function for env and conf.


class DSRunnerMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    @wait_retry(retry_n=5)
    def implement_one_task(
        self,
        target_task: CoSTEERTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:

        if prev_task_feedback is None:
            # if no prev_task_feedback, it is the first loop; we do not make any changes and goto evaluators directly.
            return {}

        # Get evolving history
        task_info = target_task.get_task_information()
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[task_info] if queried_knowledge is not None else []
        )[0]

        # Set output agent
        if self.settings.diff_mode:
            output_spec = PythonBatchPatchOut.get_spec()
            extract_output_fn = PythonBatchPatchOut.extract_output
        else:
            output_spec = PythonBatchEditOut.get_spec(with_del=False)
            extract_output_fn = PythonBatchEditOut.extract_output

        if prev_task_feedback.acceptable is False:
            task_information_str = target_task.get_task_information()
            # Use system_debugger for error fixing and debugging
            system_prompt = T(".prompts:DSCoSTEER.system_debugger").r(
                task_desc=task_information_str,
                out_spec=output_spec,
                diff_mode=self.settings.diff_mode,
            )
        else:
            # Use system_refine for hyperparameter tuning
            system_prompt = T(".prompts:DSCoSTEER.system_refine").r(
                out_spec=output_spec,
                diff_mode=self.settings.diff_mode,
            )

        # Start multi-turn chat session
        session = APIBackend().build_chat_session(
            session_system_prompt=system_prompt,
        )

        # Code
        user_prompt = T(".prompts:DSCoSTEER.user").r(
            code=workspace.all_codes,
            change_summary=workspace.change_summary,
            feedback=prev_task_feedback,
            hyperparameter_tuning_suggestion=(
                prev_task_feedback.hyperparameter_tuning_suggestion if prev_task_feedback.acceptable else None
            ),
            queried_former_failed_knowledge=queried_former_failed_knowledge,
        )

        code = session.build_chat_completion(user_prompt=user_prompt)
        if self.settings.diff_mode:
            code_batch_edit = extract_output_fn(code, prefix=workspace.workspace_path)
        else:
            code_batch_edit = extract_output_fn(code)
        code_batch_edit = {k: v for k, v in code_batch_edit.items() if k in workspace.file_dict.keys()}

        if DS_RD_SETTING.runner_enable_code_change_summary:
            # Change Summary
            user_prompt = (
                "Based on the previous conversation and your latest code modifications, "
                "please provide a concise and structured summary of the changes you made to the original code. "
                "Clearly specify what was changed and how, focusing on key modifications. "
                "Limit your summary to plain text, no more than three sentences."
            )
            change_summary = session.build_chat_completion(user_prompt=user_prompt)
            code_batch_edit.update({"__change_summary__": change_summary})

        return code_batch_edit

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
            if self.KEY_CHANGE_SUMMARY in code_list[index]:
                evo.sub_workspace_list[index].change_summary = code_list[index].pop(self.KEY_CHANGE_SUMMARY)
            evo.sub_workspace_list[index].inject_files(**code_list[index])
        return evo


class DSCoSTEERRunner(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:

        from rdagent.scenarios.data_science.dev.runner.eval import (
            DSRunnerEvaluator,  # avoid circular import
        )

        eval_l = [DSRunnerEvaluator(scen=scen)]
        if DS_RD_SETTING.enable_model_dump:
            eval_l.append(ModelDumpEvaluator(scen=scen, data_type="full"))

        eva = CoSTEERMultiEvaluator(
            single_evaluator=eval_l, scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        settings = DSRunnerCoSTEERSettings()
        es = DSRunnerMultiProcessEvolvingStrategy(scen=scen, settings=settings)

        # In runner, we don't need very big loops, so we set max_loop to runner_max_loop
        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            max_loop=DS_RD_SETTING.runner_max_loop,
            **kwargs,
        )

    def get_develop_max_seconds(self) -> int | None:
        """
        The coder uses the scenario's real debug timeout as the maximum seconds for development.
        """
        return int(self.scen.real_full_timeout() * self.settings.max_seconds_multiplier)

    def should_use_new_evo(self, base_fb: CoSTEERMultiFeedback | None, new_fb: CoSTEERMultiFeedback) -> bool:
        if not new_fb.is_acceptable():
            return False

        # In data science, we only have a single feedback.
        # Note: new_fb should always exists as indicated by _get_last_fb() function.
        if base_fb is None:
            return True

        base_fb = base_fb[0]
        new_fb = new_fb[0]

        def compare_scores(s1, s2) -> bool:
            if s2 is None:
                return False
            if s1 is None:
                return True
            return (s2 > s1) == self.scen.metric_direction

        return compare_scores(base_fb.score, new_fb.score)

    def develop(self, exp):
        bak_sub_tasks = exp.pending_tasks_list
        exp.sub_tasks = [
            CoSTEERTask(
                name="Debug running solution",
                description=f"You'll be provided with the source code and the running and testing stdout. "
                "Please check the error messages and debug the source code if any errors occur.\n"
                f"Original task: {bak_sub_tasks[0][0].get_task_information()}\n"
                f"Current code repo md5: {md5_hash(exp.experiment_workspace.all_codes)}",
            ),
        ]
        exp = super().develop(exp)  # run strategy(code implementation & evaluation loops)
        exp.sub_tasks = bak_sub_tasks

        # NOTE: after running the loops, we expect some results are generated
        #
        # 1) scores of the models and ensemble
        score_fp = exp.experiment_workspace.workspace_path / "scores.csv"
        if not score_fp.exists():
            logger.error("Metrics file (scores.csv) is not generated.")
            raise RunnerError(f"Metrics file (scores.csv) is not generated")
        exp.result = pd.read_csv(score_fp, index_col=0)
        exp.running_info.running_time = exp.experiment_workspace.running_info.running_time

        # 2) if mle-bench, then the submission format checking will be used.
        # DockerEnv for MLEBench submission validation
        if DS_RD_SETTING.if_using_mle_data:
            score_fp = exp.experiment_workspace.workspace_path / "test" / "mle_submission_format_test.output"
            with score_fp.open() as f:
                exp.format_check_result = f.read()
        return exp
