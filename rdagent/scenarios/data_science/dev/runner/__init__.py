from pathlib import Path

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder import CoSTEER
from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.CoSTEER.evolvable_subjects import FBWorkspace
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    CoSTEERQueriedKnowledge,
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.exception import RunnerError
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.dev.runner.eval import DSCoSTEERCoSTEEREvaluator
from rdagent.utils.agent.ret import BatchEditOut
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DockerEnv, MLEBDockerConf


class DSRunnerMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: CoSTEERTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
    ) -> dict[str, str]:
        if workspace.feedback is None:
            return {}

        task_information_str = target_task.get_task_information()
        # 1. code
        system_prompt = T(".prompts:DSCoSTEER_debugger.system").r(
            task_desc=task_information_str,
            out_spec=BatchEditOut.get_spec(with_del=False),
        )
        user_prompt = T(".prompts:DSCoSTEER_debugger.user").r(
            code=workspace.all_codes,
            feedback=workspace.feedback,
        )

        batch_edit = BatchEditOut.extract_output(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=BatchEditOut.json_mode,
            )
        )

        batch_edit = {k: v for k, v in batch_edit.items() if k in workspace.file_dict.keys()}

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


class DSCoSTEERRunner(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        eva = CoSTEERMultiEvaluator(
            DSCoSTEERCoSTEEREvaluator(scen=scen), scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        es = DSRunnerMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)

        super().__init__(*args, settings=CoSTEER_SETTINGS, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)

    def develop(self, exp):
        bak_sub_tasks = exp.sub_tasks
        exp.sub_tasks = [
            CoSTEERTask(
                name="Debug running solution",
                description=f"The whole workflow of the solution has finished with some execution error, please check the error message and debug the whole code repo.\nCurrent code repo md5: {md5_hash(exp.experiment_workspace.all_codes)}",
            )
        ]
        exp = super().develop(exp)
        exp.sub_tasks = bak_sub_tasks

        score_fp = exp.experiment_workspace.workspace_path / "scores.csv"
        if not score_fp.exists():
            logger.error("Metrics file (scores.csv) is not generated.")
            raise RunnerError(f"Metrics file (scores.csv) is not generated")
        exp.result = pd.read_csv(score_fp, index_col=0)

        # DockerEnv for MLEBench submission validation
        mle_de_conf = MLEBDockerConf()
        mle_de_conf.extra_volumes = {
            f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data",
        }
        mde = DockerEnv(conf=mle_de_conf)
        mde.prepare()
        # MLEBench Check
        mle_check_code = (
            (Path(__file__).absolute().resolve().parent / "eval_tests" / "mle_submission_format_test.txt")
            .read_text()
            .replace("<competition_id>", self.scen.competition)
        )
        exp.experiment_workspace.inject_files(**{"mle_submission_format_test.py": mle_check_code})
        exp.format_check_result = exp.experiment_workspace.execute(
            env=mde, entry=f"python mle_submission_format_test.py"
        )

        return exp
