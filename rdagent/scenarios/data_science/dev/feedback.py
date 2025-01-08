import json
from pathlib import Path

from rdagent.components.knowledge_management.graph import UndirectedNode
from rdagent.core.experiment import Experiment
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, ExperimentFeedback
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T


from typing import List
import difflib
from pathlib import Path

# TODO:  find a better place.
def generate_diff(dir1: str, dir2: str) -> List[str]:
    """
    Generate a diff between two directories, considering only .py files.
    It is mocking `diff -durN dir1 dir2` in linux.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.

    Returns:
        List[str]: A list of diffs for .py files that are different between the two directories.
    """

    diff_files = []

    dir1_files = {f.relative_to(dir1) for f in Path(dir1).rglob('*.py') if f.is_file()}
    dir2_files = {f.relative_to(dir2) for f in Path(dir2).rglob('*.py') if f.is_file()}

    all_files = dir1_files.union(dir2_files)

    for file in all_files:
        file1 = Path(dir1) / file
        file2 = Path(dir2) / file

        if file1.exists() and file2.exists():
            with file1.open() as f1, file2.open() as f2:
                diff = list(difflib.unified_diff(
                    f1.readlines(),
                    f2.readlines(),
                    fromfile=str(file1),
                    tofile=str(file2)
                ))
                if diff:
                    diff_files.extend(diff)
        else:
            if file1.exists():
                with file1.open() as f1:
                    diff = list(difflib.unified_diff(
                        f1.readlines(),
                        [],
                        fromfile=str(file1),
                        tofile=str(file2) + " (empty file)"
                    ))
                    diff_files.extend(diff)
            elif file2.exists():
                with file2.open() as f2:
                    diff = list(difflib.unified_diff(
                        [],
                        f2.readlines(),
                        fromfile=str(file1) + " (empty file)",
                        tofile=str(file2)
                    ))
                    diff_files.extend(diff)

    return diff_files

class DSExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: DSExperiment, trace: DSTrace) -> ExperimentFeedback:
        # 用哪些信息来生成feedback
        # 1. sub_tasks[0] 任务的描述
        # 2. hypothesis 任务的假设
        # 3. 相对sota_exp的改动
        # 4. result 任务的结果
        # 5. sota_exp.result 之前最好的结果
        sota_exp = trace.sota_experiment()
        sota_desc = T("scenarios.data_science.share:describe.exp").r(exp=sota_exp, heading="SOTA of previous exploration of the scenario")

        # Get feedback description using shared template
        feedback_desc = T("scenarios.data_science.share:describe.feedback").r(
            exp_and_feedback=(trace.hist[-1] if trace.hist else None),
            heading="Previous Trial Feedback"
        )

        # TODO:
        # -  Should we choose between the diff from last experiment or last sota ?

        # Retrieve the last experiment from the history
        last_exp = trace.hist[-1][0] if trace.hist else None
        if last_exp:
            last_workspace_path = last_exp.experiment_workspace.workspace_path
            current_workspace_path = exp.experiment_workspace.workspace_path
            # Generate a diff between the two workspaces
            diff_edition = generate_diff(last_workspace_path, current_workspace_path)
        else:
            diff_edition = []

        # assumption:
        # The feedback should focus on experiment **improving**.
        # Assume that all the the sota exp is based on the previous sota experiment

        system_prompt = T(".prompts:exp_feedback.system").r(scenario=self.scen.get_scenario_all_desc())
        user_prompt = T(".prompts:exp_feedback.user").r(
            sota_desc=sota_desc,
            cur_exp=exp,
            diff_edition=diff_edition,
            feedback_desc=feedback_desc,
        )

        resp_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
        )

        return HypothesisFeedback(
            observations=resp_dict.get("Observations", "No observations provided"),
            hypothesis_evaluation=resp_dict.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=resp_dict.get("New Hypothesis", "No new hypothesis provided"),
            reason=resp_dict.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(resp_dict.get("Replace Best Result", "no")),
        )
