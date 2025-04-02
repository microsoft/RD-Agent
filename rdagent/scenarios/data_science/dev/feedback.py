import json
from typing import Dict

import pandas as pd

from rdagent.core.proposal import (
    Experiment2Feedback,
    ExperimentFeedback,
    HypothesisFeedback,
)
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T
from rdagent.utils.repo.diff import generate_diff_from_dict


class DSExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: DSExperiment, trace: DSTrace) -> ExperimentFeedback:
        # 用哪些信息来生成feedback
        # 1. pending_tasks_list[0][0] 任务的描述
        # 2. hypothesis 任务的假设
        # 3. 相对sota_exp的改动
        # 4. result 任务的结果
        # 5. sota_exp.result 之前最好的结果
        sota_exp = trace.sota_experiment()
        sota_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp, heading="SOTA of previous exploration of the scenario"
        )

        # Get feedback description using shared template
        feedback_desc = T("scenarios.data_science.share:describe.feedback").r(
            exp_and_feedback=(trace.hist[-1] if trace.hist else None), heading="Previous Trial Feedback"
        )

        # TODO:
        # -  Should we choose between the diff from last experiment or last sota ?

        # Retrieve the last experiment from the history
        last_exp = trace.hist[-1][0] if trace.hist else None
        if last_exp and last_exp.experiment_workspace and exp.experiment_workspace:
            # Generate a diff between the two workspaces
            last_exp_files = last_exp.experiment_workspace.file_dict
            current_exp_files = exp.experiment_workspace.file_dict
            diff_edition = generate_diff_from_dict(last_exp_files, current_exp_files)
        else:
            diff_edition = []

        # assumption:
        # The feedback should focus on experiment **improving**.
        # Assume that all the the sota exp is based on the previous sota experiment
        cur_vs_sota_score = None
        if sota_exp:
            cur_score = pd.DataFrame(exp.result).loc["ensemble"].iloc[0]
            sota_score = pd.DataFrame(sota_exp.result).loc["ensemble"].iloc[0]
            cur_vs_sota_score = (
                f"The current score is {cur_score}, while the SOTA score is {sota_score}. "
                f"{'In this competition, higher is better.' if self.scen.metric_direction else 'In this competition, lower is better.'}"
            )

        system_prompt = T(".prompts:exp_feedback.system").r(scenario=self.scen.get_scenario_all_desc())
        user_prompt = T(".prompts:exp_feedback.user").r(
            sota_desc=sota_desc,
            cur_exp=exp,
            diff_edition=diff_edition,
            feedback_desc=feedback_desc,
            cur_vs_sota_score=cur_vs_sota_score,
        )

        resp_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_target_type=Dict[str, str | bool | int],
            )
        )

        # Currently, we do not use `observations`, `hypothesis_evaluation`, and `new_hypothesis` in the framework.
        # `new_hypothesis` should not exist in the feedback.
        return HypothesisFeedback(
            observations=resp_dict.get("Observations", "No observations provided"),
            hypothesis_evaluation=resp_dict.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=resp_dict.get("New Hypothesis", "No new hypothesis provided"),
            reason=resp_dict.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(resp_dict.get("Replace Best Result", "no")),
        )
