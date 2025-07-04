import json
from typing import Dict

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import (
    Experiment2Feedback,
    ExperimentFeedback,
    HypothesisFeedback,
)
from rdagent.log.utils import dict_get_with_warning
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.idea_pool import DSIdea
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
            exp_and_feedback=trace.last_exp_fb(), heading="Previous Trial Feedback"
        )

        # TODO:
        # -  Should we choose between the diff from last experiment or last sota ?

        # Retrieve the last experiment from the history
        if sota_exp and sota_exp.experiment_workspace and exp.experiment_workspace:
            # Generate a diff between the two workspaces
            sota_exp_files = sota_exp.experiment_workspace.file_dict
            current_exp_files = exp.experiment_workspace.file_dict
            diff_edition = generate_diff_from_dict(sota_exp_files, current_exp_files)
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
        if DS_RD_SETTING.rule_base_eval:
            if sota_exp:
                if cur_score > sota_score:
                    return HypothesisFeedback(
                        observations="The current score bigger than the SOTA score.",
                        hypothesis_evaluation="The current score is bigger than the SOTA score.",
                        new_hypothesis="No new hypothesis provided",
                        reason="The current score is bigger than the SOTA score.",
                        decision=True if self.scen.metric_direction else False,
                    )
                elif cur_score < sota_score:
                    return HypothesisFeedback(
                        observations="The current score smaller than the SOTA score.",
                        hypothesis_evaluation="The current score is smaller than the SOTA score.",
                        new_hypothesis="No new hypothesis provided",
                        reason="The current score is smaller than the SOTA score.",
                        decision=False if self.scen.metric_direction else True,
                    )
                else:
                    return HypothesisFeedback(
                        observations="The current score equals to the SOTA score.",
                        hypothesis_evaluation="The current score equals to the SOTA score.",
                        new_hypothesis="No new hypothesis provided",
                        reason="The current score equals to the SOTA score.",
                        decision=False,
                    )

        eda_output = exp.experiment_workspace.file_dict.get("EDA.md", None)
        system_prompt = T(".prompts:exp_feedback.system").r(
            scenario=self.scen.get_scenario_all_desc(eda_output=eda_output)
        )
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

        if evaluation_not_aligned := dict_get_with_warning(resp_dict, "Evaluation Aligned With Task", "no") == "no":
            exp.result = None

        # Currently, we do not use `observations`, `hypothesis_evaluation`, and `new_hypothesis` in the framework.
        # `new_hypothesis` should not exist in the feedback.
        hypothesis_feedback = HypothesisFeedback(
            observations=dict_get_with_warning(resp_dict, "Observations", "No observations provided"),
            hypothesis_evaluation=dict_get_with_warning(resp_dict, "Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=dict_get_with_warning(resp_dict, "New Hypothesis", "No new hypothesis provided"),
            reason=dict_get_with_warning(resp_dict, "Reasoning", "No reasoning provided")
            + ("\nRejected because evaluation code not aligned with task." if evaluation_not_aligned else ""),
            code_change_summary=dict_get_with_warning(
                resp_dict, "Code Change Summary", "No code change summary provided"
            ),
            decision=(
                False
                if evaluation_not_aligned
                else convert2bool(dict_get_with_warning(resp_dict, "Replace Best Result", "no"))
            ),
        )

        if hypothesis_feedback and DS_RD_SETTING.enable_knowledge_base:
            ds_idea = DSIdea(
                {
                    "competition": self.scen.get_competition_full_desc(),
                    "idea": exp.hypothesis.hypothesis,
                    "method": exp.pending_tasks_list[0][0].get_task_information(),
                    "hypothesis": {exp.hypothesis.problem_label: exp.hypothesis.problem_desc},
                }
            )
            trace.knowledge_base.add_idea(idea=ds_idea)

        return hypothesis_feedback
