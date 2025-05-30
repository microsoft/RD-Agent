import json
from typing import Dict

import pandas as pd
from pydantic import BaseModel, Field

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import (
    Experiment2Feedback,
    ExperimentFeedback,
    HypothesisFeedback,
)
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.idea_pool import DSIdea
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T
from rdagent.utils.repo.diff import generate_diff_from_dict


class AspectDecision(BaseModel):
    reasoning: str = Field(
        description="A concise 3-5 sentence (unless otherwise specified) explanation supporting the decision for this aspect. Reference specific data or code elements where applicable."
    )
    decision: bool = Field(
        description="Boolean flag indicating if the current experiment passes the check or meets the criteria for this specific aspect."
    )

class ExperimentFeedbackInAspects(BaseModel):
    submission_format_valid: AspectDecision = Field(
        description="Evaluates if the submission format of the current experiment is valid."
    )
    is_first_valid_submission: AspectDecision = Field(
        description="Determines if this is the historically first experiment with a valid submission format. Only evaluate if submission_format_valid.decision is true."
    )
    evaluation_aligned_with_competition: AspectDecision = Field(
        description="Assesses if the current experiment's setup (validation metric, prediction methodology, risk of overfitting/leakage) aligns with competition requirements and best practices. Only evaluate if submission_format_valid.decision is true."
    )
    performance_exceeds_sota: AspectDecision = Field(
        description="Compares the current experiment's ensemble performance against the SOTA ensemble. 'decision' is true if performance is obviously better or if this establishes the first SOTA. 'decision' is false if obviously worse. If similar, 'decision' can be true but reasoning must note similarity, deferring final judgment to code quality analysis for the overall_recommendation. Only evaluate if evaluation_aligned_with_competition.decision is true."
    )
    hypothesis_supported: AspectDecision = Field(
        description="Evaluates whether the experimental results confirm or refute the stated hypothesis for the current experiment, based on data trends. This does not directly determine if the experiment is a 'good' submission but rather if the learning objective was met."
    )
    code_quality_and_robustness_superior_or_establishes_sota: AspectDecision = Field(
        description="Detailed analysis of the current experiment's code quality (less overfitting risk, best practices, interpretability, efficiency) compared to SOTA. This is especially crucial if performance_exceeds_sota.decision is true due to similar performance. If this is the first SOTA, then code quality establishes the baseline. Only evaluate if performance_exceeds_sota.decision is true."
    )
    inherent_risk_assessment: AspectDecision = Field(
        description="Assesses the inherent risks of the current experiment, such as overfitting, evaluation soundness that could undermine the validity of the evaluation results."
    )
    overall_recommendation_to_submit: AspectDecision = Field(
        description="Final recommendation on whether to submit this experiment's results and potentially replace the SOTA. The 'decision' is based on a cascade of the previous checks. The 'reasoning' MUST summarize the primary basis for this decision, starting with a specific tag: [Submission format error], [Evaluation alignment error], [Performance regression], [Quality or robustness inferior], [High inherent risk], [Accepted new SOTA performance], [Accepted code improvement], or [Accepted first SOTA]. Whether the hypothesis was supported by the result should also be included in the reasoning of the overall recommendation, although it's not a direct factor in the decision. The reasoning can be longer than other aspects (e.g., up to 10 sentences), but still concise."
    )


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

        last_exp = trace.last_exp()

        # Get feedback description using shared template
        feedback_desc = T("scenarios.data_science.share:describe.feedback").r(
            exp_and_feedback=trace.hist[-1] if trace.hist else None, heading="Previous Trial Feedback"
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

        if resp_dict.get("Evaluation Aligned With Task", "no") == "no":
            exp.result = None

        # Currently, we do not use `observations`, `hypothesis_evaluation`, and `new_hypothesis` in the framework.
        # `new_hypothesis` should not exist in the feedback.
        hypothesis_feedback = HypothesisFeedback(
            observations=resp_dict.get("Observations", "No observations provided"),
            hypothesis_evaluation=resp_dict.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=resp_dict.get("New Hypothesis", "No new hypothesis provided"),
            reason=resp_dict.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(resp_dict.get("Replace Best Result", "no")),
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


class DSExperiment2FeedbackV3(DSExperiment2Feedback):
    def generate_feedback(self, exp: DSExperiment, trace: DSTrace) -> ExperimentFeedback:
        # 用哪些信息来生成feedback
        # 1. pending_tasks_list[0][0] 任务的描述
        # 2. hypothesis 任务的假设
        # 3. 相对sota_exp的改动
        # 4. result 任务的结果
        # 5. sota_exp.result 之前最好的结果
        sota_exp = trace.sota_experiment()

        last_exp = trace.last_exp()
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
        system_prompt = T(".prompts_v3:exp_feedback.system").r()
        user_prompt = T(".prompts_v3:exp_feedback.user").r(
            scenario_desc=trace.scen.get_scenario_all_desc(eda_output=eda_output),
            sota_exp=sota_exp,
            cur_exp=exp,
            exp_and_feedback=trace.hist[-1] if trace.hist else None,
            diff_edition=diff_edition,
            cur_vs_sota_score=cur_vs_sota_score,
        )

        resp_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                response_format=ExperimentFeedbackInAspects,
            )
        )

        feedbacks = ExperimentFeedbackInAspects(**resp_dict)
        
        # Translate to hypothesis feedback
        if not feedbacks.evaluation_aligned_with_competition.decision:
            exp.result = None
        hypothesis_feedback = HypothesisFeedback(
            observations=feedbacks.performance_exceeds_sota.reasoning,
            hypothesis_evaluation=feedbacks.hypothesis_supported.reasoning,
            new_hypothesis="No new hypothesis provided",
            reason=feedbacks.overall_recommendation_to_submit.reasoning,
            decision=feedbacks.overall_recommendation_to_submit.decision,
        )

        logger.info(f"Feedback for hypothesis:\n{hypothesis_feedback}")

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
