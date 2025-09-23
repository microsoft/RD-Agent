import json
import math
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.experiment import UserInstructions
from rdagent.core.proposal import ExpGen
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RD_Agent_TIMER_wrapper
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.dev.feedback import ExperimentFeedback
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.draft.draft import (
    DSDraftExpGen,  # TODO: DSDraftExpGen should be moved to router in the further
)
from rdagent.scenarios.data_science.proposal.exp_gen.idea_pool import DSIdea
from rdagent.scenarios.data_science.proposal.exp_gen.planner import (
    DSExperimentPlan,
    RD_Agent_TIMER_wrapper,
)
from rdagent.scenarios.data_science.proposal.exp_gen.select.submit import (
    BestValidSelector,
)
from rdagent.scenarios.data_science.proposal.exp_gen.utils import get_packages
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction
from rdagent.utils.agent.tpl import T
from rdagent.utils.repo.diff import generate_diff_from_dict
from rdagent.utils.workflow import wait_retry

_COMPONENT_META: Dict[str, Dict[str, Any]] = {
    "DataLoadSpec": {
        "target_name": "Data loader and specification generation",
        "spec_file": "spec/data_loader.md",
        "output_format_key": ".prompts:output_format.data_loader",
        "task_class": DataLoaderTask,
    },
    "FeatureEng": {
        "target_name": "Feature engineering",
        "spec_file": "spec/feature.md",
        "output_format_key": ".prompts:output_format.feature",
        "task_class": FeatureTask,
    },
    "Model": {
        "target_name": "Model",
        "spec_file": "spec/model.md",
        "output_format_key": ".prompts:output_format.model",
        "task_class": ModelTask,
    },
    "Ensemble": {
        "target_name": "Ensemble",
        "spec_file": "spec/ensemble.md",
        "output_format_key": ".prompts:output_format.ensemble",
        "task_class": EnsembleTask,
    },
    "Workflow": {
        "target_name": "Workflow",
        "spec_file": "spec/workflow.md",
        "output_format_key": ".prompts:output_format.workflow",
        "task_class": WorkflowTask,
    },
    "Pipeline": {
        "target_name": "Pipeline",
        "spec_file": None,
        "output_format_key": ".prompts:output_format.pipeline",
        "task_class": PipelineTask,
    },
}


def get_component(name: str) -> Dict[str, Any]:
    meta = _COMPONENT_META.get(name)
    if meta is None:
        raise KeyError(f"Unknown component: {name!r}")

    return {
        "target_name": meta["target_name"],
        "spec_file": meta["spec_file"],
        "task_output_format": T(meta["output_format_key"]).r(),
        "task_class": meta["task_class"],
    }


class ScenarioChallengeCategory(str, Enum):
    DATASET_DRIVEN = "dataset-driven"
    DOMAIN_INFORMED = "domain-informed"


class ScenarioChallengeDetail(BaseModel):
    reasoning: str = Field(
        description=(
            "Explanation (max 3 sentences) of how the Core Analysis Dimensions "
            "(SOTA Alignment Analysis, Gap Identification, Domain-Implementation Coherence Check, Scenario-First Focus) "
            "specifically led to identifying THIS challenge."
        )
    )
    category: ScenarioChallengeCategory = Field(description="The category of the improvement challenge.")
    statement: str = Field(
        description="Description of the challenge in no more than three sentences, outlining the specific area for improvement."
    )
    metric_impact: str = Field(
        description="Brief explanation in no more than two sentences of why addressing this challenge is expected to improve the target metric."
    )
    caption: str = Field(description="Summarize the challenge in around 5-15 words.")


class ScenarioAnalysis(BaseModel):
    sota_alignment_analysis: str = Field(description="Comparing SOTA to data/domain insights; 'N/A' if not available.")
    gap_identification: str = Field(
        description="Unaddressed challenges or workarounds in successful solutions; 'N/A' if none."
    )
    domain_implementation_coherence_check: str = Field(
        description="Technical methods conflicting with domain rules or oversimplifying; 'N/A' if none."
    )
    scenario_first_focus: str = Field(
        description="Foundational scenario strategies, key if no SOTA exists; 'N/A' if SOTA already exists."
    )


class ScenarioChallenges(BaseModel):

    analysis: ScenarioAnalysis = Field(
        description="Analysis of provided information following the Core Analysis Dimensions."
    )
    challenges: List[ScenarioChallengeDetail] = Field(
        description='At most five challenges, prioritizing "FEWER BUT BETTER": '
        "select the most valuable and potentially unexplored avenues. Each challenge must be tightly relevant to the improvement of the target metric."
    )


class TraceAnalysisDetail(BaseModel):

    category: str = Field(
        description="Describe the specific area of this analysis in a few words, such as 'Explicit Suggestions', 'Feature Engineering', 'Presistent Issues'"
    )
    statement: str = Field(
        description="Description of the analysis in no more than three sentences, outlining the specific problem."
    )


class TraceAnalysis(BaseModel):

    feedback: List[TraceAnalysisDetail] = Field(
        description="Analysis points derived from feedback on previous experiments."
    )
    implementation_review: List[TraceAnalysisDetail] = Field(
        description="Analysis points from reviewing previous code implementations."
    )
    trace_history: List[TraceAnalysisDetail] = Field(
        description="Analysis points identified from the history of experiment traces."
    )


class TraceChallengeDetail(BaseModel):
    reasoning: str = Field(
        description=(
            "Explanation (max 3 sentences) of how the previous analysis specifically led to identifying THIS challenge."
        )
    )
    category: str = Field(
        description=(
            "The specific category of the challenge, reflecting its origin or nature (e.g., 'Feedback - Explicit Suggestion', "
            "'Implementation - Feature Engineering Flaw', 'Trace - Recurring Error'). This should align with and be more specific than the source analysis group (feedback, implementation_review, trace_history)."
        )
    )
    statement: str = Field(
        description=(
            "Description of the challenge in no more than three sentences, outlining the specific issue, "
            "observation, or area for improvement derived from past experiments or feedback."
        )
    )
    metric_impact: str = Field(
        description=(
            "Brief explanation (max 2 sentences) of why acting on this challenge (e.g., addressing the identified issue "
            "or leveraging the observation) is expected to improve the target metric or future iterations."
        )
    )
    caption: str = Field(description="Summarize the challenge concisely in around 5-15 words.")


class TraceChallenges(BaseModel):
    analysis: TraceAnalysis = Field(
        description=(
            "A structured summary of the analysis performed on feedback, implementation reviews, "
            "and experiment traces, which forms the basis for the challenges."
        )
    )
    challenges: List[TraceChallengeDetail] = Field(
        description=(
            "A list of challenges and learnings (e.g., at most five, prioritizing 'FEWER BUT BETTER') derived from the analysis. "
            "Each challenge should represent a valuable learning point aimed at guiding improvements for the target metric in subsequent experiments."
        )
    )


class HypothesisComponent(str, Enum):
    DataLoadSpec = "DataLoadSpec"
    FeatureEng = "FeatureEng"
    Model = "Model"
    Ensemble = "Ensemble"
    Workflow = "Workflow"


class HypothesisEvaluationReasoningScore(BaseModel):
    reasoning: str = Field(
        description="What is the quality of the hypothesis under this criteria? Answer in 1-2 sentence."
    )
    score: float = Field(description="The score of the hypothesis under this criteria between 1 and 10.")


class HypothesisEvaluation(BaseModel):
    alignment: HypothesisEvaluationReasoningScore = Field(
        description="The alignment of the proposed hypothesis with the identified challenge."
    )
    impact: HypothesisEvaluationReasoningScore = Field(
        description="The expected impact of the proposed hypothesis on the current SOTA implementation."
    )
    novelty: HypothesisEvaluationReasoningScore = Field(
        description="The novelty of the proposed hypothesis compared to existing solutions."
    )
    feasibility: HypothesisEvaluationReasoningScore = Field(
        description="The feasibility of implementing the proposed hypothesis in the current SOTA implementation."
    )
    risk_reward_balance: HypothesisEvaluationReasoningScore = Field(
        description="The risk-reward balance of implementing the proposed hypothesis."
    )


class HypothesisDetail(BaseModel):
    caption: str = Field(description="The caption of the challenge it is based on.")
    challenge: str = Field(
        description="Reaffirm the challenge within the current context (e.g., trace history, domain principles, or competition constraints). It should be no more than 2-3 sentences."
    )
    hypothesis: str = Field(
        description="The statement of the hypothesis. It could be a design of a new component, or a concise, testable statement derived from previous experimental outcomes."
    )
    metric_impact: str = Field(
        description=(
            "Brief explanation (max 2 sentences) of the expected impact of the hypothesis on the target metric."
        )
    )
    component: HypothesisComponent = Field(description="The component tag of the hypothesis.")
    evaluation: HypothesisEvaluation = Field(description="Evaluate the quality of the hypothesis.")


class HypothesisSimple(BaseModel):
    hypothesis: str = Field(
        description="The statement of the hypothesis. It could be a design of a new component, or a concise, testable statement derived from previous experimental outcomes."
    )
    component: HypothesisComponent = Field(description="The component tag of the hypothesis.")


class HypothesisList(BaseModel):
    deduplicated_challenges: List[str] = Field(
        description="A list of deduplicated challenge captions. Each must retain its original wording. If multiple captions are semantically identical, keep the first one."
    )
    hypotheses: List[HypothesisDetail] = Field(
        description="A non-empty list of hypotheses proposed for the next iteration, each corresponding to one challenge. The list length should match the number of challenges."
    )


class CodingSketch(BaseModel):
    current_state: str = Field(
        description="A summary of the current `main.py` script that serves as the baseline for the planned changes. Focusing on parts that are related to the hypothesis. If `main.py` does not yet exist (i.e., it will be created from scratch based on this sketch), use the string 'N/A'."
    )
    modifications: List[str] = Field(
        description="A list of specific, targeted changes to be applied to the existing code identified in `current_state`. Each string in the list should concisely describe (in 3-4 sentences): "
        "(a) the specific part of the code to be altered (e.g., a function name, a class, or a logical block); "
        "(b) the nature of the modification (e.g., bug fix, feature addition, refactoring of a small section, performance optimization, deletion); and "
        "(c) a brief explanation or high-level sketch of the new logic or change. "
        "If no direct modifications to existing code are planned (e.g., if creating an entirely new `main.py` as detailed in `structure`), this list should be empty."
    )
    structure: List[str] = Field(
        description="An outline of the new high-level architectural components (primarily functions and classes) if a new `main.py` script is being created from scratch, or if the existing `main.py` is undergoing a major refactor that fundamentally alters or replaces its core structure. "
        "Each string in the list should define a planned function or class, detailing its name, primary responsibility, key parameters (if applicable), return values (if applicable), and core functionality in 2-3 sentences. "
        "This field is typically used when `current_state` is 'N/A' or when the scope of change requires a new architectural blueprint rather than just targeted `modifications`. "
        "Leave empty if the plan only involves direct `modifications` to the existing structure in `current_state`."
    )
    sketch: str = Field(
        description="A detailed, step-by-step narrative that elaborates on how to implement the planned code. "
        "This section should synthesize the information from `modifications` (if any) and/or `structure` (if any) into a comprehensive and actionable coding plan for `main.py`. "
        "The content **must** be formatted using Markdown, with logical sections, key decision points, or implementation steps clearly organized by level-3 headings (i.e., `###`). "
        "This field should provide sufficient detail for a developer to understand the implementation flow, algorithms, data handling, and key logic points without ambiguity."
    )
    packages: List[str] = Field(
        default=None,
        description="A list of third-party package names (PyPI) that the planned implementation will import. "
        "Used to query the runtime environment dynamically. Leave `null` or omit if not applicable.",
    )


def draft_exp_in_decomposition(scen: Scenario, trace: DSTrace) -> None | DSDraftExpGen:
    next_missing_component = trace.next_incomplete_component()
    if next_missing_component is not None:
        return DSDraftExpGen(scen=scen).gen(
            component=next_missing_component,
            trace=trace,
        )
    else:
        return None


class DSProposalV1ExpGen(ExpGen):
    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
        # Drafting Stage
        if draft_exp := draft_exp_in_decomposition(self.scen, trace):
            return draft_exp

        # Guidelines:
        # System prompts: Shared condition you are facing
        # - scenario description: `scenario_desc`
        # - expected output format
        # User prompts: Task Specific information
        # - Previous Feedback
        # - Current sota implementation (encourage change based on it)
        # - Extra RAG
        sota_exp = trace.sota_experiment()
        if not isinstance(sota_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = sota_exp.experiment_workspace.file_dict.get("EDA.md", None)
        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)

        assert sota_exp is not None, "SOTA experiment is not provided."
        last_exp = trace.last_exp()
        # exp_and_feedback = trace.hist[-1]
        # last_exp = exp_and_feedback[0]

        # Step 1: Generate component
        # Describe current best solution using shared template
        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp, heading="Best of previous exploration of the scenario"
        )
        last_exp_diff = "\n".join(
            generate_diff_from_dict(sota_exp.experiment_workspace.file_dict, last_exp.experiment_workspace.file_dict)
        )  # we use file_dict for hitting the cache when replicate the experiment in another machine.

        all_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="all")

        exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=all_exp_feedback_list,
            type="all",
        )

        # Generate component using template with proper context
        component_sys_prompt = T(".prompts:component_gen.system").r(
            scenario=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            last_exp_diff=last_exp_diff,
            component_desc="\n".join(
                [
                    f"[{key}] {value}"
                    for key, value in T("scenarios.data_science.share:component_description").template.items()
                ]
            ),
        )

        component_user_prompt = T(".prompts:component_gen.user").r(
            exp_and_feedback_list_desc=exp_feedback_list_desc,
        )

        resp_dict_component: dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                component_user_prompt, component_sys_prompt, json_mode=True, json_target_type=Dict[str, str]
            )
        )

        component = resp_dict_component.get("component", "Component not provided")
        component_reason = resp_dict_component.get("reason", "Reason not provided")
        sota_exp_model_file_count = len(
            [
                k
                for k in sota_exp.experiment_workspace.file_dict.keys()
                if k.endswith(".py") and "test" not in k and k.startswith("model")
            ]
        )
        if sota_exp_model_file_count <= 1 and component == "Ensemble":
            component = "Model"

        # Why we should split component selection and steps after?
        # - after we know the selected component, we can use RAG.

        # Step 2: Generate the rest of the hypothesis & task
        component_info = get_component(component)

        if component_info:
            if DS_RD_SETTING.spec_enabled:
                task_spec = sota_exp.experiment_workspace.file_dict[component_info["spec_file"]]
            else:
                task_spec = T(f"scenarios.data_science.share:component_spec.{component}").r(
                    enable_notebook_conversion=DS_RD_SETTING.enable_notebook_conversion,
                )
            system_prompt = T(".prompts:direct_exp_gen.system").r(
                targets=component_info["target_name"],
                component=component,
                scenario=scenario_desc,
                hypothesis_specification=T(".prompts:hypothesis_specification").r(),
                hypothesis_output_format=T(".prompts:output_format.hypothesis").r(),
                task_specification=task_spec,
                task_output_format=component_info["task_output_format"],
                workflow_check=(not component == "Workflow"),
            )

            user_prompt = T(".prompts:direct_exp_gen.user").r(
                targets=component_info["target_name"],
                sota_exp_desc=sota_exp_desc,
                exp_and_feedback_list_desc=exp_feedback_list_desc,
                last_exp_diff=last_exp_diff,
            )

            def _append_retry(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
                # Only modify the user_prompt on retries (i > 0)
                user_prompt = args[0]
                user_prompt += "\n\nretrying..."
                return (user_prompt,), kwargs

            @wait_retry(retry_n=5, transform_args_fn=_append_retry)
            def _f(user_prompt):
                resp_dict = json.loads(
                    APIBackend().build_messages_and_create_chat_completion(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        json_mode=True,
                        # NOTE: corner cases.
                        # workflow_update may be a string
                        # model could have 2 level nested dict.
                        json_target_type=dict[str, dict[str, str | dict] | str],
                    )
                )
                assert "hypothesis_proposal" in resp_dict, "Hypothesis proposal not provided."
                assert "task_design" in resp_dict, "Task design not provided."
                task_class = component_info["task_class"]
                hypothesis_proposal = resp_dict.get("hypothesis_proposal", {})
                hypothesis = DSHypothesis(
                    component=component,
                    hypothesis=hypothesis_proposal.get("hypothesis", ""),
                    reason=component_reason + "\n" + hypothesis_proposal.get("reason", ""),
                    concise_reason=hypothesis_proposal.get("concise_reason", ""),
                    concise_observation=hypothesis_proposal.get("concise_observation", ""),
                    concise_justification=hypothesis_proposal.get("concise_justification", ""),
                    concise_knowledge=hypothesis_proposal.get("concise_knowledge", ""),
                )

                task_design = resp_dict.get("task_design", {})
                task_name = task_design["model_name"] if component == "Model" else component
                description = task_design.get(
                    "description", f"{component_info['target_name']} description not provided"
                )
                task = task_class(
                    name=task_name,
                    description=description,
                    **{k: task_design.get(k, v) for k, v in component_info.get("extra_params", {}).items()},
                )
                new_workflow_desc = resp_dict.get("workflow_update", "No update needed")
                return hypothesis, task, new_workflow_desc

            hypothesis, task, new_workflow_desc = _f(user_prompt)

            exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)
            # exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp.experiment_workspace)

            if new_workflow_desc != "No update needed":
                workflow_task = WorkflowTask(
                    name="Workflow",
                    description=new_workflow_desc,
                )
                exp.pending_tasks_list.append([workflow_task])
            return exp
        else:
            raise ValueError(f"Unknown component: {component}")


class DSProposalV2ExpGen(ExpGen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_response_schema = APIBackend().supports_response_schema()

    def identify_scenario_problem(
        self,
        scenario_desc: str,
        sota_exp_desc: str,
        exp_gen_plan: Dict,
        sibling_exp: List[DSExperiment] | None = None,
    ) -> Dict:
        sibling_hypotheses = [exp.hypothesis for exp in sibling_exp] if sibling_exp else None
        sys_prompt = T(".prompts_v2:scenario_problem.system").r(
            problem_output_format=(
                T(".prompts_v2:output_format.problem").r() if not self.supports_response_schema else None
            ),
            plan=exp_gen_plan,
            sibling_hypotheses=sibling_hypotheses,
        )
        user_prompt = T(".prompts_v2:scenario_problem.user").r(
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            response_format=ScenarioChallenges if self.supports_response_schema else {"type": "json_object"},
            json_target_type=Dict[str, Dict[str, str]] if not self.supports_response_schema else None,
        )
        if self.supports_response_schema:
            challenges = ScenarioChallenges(**json.loads(response))
            # Translate to problems
            problems = {o.caption: {"problem": o.statement, "reason": o.reasoning} for o in challenges.challenges}
            logger.info(f"Identified scenario problems:\n" + json.dumps(problems))
        else:
            problems = json.loads(response)
            logger.info(f"Identified scenario problems:\n" + json.dumps(problems))
        return problems

    def identify_feedback_problem(
        self,
        scenario_desc: str,
        exp_feedback_list_desc: str,
        sota_exp_desc: str,
        inject_diverse: bool = False,
        sibling_exp: List[DSExperiment] | None = None,
    ) -> Dict:
        sibling_hypotheses = [exp.hypothesis for exp in sibling_exp] if sibling_exp else None
        sys_prompt = T(".prompts_v2:feedback_problem.system").r(
            problem_output_format=(
                T(".prompts_v2:output_format.problem").r() if not self.supports_response_schema else None
            ),
            inject_diverse=inject_diverse,
            sibling_hypotheses=sibling_hypotheses,
        )
        user_prompt = T(".prompts_v2:feedback_problem.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            response_format=TraceChallenges if self.supports_response_schema else {"type": "json_object"},
            json_target_type=Dict[str, Dict[str, str]] if not self.supports_response_schema else None,
        )
        if self.supports_response_schema:
            challenges = TraceChallenges(**json.loads(response))
            # Translate to problems
            problems = {o.caption: {"problem": o.statement, "reason": o.reasoning} for o in challenges.challenges}
            logger.info(f"Identified feedback problems:\n" + json.dumps(problems))
        else:
            problems = json.loads(response)
            logger.info(f"Identified feedback problems:\n" + json.dumps(problems))
        return problems

    def identify_problem(
        self,
        current_sub_trace,
        scenario_desc,
        sota_exp_desc,
        exp_feedback_list_desc,
        inject_diverse,
        exp_gen_plan,
        sibling_exp: List[DSExperiment] | None = None,
    ) -> Dict:
        sota_exp_num = sum(1 for _, fb in current_sub_trace if fb.decision)
        failed_exp_num = len(current_sub_trace) - sota_exp_num
        weighted_exp_num = (sota_exp_num * 3 + failed_exp_num * 2) // 2
        self.scen_prob_multiplier = max(0, 3 - weighted_exp_num // 4)

        all_problems = {}
        if self.scen_prob_multiplier > 0:
            scen_problems = self.identify_scenario_problem(
                scenario_desc=scenario_desc,
                sota_exp_desc=sota_exp_desc,
                exp_gen_plan=exp_gen_plan,
                sibling_exp=sibling_exp,
            )
            for problem_name in scen_problems:
                scen_problems[problem_name]["label"] = "SCENARIO_PROBLEM"
                all_problems[problem_name] = scen_problems[problem_name]

        if self.scen_prob_multiplier < 3:
            fb_problems = self.identify_feedback_problem(
                scenario_desc=scenario_desc,
                exp_feedback_list_desc=exp_feedback_list_desc,
                sota_exp_desc=sota_exp_desc,
                inject_diverse=inject_diverse,
            )
            for problem_name in fb_problems:
                fb_problems[problem_name]["label"] = "FEEDBACK_PROBLEM"
                all_problems[problem_name] = fb_problems[problem_name]
        return all_problems

    @wait_retry(retry_n=10)
    def hypothesis_gen(
        self,
        component_desc: str,
        scenario_desc: str,
        exp_feedback_list_desc: str,
        sota_exp_desc: str,
        problems: dict,
        pipeline: bool,
        enable_idea_pool: bool,
        is_new_tree: bool,
        inject_diverse: bool = False,
        exp_gen_plan: Optional[Dict] = None,
        sibling_exp: List[DSExperiment] | None = None,
        former_user_instructions: UserInstructions | None = None,
    ) -> Dict:
        problem_formatted_str = ""
        for i, (problem_name, problem_dict) in enumerate(problems.items()):
            problem_formatted_str += f"## {i+1}. {problem_name}\n"
            problem_formatted_str += f"{problem_dict['problem']}\n"
            if "idea" in problem_dict:
                idea_formatted_str = DSIdea(problem_dict["idea"]).to_formatted_str()
                problem_formatted_str += f"Sampled Idea by user: \n{idea_formatted_str}\n"
            problem_formatted_str += "\n\n"
        sibling_hypotheses = [exp.hypothesis for exp in sibling_exp] if sibling_exp else None

        sys_prompt = T(".prompts_v2:hypothesis_gen.system").r(
            hypothesis_output_format=(
                T(".prompts_v2:output_format.hypothesis").r(pipeline=pipeline, enable_idea_pool=enable_idea_pool)
                if not self.supports_response_schema
                else None
            ),
            pipeline=pipeline,
            enable_idea_pool=enable_idea_pool,
            inject_diverse=inject_diverse,
            plan=exp_gen_plan,
            generate_unique_hypothesis=DS_RD_SETTING.enable_generate_unique_hypothesis and is_new_tree,
            enable_simple_hypothesis=DS_RD_SETTING.enable_simple_hypothesis,
            sibling_hypotheses=sibling_hypotheses,
            former_user_instructions_str=str(former_user_instructions) if former_user_instructions else None,
        )
        user_prompt = T(".prompts_v2:hypothesis_gen.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
            problems=problem_formatted_str,
            enable_idea_pool=enable_idea_pool,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            response_format=HypothesisList if self.supports_response_schema else {"type": "json_object"},
            json_target_type=(
                Dict[str, Dict[str, str | Dict[str, str | int]]] if not self.supports_response_schema else None
            ),
        )
        if self.supports_response_schema:
            hypotheses = HypothesisList(**json.loads(response))
            resp_dict = {
                h.caption: {
                    "reason": h.challenge,
                    "component": h.component.value,
                    "hypothesis": h.hypothesis,
                    "evaluation": {
                        "alignment_score": h.evaluation.alignment.score,
                        "impact_score": h.evaluation.impact.score,
                        "novelty_score": h.evaluation.novelty.score,
                        "feasibility_score": h.evaluation.feasibility.score,
                        "risk_reward_balance_score": h.evaluation.risk_reward_balance.score,
                    },
                }
                for h in hypotheses.hypotheses
            }
        else:
            resp_dict = json.loads(response)
        logger.info(f"Generated hypotheses:\n" + json.dumps(resp_dict, indent=2))

        # make sure the problem name is aligned
        problem_keys = set(problems.keys())
        resp_keys = set(resp_dict.keys())
        if not resp_keys.issubset(problem_keys):
            logger.error("Problem names are not fully aligned. Retrying...")
            raise ValueError("Problem names are not fully aligned.")

        return resp_dict

    @wait_retry(retry_n=5)
    def hypothesis_critique(
        self,
        hypothesis_dict: Dict,
        problems_dict: Dict,
        scenario_desc: str,
        sota_exp_desc: str,
        exp_feedback_list_desc: str,
    ) -> Dict:
        """
        Critique the generated hypotheses, identifying flaws and suggesting improvements.
        """
        hypotheses_formatted = ""
        for i, (problem_name, hypothesis_data) in enumerate(hypothesis_dict.items()):

            problem_info = problems_dict.get(problem_name, {})
            hypotheses_formatted += f"## {i+1}. **Problem Name:** {problem_name}\n"
            hypotheses_formatted += f"**Original Problem:** {problem_info.get('problem', 'Not available')}\n"
            hypotheses_formatted += f"**Component:** {hypothesis_data.get('component', 'Unknown')}\n"
            hypotheses_formatted += f"**Hypothesis:** {hypothesis_data.get('hypothesis', 'Not provided')}\n"
            hypotheses_formatted += f"**Reason:** {hypothesis_data.get('reason', 'Not provided')}\n\n"

        sys_prompt = T(".prompts_v2:hypothesis_critique.system").r(
            critique_output_format=T(".prompts_v2:output_format.critique").r(),
        )
        user_prompt = T(".prompts_v2:hypothesis_critique.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
            hypotheses_formatted=hypotheses_formatted,
        )

        # Use json_object mode since hypothesis names are dynamic
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            response_format={"type": "json_object"},
            json_target_type=dict,
        )

        response_dict = json.loads(response)

        # Improved error handling and validation
        if "critiques" in response_dict:
            critiques = response_dict["critiques"]
        else:
            # If format is incorrect, try to extract critiques directly
            # Validate that all expected problem names are present
            expected_problems = set(hypothesis_dict.keys())
            available_problems = set(response_dict.keys())

            if expected_problems.issubset(available_problems):
                critiques = response_dict
            else:
                raise ValueError(
                    f"Critique response missing expected problems. Expected: {expected_problems}, Got: {available_problems}"
                )

        # Validate that we have critiques for all hypotheses
        missing_critiques = set(hypothesis_dict.keys()) - set(critiques.keys())
        if missing_critiques:
            logger.warning(f"Missing critiques for problems: {missing_critiques}")
            # Add default critiques for missing ones
            for problem_name in missing_critiques:
                critiques[problem_name] = {"critique": "No specific critique available for this hypothesis."}

        logger.info(f"Generated critiques for {len(critiques)} hypothesis")
        return critiques

    @wait_retry(retry_n=5)
    def hypothesis_rewrite(
        self,
        hypothesis_dict: Dict,
        critiques_dict: Dict,
        scenario_desc: str,
        sota_exp_desc: str,
        exp_feedback_list_desc: str,
        sibling_exp: List[DSExperiment] | None = None,
        former_user_instructions: UserInstructions | None = None,
    ) -> Dict:
        """
        Generate improved hypotheses based on critique feedback for each original hypothesis.
        Returns a dict with the same keys as hypothesis_dict, containing improved versions.
        """
        sibling_hypotheses = [exp.hypothesis for exp in sibling_exp] if sibling_exp else None

        hypothesis_critique_pairs = ""
        for i, problem_name in enumerate(hypothesis_dict.keys()):
            hypothesis_data = hypothesis_dict[problem_name]
            critique_data = critiques_dict.get(problem_name, {})

            hypothesis_critique_pairs += f"## Original Hypothesis {i+1}: {problem_name}\n"
            hypothesis_critique_pairs += f"**Hypothesis:** {hypothesis_data.get('hypothesis', 'Not provided')}\n"
            hypothesis_critique_pairs += f"**Component:** {hypothesis_data.get('component', 'Unknown')}\n"
            hypothesis_critique_pairs += f"**Reasoning:** {hypothesis_data.get('reason', 'Not provided')}\n"
            hypothesis_critique_pairs += f"**Critique:** {critique_data.get('critique', 'No critique available')}\n\n"

        time_status = None
        if DS_RD_SETTING.enable_scale_check and RD_Agent_TIMER_wrapper.timer.started:
            remain_time = RD_Agent_TIMER_wrapper.timer.remain_time()
            all_duration = RD_Agent_TIMER_wrapper.timer.all_duration
            remain_percent = remain_time / all_duration
            time_status = (
                f"Remain time: {remain_time.total_seconds() / 3600:.2f} hours, "
                f"{remain_percent:.2%} remaining of total time: {all_duration.total_seconds() / 3600:.2f} hours."
            )

        sys_prompt = T(".prompts_v2:hypothesis_rewrite.system").r(
            rewrite_output_format=T(".prompts_v2:output_format.rewrite").r(
                enable_scale_check=DS_RD_SETTING.enable_scale_check
            ),
            enable_scale_check=DS_RD_SETTING.enable_scale_check,
            sibling_hypotheses=sibling_hypotheses,
            former_user_instructions_str=str(former_user_instructions) if former_user_instructions else None,
        )
        user_prompt = T(".prompts_v2:hypothesis_rewrite.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
            hypothesis_critique_pairs=hypothesis_critique_pairs,
            time_status=time_status,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            response_format={"type": "json_object"},
            json_target_type=dict,
        )

        improved_hypotheses_dict = json.loads(response)

        # Validate that we have rewritten hypotheses for all original hypotheses
        expected_problems = set(hypothesis_dict.keys())
        available_problems = set(  # The code snippet provided is a comment in Python. It appears to be
            # a placeholder for a function or variable named
            # `improved_hypotheses_dict`. The actual implementation of this
            # function or variable is not provided in the code snippet.
            improved_hypotheses_dict.keys()
        )

        if not expected_problems.issubset(available_problems):
            missing_problems = expected_problems - available_problems
            # Raise exception to trigger retry mechanism
            raise ValueError(f"Rewrite response missing expected problems. Missing: {missing_problems}")

        # Note: We don't preserve 'inspired' field from original hypotheses
        # because after critique and rewrite, the hypothesis may have changed significantly
        # and the original inspiration may no longer be relevant

        logger.info(
            f"Generated rewritten versions of {len(improved_hypotheses_dict)} hypotheses based on critique feedback"
        )
        return improved_hypotheses_dict

    def compute_top_scores(
        self,
        hypothesis_dict: dict,
    ) -> pd.Series:
        """
        Compute weighted total scores for each hypothesis and return the top five.
        """
        weights = {
            "alignment_score": 0.2,
            "impact_score": 0.4,
            "novelty_score": 0.2,
            "feasibility_score": 0.1,
            "risk_reward_balance_score": 0.1,
        }
        scores_dict = {}
        for problem_name in hypothesis_dict:
            if "hypothesis" not in hypothesis_dict[problem_name]:
                continue
            scores_dict[problem_name] = {}
            for score_key in weights:
                if score_key not in hypothesis_dict[problem_name]["evaluation"]:
                    scores_dict[problem_name][score_key] = 0
                else:
                    try:
                        scores_dict[problem_name][score_key] = (
                            float(hypothesis_dict[problem_name]["evaluation"][score_key]) * weights[score_key]
                        )
                    except (ValueError, TypeError):
                        scores_dict[problem_name][score_key] = 0

        scores = pd.DataFrame(scores_dict)
        scores_sorted = scores.sum().sort_values(ascending=False)
        return scores_sorted[:5]

    def select_hypothesis(
        self,
        scores_sorted: pd.Series,
        hypothesis_dict: dict,
        problem_dict: dict,
    ) -> int:
        """
        From the top five hypotheses (by weighted score), select one based on additional weighting rules
        for 'inspired' flag and 'SCENARIO_PROBLEM' label. Returns the chosen hypothesis name and a
        DSHypothesis instance.
        """
        # Increase the weight of the hypothesis that is inspired by the idea pool to 3x.
        # Linear decay the weight of the scenario problem from 3x to 0x.
        index_to_pick_pool_list = []
        for j, problem_name in enumerate(scores_sorted.index):
            if hypothesis_dict[problem_name].get("inspired", False):
                index_to_pick_pool_list.extend([j] * 2)
            if problem_dict[problem_name]["label"] == "SCENARIO_PROBLEM":
                index_to_pick_pool_list.extend([j] * self.scen_prob_multiplier)
            elif problem_dict[problem_name]["label"] == "FEEDBACK_PROBLEM":
                index_to_pick_pool_list.extend([j] * (3 - self.scen_prob_multiplier))
            else:
                index_to_pick_pool_list.extend([j] * 1)
        logger.info(f"index_to_pick_pool_list: {index_to_pick_pool_list}")

        # Create a random but reproducible integer
        reproducible_int = int.from_bytes(bytes.fromhex(md5_hash(scores_sorted.to_string())), byteorder="big") % len(
            index_to_pick_pool_list
        )
        return index_to_pick_pool_list[reproducible_int]

    def _cosine_similarity_matrix_torch(self, A, B):
        import torch

        dot_products = torch.matmul(A, B.T)
        A_norms = torch.norm(A, dim=1, keepdim=True)
        B_norms = torch.norm(B, dim=1, keepdim=True).T
        return dot_products / (A_norms * B_norms)

    def _prob_dis_torch(
        self,
        current_sota_score_in_current_trace,
        extra_hypo_l: list[tuple[DSHypothesis, float]],
        hypothesis_candidates,
        competition,
        path_length,
    ):
        import torch

        history_hypo_str, history_scores = [], []
        for hypo, score in extra_hypo_l:
            history_hypo_str.append(hypo.hypothesis)
            history_scores.append(score)

        target_texts = [v["hypothesis"] for v in hypothesis_candidates.values()]
        target_embs = torch.tensor(APIBackend().create_embedding(target_texts), dtype=torch.float32)

        if not history_hypo_str:
            return []
        history_embs = torch.tensor(APIBackend().create_embedding(history_hypo_str), dtype=torch.float32)
        sim_matrix = self._cosine_similarity_matrix_torch(target_embs, history_embs)
        candidate_scores = [current_sota_score_in_current_trace for i in range(len(target_texts))]
        candidate_scores = torch.tensor(candidate_scores, dtype=torch.float32).unsqueeze(1)
        history_scores = torch.tensor(history_scores, dtype=torch.float32).unsqueeze(0)
        bigger_is_better = get_metric_direction(competition)
        if bigger_is_better:
            score_diff_matrix = history_scores - candidate_scores
        else:
            score_diff_matrix = candidate_scores - history_scores
        alpha, beta = 1.0, 1.0
        if current_sota_score_in_current_trace == -1:
            alpha, beta = 1.0, 0
        gamma = math.log(2) / 30
        logits = alpha * sim_matrix * math.exp(-gamma * path_length) + beta * torch.tanh(score_diff_matrix)
        logits = torch.clamp(logits, min=-2, max=2)
        probs = torch.softmax(logits, dim=1)

        num_candidates = probs.size(-1)
        n_samples = min(2, num_candidates)
        sampled_indices = torch.multinomial(probs, num_samples=n_samples).squeeze(1)
        flat_indices = sampled_indices.flatten().unique().tolist()
        if bigger_is_better:
            best_idx = history_scores[0].argmax().item()
            best_entry = (history_hypo_str[best_idx], history_scores[0, best_idx])
        else:
            best_idx = history_scores[0].argmin().item()
            best_entry = (history_hypo_str[best_idx], history_scores[0, best_idx])
        if len(flat_indices) > 2:
            flat_indices = flat_indices[:2]
        sampled_history_list = [best_entry] + [
            (history_hypo_str[i], history_scores[0, i]) for i in flat_indices if i != best_idx
        ]
        return sampled_history_list

    def _get_path(self, node, parent_nodes):
        # FIXME: we should remove it in the future.
        path = [node]
        parent = parent_nodes.get(node)
        if parent is not None:
            path.extend(self._get_path(parent, parent_nodes))
        return path

    def _get_current_exp_score_list(self, trace, competition):
        parent_nodes = {}
        for node in range(len(trace.hist)):
            parents = trace.get_parents(node)
            parent_nodes[node] = parents[-2] if len(parents) > 1 else None
        # FIXME: add the convert logic to method in trace
        if hasattr(trace, "idx2loop_id"):
            parent_nodes = {
                trace.idx2loop_id[n]: trace.idx2loop_id[r] if r is not None else r for n, r in parent_nodes.items()
            }
        if trace.current_selection:
            current_parent_record_id = trace.current_selection[0]  # record id
        else:
            return -1, 0
        # current_parent_loop_id = trace.idx2loop_id[current_parent_record_id]# loop id
        loop_id2idx = {v: k for k, v in trace.idx2loop_id.items()}

        loop_id_list = self._get_path(trace.idx2loop_id[current_parent_record_id], parent_nodes)

        score_list = [
            trace.hist[loop_id2idx[loop_id]][0].result.loc["ensemble"].iloc[0].round(3)
            for loop_id in loop_id_list
            if trace.hist[loop_id2idx[loop_id]][1].decision == True
        ]
        if score_list:
            bigger_is_better = get_metric_direction(competition)
            if bigger_is_better:
                return max(score_list), len(loop_id_list)
            else:
                return min(score_list), len(loop_id_list)
        else:
            return -1, len(loop_id_list)

    def _llm_select_extra_hypo(self, trace: DSTrace) -> list[tuple[str, float]]:
        """
        Retrieve a list of additional hypotheses along with their ensemble scores
        from the given experiment trace, intended for input into an LLM-based selection mechanism.

        Parameters:
            trace (DSTrace):

        Returns:
            list[tuple[str, float]]:
                A list of tuples, where each tuple consists of:
                    - str: The hypothesis description from a selected experiment.
                      Example: "Use XGBoost with tuned learning_rate".
                    - float: The associated ensemble result score, rounded to 3 decimal places.
                      Example: 0.845
                Example:
                    [
                        ("Try RandomForest with 200 estimators", 0.812),
                        ("Use LightGBM with early stopping", 0.834)
                    ]
        """
        return [
            (exp.hypothesis, exp.result.loc["ensemble"].iloc[0])
            for exp, _ in trace.experiment_and_feedback_list_after_init(return_type="sota", search_type="all")
        ]

    @wait_retry(retry_n=5)
    def hypothesis_select_with_llm(
        self,
        scenario_desc: str,
        exp_feedback_list_desc: str,
        sota_exp_desc: str,
        hypothesis_candidates: dict,
        trace: DSTrace,
    ):
        res_time = RD_Agent_TIMER_wrapper.timer.remain_time()
        ratio_merge_or_ensemble = DS_RD_SETTING.ratio_merge_or_ensemble

        total_time = RD_Agent_TIMER_wrapper.timer.all_duration
        # FIXME: total_time could be None
        use_time = round(total_time.total_seconds(), 2) - round(res_time.total_seconds(), 2)
        use_ratio = 100 * use_time / round(total_time.total_seconds(), 2)
        use_ratio = round(use_ratio, 2)

        full_time = self.scen.real_full_timeout() / 3600
        # FIXME: less magic number
        time_list_success = [-3600] + [
            tr[0].running_info.running_time
            for tr in trace.retrieve_search_list(search_type="ancestors")
            if getattr(tr[1], "decision", False)
        ]
        time_max = max(time_list_success) / 3600
        sota_flag = (
            hasattr(trace, "sota_exp_to_submit") and trace.sota_exp_to_submit is not None
        )  # ----> V10 CODE VERSION
        # bvs = BestValidSelector()  # ----> V14 CODE VERSION
        # sota_exp = bvs.get_sota_exp_to_submit(trace)  # ----> V14 CODE VERSION
        # sota_flag = sota_exp is not None and sota_exp.result is not None  # ----> V14 CODE VERSION

        if sota_flag:
            # current_sota_score = sota_exp.result.loc["ensemble"].iloc[0].round(3)  # ----> V14 CODE VERSION
            current_sota_score = (
                trace.sota_exp_to_submit.result.loc["ensemble"].iloc[0].round(3)
            )  # ----> V10 CODE VERSION
        else:
            current_sota_score = -1

        competition = trace.scen.competition
        if sota_flag:
            current_sota_score_in_current_trace, path_length = self._get_current_exp_score_list(trace, competition)
        else:
            current_sota_score_in_current_trace = -1
            path_length = 0

        # extra_exp_feedback_list_desc: str,
        # exp_feedback_scores: list,
        extra_hypo_l = self._llm_select_extra_hypo(trace)
        if len(extra_hypo_l) > 0:
            # TODO:
            selected_extra_hypo_l = self._prob_dis_torch(
                current_sota_score_in_current_trace,
                extra_hypo_l,
                hypothesis_candidates,
                competition,
                path_length,
            )
        else:
            selected_extra_hypo_l = None
        hypothesis_candidates = str(json.dumps(hypothesis_candidates, indent=2))

        sys_prompt = T(".prompts_v2:hypothesis_select.system").r(
            hypothesis_candidates=hypothesis_candidates,
            res_time=round(res_time.total_seconds() / 3600, 2),
            full_time=full_time,
            use_ratio=use_ratio,
            time_max=round(time_max, 2),
            merge_hours=DS_RD_SETTING.merge_hours,
            # extra_exp_feedback_list_desc=extra_exp_feedback_list_str,
            selected_extra_hypo_l=selected_extra_hypo_l,
            hypothesis_output_format=(
                T(".prompts_v2:output_format.hypothesis_select_format").r()
                if not self.supports_response_schema
                else None
            ),
            sota_flag=sota_flag,
            current_sota_score=current_sota_score,
            ratio_merge_or_ensemble=ratio_merge_or_ensemble,
            current_sota_score_in_current_trace=current_sota_score_in_current_trace,
        )

        user_prompt = T(".prompts_v2:hypothesis_select.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            response_format=HypothesisSimple if self.supports_response_schema else {"type": "json_object"},
            json_target_type=(Dict[str, str] if not self.supports_response_schema else None),
        )

        response_dict = json.loads(response)
        assert response_dict.get("component") in HypothesisComponent.__members__, f"Invalid component"
        assert response_dict.get("hypothesis") is not None, f"Invalid hypothesis"
        return response_dict

    # END: for support llm-based hypothesis selection  -----

    def hypothesis_rank(
        self, hypothesis_dict: dict, problem_dict: dict, selected_idx: Optional[int] = None
    ) -> Tuple[str, DSHypothesis]:
        """
        Wrapper method that computes the top five hypotheses by weighted scoring and then selects one
        according to additional weighting rules.
        """
        scores_sorted = self.compute_top_scores(hypothesis_dict)
        if selected_idx is None:
            selected_idx = self.select_hypothesis(
                scores_sorted=scores_sorted, hypothesis_dict=hypothesis_dict, problem_dict=problem_dict
            )

        max_score_problem_name = scores_sorted.index[selected_idx]
        problem_dict = problem_dict.get(max_score_problem_name, {})

        return max_score_problem_name, DSHypothesis(
            component=hypothesis_dict[max_score_problem_name].get("component", "Model"),
            hypothesis=hypothesis_dict[max_score_problem_name].get("hypothesis", "Hypothesis not provided"),
            reason=hypothesis_dict[max_score_problem_name].get("reason", "Reason not provided"),
            problem_name=max_score_problem_name,
            problem_desc=problem_dict.get("problem", "Problem description not provided"),
            problem_label=problem_dict.get("label", "FEEDBACK_PROBLEM"),
            appendix=hypothesis_dict[max_score_problem_name].get("appendix", None),
        )

    def task_gen(
        self,
        component_desc: str,
        scenario_desc: str,
        sota_exp_desc: str,
        sota_exp: DSExperiment,
        hypotheses: list[DSHypothesis],
        hypotheses_candidates: list[DSHypothesis],
        pipeline: bool,
        failed_exp_feedback_list_desc: str,
        fb_to_sota_exp: ExperimentFeedback | None = None,
        sibling_exp: List[DSExperiment] | None = None,
        former_user_instructions: UserInstructions = None,
    ) -> DSExperiment:
        if pipeline:
            component_info = get_component("Pipeline")
        else:
            component_info = get_component(hypotheses[0].component)
        data_folder_info = self.scen.processed_data_folder_description
        workflow_check = not pipeline and hypotheses[0].component != "Workflow"

        sibling_tasks = [exp.pending_tasks_list[0][0].description for exp in sibling_exp] if sibling_exp else []
        sys_prompt = T(".prompts_v2:task_gen.system").r(
            task_output_format=component_info["task_output_format"] if not self.supports_response_schema else None,
            component_desc=component_desc,
            workflow_check=workflow_check,
            metric_name=self.scen.metric_name,
            sibling_tasks=sibling_tasks,
            fix_seed_and_data_split=DS_RD_SETTING.fix_seed_and_data_split,
            former_user_instructions_str=str(former_user_instructions) if former_user_instructions else None,
        )
        user_prompt = T(".prompts_v2:task_gen.user").r(
            scenario_desc=scenario_desc,
            data_folder_info=data_folder_info,
            sota_exp_desc=sota_exp_desc,
            hypotheses=hypotheses,
            failed_exp_and_feedback_list_desc=failed_exp_feedback_list_desc,
            eda_improvement=fb_to_sota_exp.eda_improvement if fb_to_sota_exp else None,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            response_format=CodingSketch if self.supports_response_schema else {"type": "json_object"},
            json_target_type=Dict[str, str | List[str] | Dict[str, str]] if not self.supports_response_schema else None,
        )

        task_dict = json.loads(response)

        # 1) explain the response and get main task_description
        not_found_str = f"{component_info['target_name']} description not provided"
        if self.supports_response_schema:
            # task_dict: {"sketch": str, ...}
            task_desc = task_dict.get("sketch", not_found_str)
        else:
            if workflow_check:
                # task_dict:  {"task_design": ...., "workflow_update": ....}
                task_desc = task_dict.get("task_design", {}).get("description", not_found_str)
            else:
                # task_dict:  {"description": ....}
                task_desc = task_dict.get("description", not_found_str)
        # task_desc: str, a description of the task

        # 2) create the main task
        logger.info(f"Task design:\n{task_desc}")
        task_name = hypotheses[0].component
        task_class = component_info["task_class"]
        task = task_class(
            name=task_name,
            description=task_desc,
        )

        assert isinstance(task, PipelineTask), f"Task {task_name} is not a PipelineTask, got {type(task)}"
        # only for llm with response schema.(TODO: support for non-schema llm?)
        # If the LLM provides a "packages" field (list[str]), compute runtime environment now and cache it for subsequent prompts in later loops.
        if isinstance(task_dict, dict) and "packages" in task_dict and isinstance(task_dict["packages"], list):
            pkgs: list[str] = [str(p) for p in task_dict["packages"]]
            # Persist for later stages
            task.package_info = get_packages(pkgs)

        exp = DSExperiment(
            pending_tasks_list=[[task]], hypothesis=hypotheses[0], hypothesis_candidates=hypotheses_candidates
        )
        if sota_exp is not None:
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp.experiment_workspace)

        # 3) create the workflow update task
        if workflow_check:
            workflow_task = WorkflowTask(
                name="Workflow",
                description=task_dict.get("workflow_update", "No update needed"),
            )
            exp.pending_tasks_list.append([workflow_task])

        # 4) set user instructions
        if former_user_instructions is not None:
            exp.set_user_instructions(former_user_instructions)
        return exp

    def get_all_hypotheses(self, problem_dict: dict, hypothesis_dict: dict) -> list[DSHypothesis]:
        result = []
        for name, data in hypothesis_dict.items():
            problem_data = problem_dict.get(name, {})
            result.append(
                DSHypothesis(
                    component=data.get("component", "Model"),
                    hypothesis=data.get("hypothesis", "Hypothesis not provided"),
                    reason=data.get("reason", "Reason not provided"),
                    problem_name=name,
                    problem_desc=problem_data.get("problem", "Problem description not provided"),
                    problem_label=problem_data.get("label", "FEEDBACK_PROBLEM"),
                    appendix=data.get("appendix", None),
                )
            )
        return result

    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
        pipeline = DS_RD_SETTING.coder_on_whole_pipeline
        if not pipeline and (draft_exp := draft_exp_in_decomposition(self.scen, trace)):
            return draft_exp

        if pipeline:
            component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()
        else:
            component_desc = "\n".join(
                [
                    f"[{key}] {value}"
                    for key, value in T("scenarios.data_science.share:component_description").template.items()
                ]
            )

        if (sota_exp_fb := trace.sota_experiment_fb()) is None:
            sota_exp, fb_to_sota_exp = None, None
        else:
            sota_exp, fb_to_sota_exp = sota_exp_fb

        if not isinstance(sota_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = sota_exp.experiment_workspace.file_dict.get("EDA.md", None)
        scenario_desc = self.scen.get_scenario_all_desc(eda_output=eda_output)

        # the only sota exp
        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp, heading="Best of previous exploration of the scenario"
        )

        # all exp and feedbacks
        exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="all"),
            type="all",
            pipeline=pipeline,
        )

        # all failed exp and feedbacks
        failed_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="failed")
        failed_exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=failed_exp_feedback_list,
            type="failed",
            pipeline=pipeline,
        )
        if len(failed_exp_feedback_list) == 0:
            former_user_instructions = None
        else:
            former_user_instructions = failed_exp_feedback_list[-1][0].user_instructions

        # NOTE: we currently don't support inject diverse problems for the parallel + multi-trace mode,
        if DS_RD_SETTING.enable_inject_diverse and len(trace.hist) > 0:
            if len(trace.current_selection) == 0:
                # start a new sub-trace, and inject diverse problems.
                inject_diverse = True
                logger.info("Start a new sub-trace, and inject diverse problems.")
            else:
                inject_diverse = False
        else:
            inject_diverse = False

        sibling_exp = trace.get_sibling_exps() if trace.should_inject_diversity() else None

        # Step 1: Identify problems
        all_problems = self.identify_problem(
            current_sub_trace=trace.get_parent_exps(),
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            exp_feedback_list_desc=exp_feedback_list_desc,
            inject_diverse=inject_diverse,
            exp_gen_plan=plan.get("exp_gen") if plan else None,
            sibling_exp=sibling_exp,
        )

        # Step 1.5: Sample ideas from idea pool
        if DS_RD_SETTING.enable_knowledge_base:
            all_problems = trace.knowledge_base.sample_ideas(
                problems=all_problems,
                scenario_desc=scenario_desc,
                exp_feedback_list_desc=exp_feedback_list_desc,
                sota_exp_desc=sota_exp_desc,
                competition_desc=self.scen.get_competition_full_desc(),
            )

        # sub-trace begin flag
        is_new_tree = trace.is_selection_new_tree()

        # Step 2: Propose hypothesis based on the identified problems (and sampled ideas)
        hypothesis_dict = self.hypothesis_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            exp_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
            problems=all_problems,
            pipeline=pipeline,
            enable_idea_pool=DS_RD_SETTING.enable_knowledge_base,
            inject_diverse=inject_diverse,
            exp_gen_plan=plan.get("exp_gen") if plan else None,
            is_new_tree=is_new_tree,
            sibling_exp=sibling_exp,
            former_user_instructions=former_user_instructions,
        )
        if not pipeline:
            sota_exp_model_file_count = len(
                [
                    k
                    for k in sota_exp.experiment_workspace.file_dict.keys()
                    if k.endswith(".py") and "test" not in k and k.startswith("model")
                ]
            )
            if sota_exp_model_file_count <= 1:
                pop_names = []
                for problem_name in hypothesis_dict:
                    if hypothesis_dict[problem_name].get("component", "") == "Ensemble":
                        pop_names.append(problem_name)
                for name in pop_names:
                    hypothesis_dict.pop(name)

        # Step 2.1 & 2.2: Hypothesis Critique and Rewrite Stage (controlled by enable_hypo_critique_rewrite)
        if DS_RD_SETTING.enable_hypo_critique_rewrite and len(trace.hist) > 0:
            logger.info(f"Hypothesis critique and rewrite enabled - processing {len(hypothesis_dict)} hypotheses")

            # Critic Stage - Evaluate and identify flaws in hypotheses
            logger.info(
                f"Starting critic stage - evaluating {len(hypothesis_dict)} hypotheses for flaws and improvements"
            )
            try:
                critiques_dict = self.hypothesis_critique(
                    hypothesis_dict=hypothesis_dict,
                    problems_dict=all_problems,
                    scenario_desc=scenario_desc,
                    sota_exp_desc=sota_exp_desc,
                    exp_feedback_list_desc=exp_feedback_list_desc,
                )
                logger.info(f"Generated critiques for {len(critiques_dict)} hypotheses")

                # Rewriter Stage - Generate improved hypotheses based on critiques
                logger.info(f"Starting rewriter stage - generating improved hypotheses based on critique feedback")
                hypothesis_dict = self.hypothesis_rewrite(
                    hypothesis_dict=hypothesis_dict,
                    critiques_dict=critiques_dict,
                    scenario_desc=scenario_desc,
                    sota_exp_desc=sota_exp_desc,
                    exp_feedback_list_desc=exp_feedback_list_desc,
                    sibling_exp=sibling_exp,
                    former_user_instructions=former_user_instructions,
                )
                logger.info(f"Successfully completed hypothesis critique and rewrite process")
            except Exception as e:
                logger.warning(f"Hypothesis critique and rewrite failed: {e}")
                logger.info(f"Using original hypotheses as fallback instead of improved versions")
        else:
            logger.info(f"Hypothesis critique and rewrite disabled - using original {len(hypothesis_dict)} hypotheses")

        # Step 3: Select the best hypothesis
        if DS_RD_SETTING.llm_select_hypothesis:
            response_dict = self.hypothesis_select_with_llm(
                scenario_desc=scenario_desc,
                exp_feedback_list_desc=exp_feedback_list_desc,
                # extra_exp_feedback_list_desc=extra_exp_feedback_list_desc,
                # exp_feedback_scores=exp_feedback_scores,
                sota_exp_desc=sota_exp_desc,
                hypothesis_candidates=hypothesis_dict,
                trace=trace,
            )
            new_hypothesis = DSHypothesis(
                component=response_dict.get("component"), hypothesis=response_dict.get("hypothesis")
            )
            pickled_problem_name = None
        else:
            pickled_problem_name, new_hypothesis = self.hypothesis_rank(
                hypothesis_dict=hypothesis_dict,
                problem_dict=all_problems,
            )

        # Step 3.5: Update knowledge base with the picked problem
        if DS_RD_SETTING.enable_knowledge_base:
            trace.knowledge_base.update_pickled_problem(all_problems, pickled_problem_name)

        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp,
            hypotheses=(
                [new_hypothesis]
                if not trace.is_selection_new_tree()
                else self.get_all_hypotheses(all_problems, hypothesis_dict)
            ),
            hypotheses_candidates=self.get_all_hypotheses(all_problems, hypothesis_dict),
            pipeline=pipeline,
            failed_exp_feedback_list_desc=failed_exp_feedback_list_desc,
            fb_to_sota_exp=fb_to_sota_exp,
            sibling_exp=sibling_exp,
            former_user_instructions=former_user_instructions,
        )
