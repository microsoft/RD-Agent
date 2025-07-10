import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.proposal import ExpGen
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.dev.feedback import ExperimentFeedback
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.draft.draft import (
    DSDraftExpGen,  # TODO: DSDraftExpGen should be moved to router in the further
)
from rdagent.scenarios.data_science.proposal.exp_gen.idea_pool import DSIdea
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
    def gen(self, trace: DSTrace) -> DSExperiment:
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
                task_spec = T(f"scenarios.data_science.share:component_spec.{component}").r()
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

    def identify_scenario_problem(self, scenario_desc: str, sota_exp_desc: str) -> Dict:
        sys_prompt = T(".prompts_v2:scenario_problem.system").r(
            problem_output_format=(
                T(".prompts_v2:output_format.problem").r() if not self.supports_response_schema else None
            ),
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
        self, scenario_desc: str, exp_feedback_list_desc: str, sota_exp_desc: str, inject_diverse: bool = False
    ) -> Dict:
        sys_prompt = T(".prompts_v2:feedback_problem.system").r(
            problem_output_format=(
                T(".prompts_v2:output_format.problem").r() if not self.supports_response_schema else None
            ),
            inject_diverse=inject_diverse,
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
        self, current_sub_trace, scenario_desc, sota_exp_desc, exp_feedback_list_desc, inject_diverse
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

    @wait_retry(retry_n=5)
    def hypothesis_gen(
        self,
        component_desc: str,
        scenario_desc: str,
        exp_feedback_list_desc: str,
        sota_exp_desc: str,
        problems: dict,
        pipeline: bool,
        enable_idea_pool: bool,
        inject_diverse: bool = False,
    ) -> Dict:
        problem_formatted_str = ""
        for i, (problem_name, problem_dict) in enumerate(problems.items()):
            problem_formatted_str += f"## {i+1}. {problem_name}\n"
            problem_formatted_str += f"{problem_dict['problem']}\n"
            if "idea" in problem_dict:
                idea_formatted_str = DSIdea(problem_dict["idea"]).to_formatted_str()
                problem_formatted_str += f"Sampled Idea by user: \n{idea_formatted_str}\n"
            problem_formatted_str += "\n\n"

        sys_prompt = T(".prompts_v2:hypothesis_gen.system").r(
            hypothesis_output_format=(
                T(".prompts_v2:output_format.hypothesis").r(pipeline=pipeline, enable_idea_pool=enable_idea_pool)
                if not self.supports_response_schema
                else None
            ),
            pipeline=pipeline,
            enable_idea_pool=enable_idea_pool,
            inject_diverse=inject_diverse,
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
                    "component": h.component,
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
        )

    def task_gen(
        self,
        component_desc: str,
        scenario_desc: str,
        sota_exp_desc: str,
        sota_exp: DSExperiment,
        hypotheses: list[DSHypothesis],
        pipeline: bool,
        failed_exp_feedback_list_desc: str,
        fb_to_sota_exp: ExperimentFeedback | None = None,
    ) -> DSExperiment:
        if pipeline:
            component_info = get_component("Pipeline")
        else:
            component_info = get_component(hypotheses[0].component)
        data_folder_info = self.scen.processed_data_folder_description
        workflow_check = not pipeline and hypotheses[0].component != "Workflow"
        sys_prompt = T(".prompts_v2:task_gen.system").r(
            task_output_format=component_info["task_output_format"] if not self.supports_response_schema else None,
            component_desc=component_desc,
            workflow_check=workflow_check,
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
            json_target_type=Dict[str, str | Dict[str, str]] if not self.supports_response_schema else None,
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
        exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypotheses[0])
        if sota_exp is not None:
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp.experiment_workspace)

        # 3) create the workflow update task
        if workflow_check:
            workflow_task = WorkflowTask(
                name="Workflow",
                description=task_dict.get("workflow_update", "No update needed"),
            )
            exp.pending_tasks_list.append([workflow_task])
        return exp

    def get_scenario_all_desc(self, trace: DSTrace, eda_output=None) -> str:
        return T(".prompts_v2:scenario_description").r(
            background=trace.scen.background,
            submission_specifications=trace.scen.submission_specifications,
            evaluation=trace.scen.metric_description,
            metric_name=trace.scen.metric_name,
            metric_direction=trace.scen.metric_direction,
            raw_description=trace.scen.raw_description,
            use_raw_description=DS_RD_SETTING.use_raw_description,
            time_limit=f"{DS_RD_SETTING.full_timeout / 60 / 60 : .2f} hours",
            eda_output=eda_output,
        )

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
                )
            )
        return result

    def gen(
        self,
        trace: DSTrace,
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
        scenario_desc = self.get_scenario_all_desc(trace, eda_output=eda_output)

        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp, heading="Best of previous exploration of the scenario"
        )

        exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="all"),
            type="all",
            pipeline=pipeline,
        )
        failed_exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="failed"),
            type="failed",
            pipeline=pipeline,
        )

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

        # Step 1: Identify problems
        all_problems = self.identify_problem(
            current_sub_trace=trace.get_parent_exps(),
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            exp_feedback_list_desc=exp_feedback_list_desc,
            inject_diverse=inject_diverse,
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

        # Step 3: Select the best hypothesis
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
                [new_hypothesis] if len(trace.hist) > 0 else self.get_all_hypotheses(all_problems, hypothesis_dict)
            ),
            pipeline=pipeline,
            failed_exp_feedback_list_desc=failed_exp_feedback_list_desc,
            fb_to_sota_exp=fb_to_sota_exp,
        )
