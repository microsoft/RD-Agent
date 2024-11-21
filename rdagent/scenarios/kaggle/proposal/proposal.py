import json
import math
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.components.coder.model_coder.model import ModelExperiment, ModelTask
from rdagent.components.knowledge_management.vector_base import VectorBase
from rdagent.components.proposal import (
    FactorAndModelHypothesis2Experiment,
    FactorAndModelHypothesisGen,
)
from rdagent.core.exception import ModelEmptyError
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.kaggle.experiment.kaggle_experiment import (
    KG_MODEL_MAPPING,
    KG_SELECT_MAPPING,
    KGFactorExperiment,
    KGModelExperiment,
)
from rdagent.scenarios.kaggle.experiment.scenario import KGScenario
from rdagent.scenarios.kaggle.knowledge_management.graph import KGKnowledgeGraph
from rdagent.scenarios.kaggle.knowledge_management.vector_base import (
    KaggleExperienceBase,
)

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


from rdagent.scenarios.kaggle.experiment.scenario import (
    KG_ACTION_FEATURE_ENGINEERING,
    KG_ACTION_FEATURE_PROCESSING,
    KG_ACTION_LIST,
    KG_ACTION_MODEL_FEATURE_SELECTION,
    KG_ACTION_MODEL_TUNING,
)


class KGHypothesis(Hypothesis):
    def __init__(
        self,
        hypothesis: str,
        reason: str,
        concise_reason: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        action: str,
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation, concise_justification, concise_knowledge
        )
        self.action = action

    def __str__(self) -> str:
        return f"""Chosen Action: {self.action}
Hypothesis: {self.hypothesis}
Reason: {self.reason}
Concise Reason & Knowledge: {self.concise_reason}
Concise Observation: {self.concise_observation}
Concise Justification: {self.concise_justification}
Concise Knowledge: {self.concise_knowledge}
"""


def generate_RAG_content(
    scen: KGScenario,
    trace: Trace,
    hypothesis_and_feedback: str,
    target: str = None,
    chosen_hypothesis: str = None,
    chosen_hypothesis_type: str = None,
) -> str:
    if scen.if_using_vector_rag:
        if scen.mini_case:
            rag_results, _ = scen.vector_base.search_experience(target, hypothesis_and_feedback, topk_k=1)
        else:
            rag_results, _ = scen.vector_base.search_experience(target, hypothesis_and_feedback, topk_k=5)
        return "\n".join([doc.content for doc in rag_results])
    if scen.if_using_graph_rag is False or trace.knowledge_base is None:
        return None
    same_competition_node = trace.knowledge_base.get_node_by_content(trace.scen.get_competition_full_desc())
    if same_competition_node is not None:
        related_hypothesis_nodes = []
        for action in KG_ACTION_LIST:
            related_hypothesis_nodes.extend(
                trace.knowledge_base.get_nodes_within_steps(
                    start_node=same_competition_node,
                    steps=1,
                    constraint_labels=[action],
                )[:1]
            )
    else:
        related_hypothesis_nodes = []
    experiences = []
    for hypothesis_node in related_hypothesis_nodes:
        experience = {"hypothesis": hypothesis_node.content}
        experiment_node_list = trace.knowledge_base.get_nodes_within_steps(
            start_node=hypothesis_node, steps=1, constraint_labels=["experiments"]
        )
        if len(experiment_node_list) > 0:
            experience["experiments"] = experiment_node_list[0].content
        else:
            experience["experiments"] = "No experiment information available."
        conclusion_node_list = trace.knowledge_base.get_nodes_within_steps(
            start_node=hypothesis_node, steps=1, constraint_labels=["conclusion"]
        )
        if len(conclusion_node_list) > 0:
            experience["conclusion"] = conclusion_node_list[0].content
        else:
            experience["conclusion"] = "No conclusion information available."
        experiences.append(experience)

    found_nodes = []
    insights = []
    if chosen_hypothesis is not None:
        similar_nodes = trace.knowledge_base.semantic_search(
            node=chosen_hypothesis,
            topk_k=2,
        )

        for similar_node in similar_nodes:
            hypothesis_nodes = trace.knowledge_base.get_nodes_within_steps(
                start_node=similar_node,
                steps=3,
                constraint_labels=[chosen_hypothesis_type],
            )
            found_nodes.extend(hypothesis_nodes[:5])

        found_nodes = sorted(list(set(found_nodes)), key=lambda x: len(x.content))

        for exp_node in found_nodes[:5]:
            insight = {"experiments": exp_node.content}
            hypothesis_node_list = trace.knowledge_base.get_nodes_within_steps(
                start_node=exp_node, steps=2, constraint_labels=KG_ACTION_LIST
            )
            if len(hypothesis_node_list) > 0:
                insight["hypothesis"] = hypothesis_node_list[0].content
            else:
                insight["hypothesis"] = "No hypothesis information available."
            conclusion_node_list = trace.knowledge_base.get_nodes_within_steps(
                start_node=exp_node, steps=2, constraint_labels=["conclusion"]
            )
            if len(conclusion_node_list) > 0:
                insight["conclusion"] = conclusion_node_list[0].content
            else:
                insight["conclusion"] = "No conclusion information available."
            insights.append(insight)
    else:
        similar_nodes = trace.knowledge_base.semantic_search(
            node=trace.scen.get_competition_full_desc(),
            topk_k=2,
        )

        for similar_node in similar_nodes:
            for hypothesis_type in KG_ACTION_LIST:
                hypothesis_nodes = trace.knowledge_base.get_nodes_within_steps(
                    start_node=similar_node,
                    steps=3,
                    constraint_labels=[hypothesis_type],
                )
                found_nodes.extend(hypothesis_nodes[:2])

        found_nodes = sorted(list(set(found_nodes)), key=lambda x: len(x.content))

        for hypothesis_node in found_nodes[:5]:
            if hypothesis_node in related_hypothesis_nodes:
                continue
            insight = {"hypothesis": hypothesis_node.content}
            experiment_node_list = trace.knowledge_base.get_nodes_within_steps(
                start_node=hypothesis_node, steps=2, constraint_labels=["experiments"]
            )
            if len(experiment_node_list) > 0:
                insight["experiments"] = experiment_node_list[0].content
            else:
                insight["experiments"] = "No experiment information available."
            conclusion_node_list = trace.knowledge_base.get_nodes_within_steps(
                start_node=hypothesis_node, steps=2, constraint_labels=["conclusion"]
            )
            if len(conclusion_node_list) > 0:
                insight["conclusion"] = conclusion_node_list[0].content
            else:
                insight["conclusion"] = "No conclusion information available."
            insights.append(insight)

    RAG_content = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["KG_hypothesis_gen_RAG"])
        .render(insights=insights, experiences=experiences)
    )
    return RAG_content


class KGHypothesisGen(FactorAndModelHypothesisGen):
    """
    # NOTE: we can share this class across different data mining scenarios
    # It may better to move the class into components folder like `rdagent/components/proposal/model_proposal.py`
    # Here is the use case:

    .. code-block:: python

        class KGHypothesisGen(ModelHypothesisGen):
            prompts: Prompts = a_specifc_prompt_dict
    """

    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def update_reward_estimates(self, trace: Trace) -> None:
        if len(trace.hist) > 0:
            last_entry = trace.hist[-1]
            last_action = last_entry[0].action
            last_result = last_entry[1].result
            # Extract performance_t
            performance_t = last_result.get("performance", 0.0)
            # Get performance_{t-1}
            if len(trace.hist) > 1:
                prev_entry = trace.hist[-2]
                prev_result = prev_entry[1].result
                performance_t_minus_1 = prev_result.get("performance", 0.0)
            else:
                performance_t_minus_1 = self.scen.initial_performance

            if self.scen.evaluation_metric_direction:
                reward = (performance_t - performance_t_minus_1) / max(performance_t_minus_1, 1e-8)
            else:
                reward = (performance_t_minus_1 - performance_t) / max(performance_t_minus_1, 1e-8)

            reward = (performance_t - performance_t_minus_1) / performance_t_minus_1
            n_o = self.scen.action_counts[last_action]
            mu_o = self.scen.reward_estimates[last_action]
            self.scen.scen.reward_estimates[last_action] += (reward - mu_o) / n_o
        else:
            # First iteration, nothing to update
            pass

    def execute_next_action(self, trace: Trace) -> str:
        actions = list(self.scen.action_counts.keys())
        t = sum(self.scen.action_counts.values()) + 1

        # If any action has not been tried yet, select it
        for action in actions:
            if self.scen.action_counts[action] == 0:
                selected_action = action
                return selected_action

        c = self.scen.confidence_parameter
        ucb_values = {}
        for action in actions:
            mu_o = self.scen.reward_estimates[action]
            n_o = self.scen.action_counts[action]
            ucb = mu_o + c * math.sqrt(math.log(t) / n_o)
            ucb_values[action] = ucb
        # Select action with highest UCB
        selected_action = max(ucb_values, key=ucb_values.get)

        return selected_action

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        if self.scen.if_action_choosing_based_on_UCB:
            action = self.execute_next_action(trace)

        hypothesis_specification = f"Hypothesis should avoid being too general and vague, and should be specific and actionable. For example, hypothesis like 'tune a model' is too general, while hypothesis like 'increase the learning rate to 0.1 of the lightgbm model will improve the performance' is specific and actionable."
        if len(trace.hist) > 0:
            sota_features = str(trace.hist[-1][1].based_experiments[-1].experiment_workspace.data_description)
            sota_models = json.dumps(
                trace.hist[-1][1].based_experiments[-1].experiment_workspace.model_description, indent=2
            )
            sota_result = trace.hist[-1][1].based_experiments[-1].result
            hypothesis_specification += f"\nYour hypothesis should based on current SOTA solution. The user will conduct experiments based on the SOTA solution to test whether your hypothesis is right on this specific ecompetition. \n\nSOTA Features: {sota_features}\n\nSOTA Models: {sota_models}\n\nSOTA Result: {sota_result}"
        if self.scen.if_action_choosing_based_on_UCB:
            hypothesis_specification += (
                "\n\nNext experiment action is "
                + action
                + "\nspecification: "
                + prompt_dict["hypothesis_specification"][action]
            )

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": generate_RAG_content(
                scen=self.scen,
                trace=trace,
                hypothesis_and_feedback=hypothesis_and_feedback,
                target=action if self.scen.if_action_choosing_based_on_UCB else None,
            ),
            "hypothesis_output_format": prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": hypothesis_specification,
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)

        hypothesis = KGHypothesis(
            hypothesis=response_dict.get("hypothesis", "Hypothesis not provided"),
            reason=response_dict.get("reason", "Reason not provided"),
            concise_reason=response_dict.get("concise_reason", "Concise reason not provided"),
            concise_observation=response_dict.get("concise_observation", "Concise observation not provided"),
            concise_justification=response_dict.get("concise_justification", "Concise justification not provided"),
            concise_knowledge=response_dict.get("concise_knowledge", "Concise knowledge not provided"),
            action=response_dict.get("action", "Action not provided"),
        )

        return hypothesis


class KGHypothesis2Experiment(FactorAndModelHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        scenario = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
        assert isinstance(hypothesis, KGHypothesis)
        experiment_output_format = (
            prompt_dict["feature_experiment_output_format"]
            if hypothesis.action in [KG_ACTION_FEATURE_ENGINEERING, KG_ACTION_FEATURE_PROCESSING]
            else prompt_dict["model_experiment_output_format"]
        )
        self.current_action = hypothesis.action

        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        experiment_list: List[ModelExperiment] = [t[1] for t in trace.hist]

        model_list = []
        for experiment in experiment_list:
            for sub_task in experiment.sub_tasks:
                model_list.extend(sub_task.get_task_information())

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": model_list,
            "RAG": generate_RAG_content(
                trace.scen,
                trace,
                hypothesis_and_feedback,
                chosen_hypothesis=hypothesis.hypothesis,
                chosen_hypothesis_type=hypothesis.action,
            ),
        }, True

    def convert_feature_experiment(self, response: str, trace: Trace) -> KGFactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = (response_dict[factor_name].get("description", "Factor description not provided"),)
            formulation = (response_dict[factor_name].get("formulation", "Factor formulation not provided"),)
            variables = (response_dict[factor_name].get("variables", "Variables not provided"),)
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    variables=variables,
                    version=2,
                )
            )

        exp = KGFactorExperiment(
            sub_tasks=tasks,
            based_experiments=(
                [KGFactorExperiment(sub_tasks=[], source_feature_size=trace.scen.input_shape[-1])]
                + [t[1] for t in trace.hist if t[2]]
            ),
        )
        return exp

    def convert_model_experiment(self, response: str, trace: Trace) -> KGModelExperiment:
        response_dict = json.loads(response)
        tasks = []
        model_type = response_dict.get("model_type", "Model type not provided")
        if not isinstance(model_type, str) or model_type not in KG_SELECT_MAPPING:
            raise ModelEmptyError(
                f"Invalid model type '{model_type}'. Allowed model types are: {', '.join(KG_SELECT_MAPPING)}."
            )

        based_experiments = [KGModelExperiment(sub_tasks=[], source_feature_size=trace.scen.input_shape[-1])] + [
            t[1] for t in trace.hist if t[2]
        ]
        model_type = response_dict.get("model_type", "Model type not provided")
        if model_type in KG_MODEL_MAPPING:
            base_code = based_experiments[-1].experiment_workspace.code_dict.get(KG_MODEL_MAPPING[model_type], None)
        else:
            base_code = None

        tasks.append(
            ModelTask(
                name=response_dict.get("model_name", "Model name not provided"),
                description=response_dict.get("description", "Description not provided"),
                architecture=response_dict.get("architecture", "Architecture not provided"),
                hyperparameters=response_dict.get("hyperparameters", "Hyperparameters not provided"),
                model_type=model_type,
                version=2,
                base_code=base_code,
            )
        )
        exp = KGModelExperiment(
            sub_tasks=tasks,
            based_experiments=based_experiments,
        )
        return exp

    def convert_response(self, response: str, trace: Trace) -> ModelExperiment:
        if self.current_action in [KG_ACTION_FEATURE_ENGINEERING, KG_ACTION_FEATURE_PROCESSING]:
            return self.convert_feature_experiment(response, trace)
        elif self.current_action in [KG_ACTION_MODEL_FEATURE_SELECTION, KG_ACTION_MODEL_TUNING]:
            return self.convert_model_experiment(response, trace)


class KGTrace(Trace[KGScenario, KGKnowledgeGraph]):
    pass
