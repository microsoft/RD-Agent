import json
from pathlib import Path
from typing import List, Tuple
import math

from jinja2 import Environment, StrictUndefined

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.components.coder.model_coder.model import ModelExperiment, ModelTask
from rdagent.components.knowledge_management.vector_base import VectorBase
from rdagent.components.proposal.model_proposal import (
    ModelHypothesis,
    ModelHypothesis2Experiment,
    ModelHypothesisGen,
)
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.kaggle.experiment.kaggle_experiment import (
    KGFactorExperiment,
    KGModelExperiment,
)
from rdagent.scenarios.kaggle.knowledge_management.vector_base import (
    KaggleExperienceBase,
)

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


KG_ACTION_FEATURE_ENGINEERING = "Feature engineering"
KG_ACTION_FEATURE_PROCESSING = "Feature processing"
KG_ACTION_MODEL_FEATURE_SELECTION = "Model feature selection"
KG_ACTION_MODEL_TUNING = "Model tuning"
KG_ACTION_LIST = [
    KG_ACTION_FEATURE_ENGINEERING,
    KG_ACTION_FEATURE_PROCESSING,
    KG_ACTION_MODEL_FEATURE_SELECTION,
    KG_ACTION_MODEL_TUNING,
]


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


class KGHypothesisGen(ModelHypothesisGen):
    """
    # NOTE: we can share this class across different data mining scenarios
    # It may better to move the class into components folder like `rdagent/components/proposal/model_proposal.py`
    # Here is the use case:

    .. code-block:: python

        class KGHypothesisGen(ModelHypothesisGen):
            prompts: Prompts = a_specifc_prompt_dict
    """

    def __init__(self, scen: Scenario, knowledge: VectorBase = None) -> Tuple[dict, bool]:
        super().__init__(scen)
        self.scen.vector_base.save(KAGGLE_IMPLEMENT_SETTING.rag_path)
        self.action_counts = {
            "Feature engineering": 0,
            "Feature processing": 0,
            "Model feature selection": 0,
            "Model tuning": 0
        }
        self.reward_estimates = {
            "Feature engineering": 0,
            "Feature processing": 0,
            "Model feature selection": 0,
            "Model tuning": 0
        }

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

        rag_results, _ = self.scen.vector_base.search_experience(hypothesis_and_feedback, topk_k=5)
        rag_content = "\n".join([doc.content for doc in rag_results])

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": None,
        }
        return context_dict, True

    def calculate_reward(self, previous_performance, current_performance):
        return previous_performance - current_performance
    
    def update_reward_estimates(self, action, reward, action_counts, reward_estimates):
        n_o = action_counts[action]
        if n_o == 0:
            reward_estimates[action] = reward
        else:
            reward_estimates[action] += (reward - reward_estimates[action]) / n_o

    def select_next_action(self, t, action_counts, reward_estimates, c=1.0):
        best_action = None
        best_ucb_value = -float('inf')

        for action in reward_estimates.keys():
            n_o = action_counts[action]
            if n_o == 0:
                return action
            else:
                ucb_value = reward_estimates[action] + c * math.sqrt(math.log(t + 1) / n_o)
                if ucb_value > best_ucb_value:
                    best_ucb_value = ucb_value
                    best_action = action

        return best_action

    def execute_next_action(self):
        trace = self.scen.trace
        t = len(trace.hist)
        next_action = self.select_next_action(t, self.action_counts, self.reward_estimates)

        previous_performance = trace.hist[t - 1][1].result if t > 0 else None
        trace.hist[t][0].action = next_action
        current_performance = trace.hist[t][1].result

        reward = self.calculate_reward(previous_performance, current_performance)
        self.action_counts[next_action] += 1
        self.update_reward_estimates(next_action, reward, self.action_counts, self.reward_estimates)
        
        return next_action


    def convert_response(self, response: str) -> ModelHypothesis:
        response_dict = json.loads(response)

        action = self.execute_next_action()

        hypothesis = KGHypothesis(
            hypothesis=response_dict["hypothesis"],
            reason=response_dict["reason"],
            concise_reason=response_dict["concise_reason"],
            concise_observation=response_dict["concise_observation"],
            concise_justification=response_dict["concise_justification"],
            concise_knowledge=response_dict["concise_knowledge"],
            action=action,
        )
        return hypothesis


class KGHypothesis2Experiment(ModelHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        scenario = trace.scen.get_scenario_all_desc()
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
            model_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": model_list,
            "RAG": ...,
        }, True

    def convert_feature_experiment(self, response: str, trace: Trace) -> KGFactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            variables = response_dict[factor_name]["variables"]
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    variables=variables,
                    version=2,
                )
            )

        exp = KGFactorExperiment(tasks)
        exp.based_experiments = [KGFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]
        return exp

    def convert_model_experiment(self, response: str, trace: Trace) -> KGModelExperiment:
        response_dict = json.loads(response)
        tasks = []
        tasks.append(
            ModelTask(
                name=response_dict["model_name"],
                description=response_dict["description"],
                architecture=response_dict["architecture"],
                hyperparameters=response_dict["hyperparameters"],
                model_type=response_dict["model_type"],
                version=2,
            )
        )
        exp = KGModelExperiment(tasks)
        exp.based_experiments = [t[1] for t in trace.hist if t[2]]
        return exp

    def convert_response(self, response: str, trace: Trace) -> ModelExperiment:
        if self.current_action in [KG_ACTION_FEATURE_ENGINEERING, KG_ACTION_FEATURE_PROCESSING]:
            return self.convert_feature_experiment(response, trace)
        elif self.current_action in [KG_ACTION_MODEL_FEATURE_SELECTION, KG_ACTION_MODEL_TUNING]:
            return self.convert_model_experiment(response, trace)
