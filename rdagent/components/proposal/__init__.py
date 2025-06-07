from abc import abstractmethod
from typing import Tuple

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import (
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    Scenario,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry


class LLMHypothesisGen(HypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)

    # The following methods are scenario related so they should be implemented in the subclass
    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str) -> Hypothesis: ...

    def gen(self, trace: Trace) -> Hypothesis:
        context_dict, json_flag = self.prepare_context(trace)

        system_prompt = T(".prompts:hypothesis_gen.system_prompt").r(
            targets=self.targets,
            scenario=(
                self.scen.get_scenario_all_desc(filtered_tag=self.targets)
                if self.targets in ["factor", "model"]
                else self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
            ),
            hypothesis_output_format=context_dict["hypothesis_output_format"],
            hypothesis_specification=context_dict["hypothesis_specification"],
        )
        user_prompt = T(".prompts:hypothesis_gen.user_prompt").r(
            targets=self.targets,
            hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
            last_hypothesis_and_feedback=(
                context_dict["last_hypothesis_and_feedback"] if "last_hypothesis_and_feedback" in context_dict else ""
            ),
            sota_hypothesis_and_feedback=(
                context_dict["sota_hypothesis_and_feedback"] if "sota_hypothesis_and_feedback" in context_dict else ""
            ),
            RAG=context_dict["RAG"],
        )

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, str]
        )

        hypothesis = self.convert_response(resp)

        return hypothesis


class FactorHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "factors"


class ModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "model tuning"


class FactorAndModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "feature engineering and model building"


class LLMHypothesis2Experiment(Hypothesis2Experiment[Experiment]):
    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> Experiment: ...

    @wait_retry(retry_n=5)
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        context, json_flag = self.prepare_context(hypothesis, trace)
        system_prompt = T(".prompts:hypothesis2experiment.system_prompt").r(
            targets=self.targets,
            scenario=trace.scen.get_scenario_all_desc(filtered_tag=self.targets),
            experiment_output_format=context["experiment_output_format"],
        )
        user_prompt = T(".prompts:hypothesis2experiment.user_prompt").r(
            targets=self.targets,
            target_hypothesis=context["target_hypothesis"],
            hypothesis_and_feedback=(
                context["hypothesis_and_feedback"] if "hypothesis_and_feedback" in context else ""
            ),
            last_hypothesis_and_feedback=(
                context["last_hypothesis_and_feedback"] if "last_hypothesis_and_feedback" in context else ""
            ),
            sota_hypothesis_and_feedback=(
                context["sota_hypothesis_and_feedback"] if "sota_hypothesis_and_feedback" in context else ""
            ),
            target_list=context["target_list"],
            RAG=context["RAG"],
        )

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, dict[str, str | dict]]
        )

        return self.convert_response(resp, hypothesis, trace)


class FactorHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "factors"


class ModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "model tuning"


class FactorAndModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "feature engineering and model building"
