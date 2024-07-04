from abc import abstractmethod
from pathlib import Path
from typing import Tuple

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.factor import FactorExperiment
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import (
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    HypothesisSet,
    Scenario,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


FactorHypothesis = Hypothesis


class FactorHypothesisGen(HypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)

    # The following methods are scenario related so they should be implemented in the subclass
    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        ...

    @abstractmethod
    def convert_response(self, response: str) -> FactorHypothesis:
        ...

    def gen(self, trace: Trace) -> FactorHypothesis:
        context_dict, json_flag = self.prepare_context(trace)

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["factor_hypothesis_gen"]["system_prompt"])
            .render(
                scenario=self.scen.get_scenario_all_desc(),
                hypothesis_output_format=context_dict["hypothesis_output_format"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["factor_hypothesis_gen"]["user_prompt"])
            .render(
                hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                RAG=context_dict["RAG"],
            )
        )

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)

        hypothesis = self.convert_response(resp)

        return hypothesis


class FactorHypothesis2Experiment(Hypothesis2Experiment[FactorExperiment]):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prepare_context(self, hs: HypothesisSet) -> Tuple[dict, bool]:
        ...

    @abstractmethod
    def convert_response(self, response: str) -> FactorExperiment:
        ...

    def convert(self, hs: HypothesisSet) -> FactorExperiment:
        context, json_flag = self.prepare_context(hs)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["factor_hypothesis2experiment"]["system_prompt"])
            .render(
                scenario=hs.trace.scen.get_scenario_all_desc(),
                experiment_output_format=context["experiment_output_format"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["factor_hypothesis2experiment"]["user_prompt"])
            .render(
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                factor_list=context["factor_list"],
                RAG=context["RAG"],
            )
        )

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)

        return self.convert_response(resp)
