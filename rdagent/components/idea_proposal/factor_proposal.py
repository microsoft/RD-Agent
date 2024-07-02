from abc import abstractmethod
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from rdagent.core.prompts import Prompts
from rdagent.core.proposal import (
    Hypothesis,
    Hypothesis2Task,
    HypothesisGen,
    Scenario,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


FactorHypothesis = Hypothesis


class FactorHypothesisGen(HypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.gen_context_flag = False
        self.gen_context_dict = None
        self.gen_json_flag = False

    # The following methods are scenario related so they should be implemented in the subclass
    @abstractmethod
    def prepare_gen_context(self, trace: Trace) -> None: ...

    @abstractmethod
    def gen_response_to_hypothesis_list(self, response: str) -> FactorHypothesis: ...

    def gen(self, trace: Trace) -> FactorHypothesis:
        assert self.gen_context_flag, "Please call prepare_gen_context before calling gen."
        self.gen_context_flag = False  # reset the flag

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["factor_hypothesis_gen"]["system_prompt"])
            .render(scenario=self.scen.get_scenario_all_desc())
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["factor_hypothesis_gen"]["user_prompt"])
            .render(self.gen_context_dict)
        )

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=self.gen_json_flag
        )

        hypothesis = self.gen_response_to_hypothesis_list(resp)

        return hypothesis


class FactorHypothesis2Task(Hypothesis2Task):
    def convert(self, bs: FactorHypothesis) -> None: ...
